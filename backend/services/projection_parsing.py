"""Parsing an uploaded projection file (.csv, .xlsx or .xls) into the canonical column set.

Extracted from build_agent so the pipeline orchestrator and the file-format concerns —
alias mapping, spreadsheet signature sniffing, charset detection, ratio-cell volume
recovery — each live in one place. parse_projection_upload is the entry point;
CORE_PROJECTION_COLUMNS is public because the upload route derives its reportable-stats
warning from it.
"""

from __future__ import annotations

import codecs
import io
import logging

import charset_normalizer
import pandas as pd

# The per-game stats a projection file CAN carry. A file need not carry all of them:
# sources that pair a projection set with a league export only that league's active
# categories, so several may legitimately be absent. Recognition therefore asks only for
# Player, Position, and _MIN_MATCHED_CORE_COLUMNS of these — still unmistakably a
# projection file, while a spreadsheet of something else entirely maps ~nothing and is
# rejected with a clear message instead of failing much later, deep in the blend, as a
# baffling "0 players available" error.
CORE_PROJECTION_COLUMNS = ('Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Threes')
_MIN_MATCHED_CORE_COLUMNS = 3

_COLUMN_ALIASES_KEY = 'projection-column-aliases'


# A .xlsx is a ZIP archive, so every one begins with this signature. The older .xls is an
# OLE2 compound file with a different one, and is read with xlrd: Basketball Monster's export,
# among others, is a genuine .xls (written by the MyXls library), not HTML under that name.
_XLSX_SIGNATURE = b'PK\x03\x04'


_XLS_SIGNATURE  = b'\xd0\xcf\x11\xe0'


# A name split over two columns, joined into 'Player'.
_SPLIT_NAME_COLUMNS = {'First Name', 'Last Name'}

# Columns a file may lack but carry the parts of. Each is built only when the file has no column
# of its own: rebounds from their two halves, points from the makes (two per field goal, one more
# per three, one per free throw), and each percentage from its makes and attempts.
_DERIVED_CORES = {'Rebounds': {'Off Rebounds', 'Def Rebounds'},
                  'Points': {'Field Goals Made', 'Threes', 'Free Throws Made'}}
_DERIVED_SHARES = {'Field Goal %': ('Field Goals Made', 'Field Goal Attempts'),
                   'Free Throw %': ('Free Throws Made', 'Free Throw Attempts'),
                   'Three %': ('Threes', 'Three Attempts')}

# SEASON TOTALS. Some sources project a season's totals, not per-game numbers. No per-game
# projection can exceed a single game's record, so one value above these ceilings settles that
# the whole file is totals, to be divided by games played. Minutes decide it when the file has
# them (nobody averages an hour a game); otherwise any counting stat does.
_MAX_MINUTES_PER_GAME = 60.0
_MAX_PER_GAME = {'Points': 100.0, 'Rebounds': 60.0, 'Assists': 35.0, 'Steals': 15.0, 'Blocks': 20.0,
                 'Threes': 20.0, 'Turnovers': 20.0, 'Field Goal Attempts': 70.0, 'Free Throw Attempts': 40.0}


# Attempts hiding inside a ratio cell: sources that print a percentage with its makes and
# attempts behind it — "0.583 (10.2/17.5)" — and ship no attempts column of their own.
# Captures the second number, the attempts.
_ATTEMPTS_IN_RATIO_CELL_PATTERN = r'\(\s*-?[\d.]+\s*/\s*(-?[\d.]+)\s*\)'


def parse_projection_upload(upload_bytes: bytes, sport_params: dict) -> pd.DataFrame:
    """Parse an uploaded projection file (.csv, .xlsx or .xls) into the canonical column set.

    There is no format detection: each column is interpreted on its own through the alias
    table (see 'projection-column-aliases' in parameters.yaml), so any source is readable
    as long as its header spellings are known, and a file already written in canonical
    names needs no aliases at all. Raises ValueError naming what could not be found when
    the file does not read as a projection set.
    """
    df_raw = _read_projection_table(upload_bytes)
    column_mapping  = _map_columns_to_canonical(df_raw, sport_params)
    # Canonical names are aliases of themselves, so this covers a file already written in them.
    renamed_columns = set(column_mapping.values())

    # A player is named by one column or by first and last name split over two. Position is
    # optional: positions come from the canonical eligibility (data_retrieval), so a file
    # without them loses nothing but the fallback for players that table does not know.
    has_player_name = 'Player' in renamed_columns or _SPLIT_NAME_COLUMNS <= renamed_columns
    derivable_cores = {core for core, parts in _DERIVED_CORES.items() if parts <= renamed_columns}
    matched_cores    = [column for column in CORE_PROJECTION_COLUMNS
                        if column in renamed_columns or column in derivable_cores]
    if has_player_name and len(matched_cores) >= _MIN_MATCHED_CORE_COLUMNS:
        return _parse_with_renamer(df_raw, column_mapping, sport_params)

    if not has_player_name:
        problem = 'no column for Player (nor First Name and Last Name)'
    else:
        missing_cores = [column for column in CORE_PROJECTION_COLUMNS
                         if column not in renamed_columns and column not in derivable_cores]
        problem = (f'only {len(matched_cores)} of {len(CORE_PROJECTION_COLUMNS)} core stats '
                   f"were recognized (no {', '.join(missing_cores)})")
    unrecognized = [column for column in df_raw.columns if column not in column_mapping]
    raise ValueError(
        f'File does not read as a projection set: {problem}. Headers that were not '
        f"recognized: {', '.join(map(str, unrecognized))}. Add their spellings to "
        f'{_COLUMN_ALIASES_KEY} to teach the parser this source.'
    )


def _read_projection_table(upload_bytes: bytes) -> pd.DataFrame:
    """Load an upload into a frame, whether it is a spreadsheet or a text file.

    People reach these tools by copying a projection table into a spreadsheet, so the file
    that comes back is as often .xlsx as .csv — and telling someone to re-export as CSV is
    asking them to do work the parser can do. Format is decided by the file's own signature
    rather than its name, because a download saved with the wrong extension is common and the
    bytes cannot lie."""
    if upload_bytes.startswith(_XLSX_SIGNATURE):
        try:
            return pd.read_excel(io.BytesIO(upload_bytes), engine='openpyxl')
        except Exception as exc:
            raise ValueError(f'Could not read this spreadsheet: {type(exc).__name__}: {exc}')
    if upload_bytes.startswith(_XLS_SIGNATURE):
        try:
            return pd.read_excel(io.BytesIO(upload_bytes), engine='xlrd')
        except Exception as exc:
            raise ValueError(f'Could not read this spreadsheet: {type(exc).__name__}: {exc}')
    return pd.read_csv(io.StringIO(_decode_projection_text(upload_bytes)))


def _decode_projection_text(csv_bytes: bytes) -> str:
    """Decode an uploaded text file whatever it was saved as.

    Spreadsheets export in whatever codepage the machine defaults to, so a file that opens
    fine locally can be UTF-8, UTF-16 (Excel's "Unicode text"), or a legacy Windows codepage.
    Assuming UTF-8 made those fail on the first non-ASCII byte — a decode error naming a byte
    offset, which tells a user nothing except that saving it again as UTF-8 helps.

    Order matters. A byte-order mark is decisive, so it is honoured first; UTF-16 in
    particular must never be guessed at, since its text decodes as plausible-looking rubbish
    under a single-byte codepage. UTF-8 is tried next because it is both the common case and
    self-validating — invalid sequences raise rather than silently mis-decode. Only then do we
    detect, which is what gets accented names right: Jokić and Dončić live in Central European
    codepages that a blind cp1252 fallback would mangle, and a mangled name resolves to no
    player id. latin-1 is the floor: it cannot raise, so a file always loads.
    """
    if csv_bytes.startswith(codecs.BOM_UTF16_LE) or csv_bytes.startswith(codecs.BOM_UTF16_BE):
        return csv_bytes.decode('utf-16')
    try:
        return csv_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass

    detected = charset_normalizer.from_bytes(csv_bytes).best()
    if detected is not None:
        logging.getLogger('fbbo').info(
            'Projection upload is not UTF-8; decoded as %s', detected.encoding)
        return str(detected)
    logging.getLogger('fbbo').warning(
        'Projection upload encoding could not be identified; decoding as latin-1, '
        'so accented names may be wrong')
    return csv_bytes.decode('latin-1')


def _map_columns_to_canonical(df_raw: pd.DataFrame, sport_params: dict) -> dict:
    """{column in the file: canonical name} for every column the aliases recognize.

    Every canonical name is an alias of itself, folded like the rest, so 'off rebounds' reads
    as 'Off Rebounds' without the table having to list it. Columns that match nothing are left
    out (the parse drops them). A canonical name that two of the file's columns both claim is
    taken by the first, so a file carrying e.g. both 'PTS' and 'Points' cannot produce a
    duplicate column label downstream.
    """
    alias_table = sport_params.get(_COLUMN_ALIASES_KEY, {})
    aliases = {_normalize_projection_header(canonical): canonical for canonical in alias_table.values()}
    aliases.update({_normalize_projection_header(alias): canonical
                    for alias, canonical in alias_table.items()})
    mapping, claimed = {}, set()
    for column in df_raw.columns:
        canonical = aliases.get(_normalize_projection_header(column))
        if canonical is not None and canonical not in claimed:
            mapping[column] = canonical
            claimed.add(canonical)
    return mapping


def _normalize_projection_header(header) -> str:
    """Header spellings differ only by case, padding and word separators far more often than by
    wording, so both sides of an alias lookup are folded to one form: lower case, underscores
    read as spaces (field_goals_attempted is 'field goals attempted'), runs of spaces as one."""
    return ' '.join(str(header).replace('_', ' ').lower().split())


def _parse_with_renamer(
    df_raw: pd.DataFrame
    , column_mapping: dict
    , sport_params: dict
) -> pd.DataFrame:
    """Rename this file's columns to canonical names, drop junk, coerce stats."""
    # Keep only the columns the mapping claimed. Unmapped extras (ranks, dollar values,
    # minutes, ...) would otherwise join the blend's column union, where every player from
    # the OTHER sources is "missing" them — and the blend drops any player missing any column
    # across all sources, so a single junk column can wipe out the entire pool. A second
    # column for an already-claimed name (both 'ORB' and 'Off Rebounds') is unmapped too, so
    # it is dropped here rather than becoming a duplicate label.
    df = df_raw[list(column_mapping)].rename(columns=column_mapping).copy()

    # Before the ratio cells are reduced to their leading number below, mine them for any
    # attempts column the file does not carry separately.
    df = _recover_volumes_from_ratio_cells(df, sport_params)

    # Sources carry non-numeric stat values: some repeat the header row inside the table
    # body (every stat cell a string), and some format ratio stats as
    # "0.474 (5.2/11.0)". Extract the leading number where a stat column holds strings,
    # then drop rows with no numeric stats at all — those are the embedded header/junk rows.
    def coerce_stat_column(column: pd.Series) -> pd.Series:
        if column.dtype == object:
            # Leading number only, and not one embedded in a word — the repeated header
            # rows contain cells like "3PM", which must NOT read as the number 3.
            column = column.astype(str).str.extract(
                r'^\s*(-?(?:\d+\.?\d*|\.\d+))(?![A-Za-z])', expand=False)
        return pd.to_numeric(column, errors='coerce')

    identity_columns = {'Player', 'Position'} | _SPLIT_NAME_COLUMNS
    stat_columns = [column for column in df.columns if column not in identity_columns]
    df[stat_columns] = df[stat_columns].apply(coerce_stat_column)
    df = df.dropna(subset=stat_columns, how='all')
    df = _join_split_name(df)
    df = _derive_missing_columns(df)
    df = _convert_season_totals_to_per_game(df, sport_params)
    df = convert_percent_columns_to_fractions(df)

    if 'Games Played %' not in df.columns:
        if 'Games Played' in df.columns:
            df['Games Played %'] = df['Games Played'] / 82.0
        else:
            raise ValueError(
                "CSV missing both 'Games Played %' and 'Games Played' columns after rename"
            )

    # Clamp GP to 0–1
    df['Games Played %'] = df['Games Played %'].clip(0, 1)

    # Raw games played is only an intermediate for the % above. Left in, it becomes an
    # upload-only column in the blend's union, and the blend's coverage rule (a player
    # must have every column covered by some source that carries them) would then drop
    # every player the upload doesn't cover — a partial upload would gut the pool. Minutes are
    # the same: read only to tell season totals from per-game numbers.
    df = df.drop(columns=['Games Played', 'Minutes'], errors='ignore')

    if 'Player' in df.columns:
        df = df.set_index('Player')

    return df


def _join_split_name(df: pd.DataFrame) -> pd.DataFrame:
    """'Player' from First Name and Last Name when the file names players over two columns. A
    file with a Player column of its own keeps it; the parts are dropped either way."""
    if 'Player' not in df.columns and _SPLIT_NAME_COLUMNS <= set(df.columns):
        first = df['First Name'].fillna('').astype(str).str.strip()
        last = df['Last Name'].fillna('').astype(str).str.strip()
        df = df.assign(Player=(first + ' ' + last).str.strip())
    return df.drop(columns=[column for column in _SPLIT_NAME_COLUMNS if column in df.columns])


def _derive_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build the columns a file lacks from the parts it carries (_DERIVED_CORES, _DERIVED_SHARES).
    A column the file has is never overwritten. A percentage with no attempts is 0, as the app's
    loaders write it: the attempts weighting makes it inert."""
    df = df.copy()
    if 'Rebounds' not in df.columns and _DERIVED_CORES['Rebounds'] <= set(df.columns):
        df['Rebounds'] = df['Off Rebounds'] + df['Def Rebounds']
    if 'Points' not in df.columns and _DERIVED_CORES['Points'] <= set(df.columns):
        df['Points'] = 2 * df['Field Goals Made'] + df['Threes'] + df['Free Throws Made']
    for share, (made, attempts) in _DERIVED_SHARES.items():
        if share not in df.columns and made in df.columns and attempts in df.columns:
            df[share] = (df[made] / df[attempts].where(df[attempts] > 0)).fillna(0.0)
    return df


def _convert_season_totals_to_per_game(df: pd.DataFrame, sport_params: dict) -> pd.DataFrame:
    """Divide a season-totals file by games played (see _MAX_PER_GAME). Every count is divided --
    the counting stats, the attempts, the makes, the minutes; shares and games are not. A totals
    file without games cannot be put per game and is refused; a player projected for no games has
    no per-game numbers and is dropped."""
    if 'Minutes' in df.columns:
        totals = df['Minutes'].max() > _MAX_MINUTES_PER_GAME
    else:
        totals = any(df[column].max() > ceiling for column, ceiling in _MAX_PER_GAME.items()
                     if column in df.columns)
    if not totals:
        return df
    if 'Games Played' not in df.columns:
        raise ValueError('These numbers are season totals (no player could average them per game), '
                         'but the file has no games-played column to divide them by.')
    volume_columns = {info['volume-statistic'] for info in sport_params['ratio-statistics'].values()}
    counts = [column for column in df.columns
              if column in sport_params['counting-statistics'] or column in volume_columns or column == 'Minutes']
    played = df['Games Played'] > 0
    if (~played).any():
        logging.getLogger('fbbo').info('Season-totals upload: dropping %d player(s) projected for no games',
                                       int((~played).sum()))
    df = df[played].copy()
    df[counts] = df[counts].div(df['Games Played'], axis=0)
    return df


def convert_percent_columns_to_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Read every share column -- the canonical names ending in '%' (Field Goal %, Free Throw %,
    Three %, Games Played %) -- as a fraction from 0 to 1, whichever way the file wrote it.

    Sources print shares both ways, 0.475 and 47.5, and the blend averages a file's numbers with
    ESPN's fractions, so a column in percent units must be converted or it swamps every score. A
    share written as a fraction can never exceed 1, so one value above 1 settles that the whole
    column is in percent units. A value above 100 is not a share in either form, so the file is
    refused rather than read as something it is not. (Assist to TO is a ratio, not a share, and can
    exceed 1; its name has no '%'.)"""
    converted = df.copy()
    for column in [column for column in df.columns if str(column).endswith('%')]:
        largest = converted[column].max()
        if largest > 100:
            raise ValueError(f"'{column}' has a value of {largest:g}, which is not a share in either form "
                             f'(a fraction from 0 to 1, or a percentage from 0 to 100).')
        if largest > 1:
            converted[column] = converted[column] / 100
    return converted


def _recover_volumes_from_ratio_cells(df: pd.DataFrame, sport_params: dict) -> pd.DataFrame:
    """Fill in a missing attempts column from the text of its percentage column.

    Attempt volume is load-bearing: a ratio G-score weights the percentage deviation by it,
    so a percentage without its volume cannot be scored at all. When a source carries that
    volume only inside the percentage cell, take it from there rather than discard it with
    the rest of the text. A file's own attempts column always wins — this fires only when
    there is none. Only the attempts are recovered, never the makes: the projection path
    does not use them, and emitting a column no other source carries would make the blend
    drop every player that source lacks.
    """
    for ratio_stat, ratio_info in sport_params['ratio-statistics'].items():
        volume_statistic = ratio_info['volume-statistic']
        if (ratio_stat not in df.columns
                or volume_statistic in df.columns
                or df[ratio_stat].dtype != object):
            continue
        attempts = pd.to_numeric(
            df[ratio_stat].astype(str).str.extract(_ATTEMPTS_IN_RATIO_CELL_PATTERN, expand=False),
            errors='coerce',
        )
        if attempts.notna().any():
            df[volume_statistic] = attempts
    return df
