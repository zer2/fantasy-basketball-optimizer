"""Parsing an uploaded projection file (.csv or .xlsx) into the canonical column set.

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


def _normalize_projection_header(header) -> str:
    """Header spellings differ only by case and padding far more often than by wording, so
    both sides of an alias lookup are folded to one form."""
    return str(header).strip().lower()


def _map_columns_to_canonical(df_raw: pd.DataFrame, params: dict) -> dict:
    """{column in the file: canonical name} for every column the aliases recognize.

    Columns that match nothing are left out (the parse drops them). A canonical name that
    two of the file's columns both claim is taken by the first, so a file carrying e.g.
    both 'PTS' and 'Points' cannot produce a duplicate column label downstream.
    """
    aliases = {_normalize_projection_header(alias): canonical
               for alias, canonical in params.get(_COLUMN_ALIASES_KEY, {}).items()}
    mapping, claimed = {}, set()
    for column in df_raw.columns:
        canonical = aliases.get(_normalize_projection_header(column))
        if canonical is not None and canonical not in claimed:
            mapping[column] = canonical
            claimed.add(canonical)
    return mapping


# A .xlsx is a ZIP archive, so every one begins with this signature. The older .xls is an
# OLE2 compound file with a different one — detected only to say so plainly, since reading it
# would need another engine and anything that can save .xls can also save .xlsx or .csv.
_XLSX_SIGNATURE = b'PK\x03\x04'
_XLS_SIGNATURE  = b'\xd0\xcf\x11\xe0'


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
        raise ValueError(
            'This is an older .xls spreadsheet, which cannot be read directly. '
            'Save it as .xlsx or .csv and upload again.')
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


def parse_projection_upload(upload_bytes: bytes, params: dict) -> pd.DataFrame:
    """Parse an uploaded projection file (.csv or .xlsx) into the canonical column set.

    There is no format detection: each column is interpreted on its own through the alias
    table (see 'projection-column-aliases' in parameters.yaml), so any source is readable
    as long as its header spellings are known, and a file already written in canonical
    names needs no aliases at all. Raises ValueError naming what could not be found when
    the file does not read as a projection set.
    """
    df_raw = _read_projection_table(upload_bytes)
    column_mapping  = _map_columns_to_canonical(df_raw, params)
    # Unrecognized columns keep their own names, so a file already using canonical names
    # is understood without any alias matching at all.
    renamed_columns = set(df_raw.rename(columns=column_mapping).columns)

    missing_identity = [column for column in ('Player', 'Position')
                        if column not in renamed_columns]
    matched_cores    = [column for column in CORE_PROJECTION_COLUMNS
                        if column in renamed_columns]
    if not missing_identity and len(matched_cores) >= _MIN_MATCHED_CORE_COLUMNS:
        # Every name the aliases can produce, so a file already written in canonical names
        # keeps those columns even though they never went through the mapping.
        canonical_columns = set(params.get(_COLUMN_ALIASES_KEY, {}).values()) | {'Games Played %'}
        return _parse_with_renamer(df_raw, column_mapping, canonical_columns, params)

    if missing_identity:
        problem = f"no column for {' or '.join(missing_identity)}"
    else:
        missing_cores = [column for column in CORE_PROJECTION_COLUMNS
                         if column not in renamed_columns]
        problem = (f'only {len(matched_cores)} of {len(CORE_PROJECTION_COLUMNS)} core stats '
                   f"were recognized (no {', '.join(missing_cores)})")
    unrecognized = [column for column in df_raw.columns if column not in column_mapping]
    raise ValueError(
        f'File does not read as a projection set: {problem}. Headers that were not '
        f"recognized: {', '.join(map(str, unrecognized))}. Add their spellings to "
        f'{_COLUMN_ALIASES_KEY} to teach the parser this source.'
    )


# Attempts hiding inside a ratio cell: sources that print a percentage with its makes and
# attempts behind it — "0.583 (10.2/17.5)" — and ship no attempts column of their own.
# Captures the second number, the attempts.
_ATTEMPTS_IN_RATIO_CELL_PATTERN = r'\(\s*-?[\d.]+\s*/\s*(-?[\d.]+)\s*\)'


def _recover_volumes_from_ratio_cells(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Fill in a missing attempts column from the text of its percentage column.

    Attempt volume is load-bearing: a ratio G-score weights the percentage deviation by it,
    so a percentage without its volume cannot be scored at all. When a source carries that
    volume only inside the percentage cell, take it from there rather than discard it with
    the rest of the text. A file's own attempts column always wins — this fires only when
    there is none. Only the attempts are recovered, never the makes: the projection path
    does not use them, and emitting a column no other source carries would make the blend
    drop every player that source lacks.
    """
    for ratio_stat, ratio_info in params['ratio-statistics'].items():
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


def _parse_with_renamer(
    df_raw: pd.DataFrame
    , column_mapping: dict
    , canonical_columns: set
    , params: dict
) -> pd.DataFrame:
    """Rename this file's columns to canonical names, drop junk, coerce stats."""
    df = df_raw.rename(columns=column_mapping)

    # Keep only canonical columns. Unmapped extras (ranks, dollar values, minutes, ...)
    # would otherwise join the blend's column union, where every player from the OTHER
    # sources is "missing" them — and the blend drops any player missing any column
    # across all sources, so a single junk column can wipe out the entire pool.
    df = df[[column for column in df.columns if column in canonical_columns]].copy()

    # Before the ratio cells are reduced to their leading number below, mine them for any
    # attempts column the file does not carry separately.
    df = _recover_volumes_from_ratio_cells(df, params)

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

    stat_columns = [column for column in df.columns if column not in ('Player', 'Position')]
    df[stat_columns] = df[stat_columns].apply(coerce_stat_column)
    df = df.dropna(subset=stat_columns, how='all')

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
    # every player the upload doesn't cover — a partial upload would gut the pool.
    df = df.drop(columns=['Games Played'], errors='ignore')

    if 'Player' in df.columns:
        df = df.set_index('Player')

    return df
