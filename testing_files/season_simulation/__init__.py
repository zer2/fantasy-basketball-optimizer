# testing_files/season_simulation/
# Offline harness that simulates historical fantasy seasons through the backend and reports the
# results in H-score terms.
#
# Design (see the plan in the project history):
#   - simulate.py collects the data: for each (season, scoring format), it runs one snake draft per
#     seat, in which that one seat drafts by H-score (the algorithm) and the other eleven draft by
#     G-score order (position-eligibility respected). It records each H-score drafter's final team
#     H-score + per-category rates, and captures the per-pick candidate tables the website would have
#     shown. Everything is written to JSON — no rendering happens here.
#   - render.py is a separate step that reads those JSON files and builds a lazy HTML report.
