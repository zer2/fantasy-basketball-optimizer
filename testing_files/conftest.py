import sys
import os

# The app requires SESSION_SECRET_KEY (it signs the login cookie; no fallback). Tests do no
# real OAuth, so a dummy value is fine — set it before any test imports backend.main.
os.environ.setdefault('SESSION_SECRET_KEY', 'test-only-session-secret')

# Add the testing_files/ directory to sys.path so that benchmark_helpers
# can be imported by the benchmark_* test modules without a package prefix.
# Pytest's working directory is the project root, so this directory would
# not otherwise be on the path.
sys.path.insert(0, os.path.dirname(__file__))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Render the behavioural-properties report (measurements for a human to read — the behavioural
    tier is not green/red; the details matter) at the end of the run, and write it to
    testing_files/behavior_report.md. No-op when no behavioural test recorded anything."""
    import sys as _sys
    behavior_module = next(
        (module for name, module in _sys.modules.items()
         if name.rsplit('.', 1)[-1] == 'test_behavior_properties' and module is not None),
        None,
    )
    if behavior_module is None:      # behavioural tests were not part of this run
        return
    render_behavior_report = behavior_module.render_behavior_report
    report_text = render_behavior_report()
    if report_text:
        terminalreporter.write_sep('=', 'BEHAVIOURAL PROPERTIES REPORT')
        terminalreporter.write(report_text + '\n')
        if getattr(behavior_module, '_RUN_SIMS', False):
            # Report mode only: quick default runs must not overwrite full multi-season results.
            page_path = behavior_module.write_behavior_report_files(os.path.dirname(__file__))
            terminalreporter.write(f'(tabbed report: {page_path}; per-section results in behavior_reports/)\n')
        else:
            terminalreporter.write('(quick mode: stored reports untouched; RUN_BEHAVIOR_SIMS=1 regenerates them)\n')
