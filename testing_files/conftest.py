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
    """Render the experiment report (measurements for a human to read — the experiment tier is not
    green/red; the details matter) at the end of the run. In report mode (RUN_EXPERIMENTS=1) also
    persist per-section results and rebuild the tabbed experiment_report.html; quick default runs
    leave the stored reports untouched. No-op when no experiment recorded anything."""
    import sys as _sys
    experiments_module = next(
        (module for name, module in _sys.modules.items()
         if name.rsplit('.', 1)[-1] == 'test_experiments' and module is not None),
        None,
    )
    if experiments_module is None:      # experiments were not part of this run
        return
    report_text = experiments_module.render_experiment_report()
    if report_text:
        terminalreporter.write_sep('=', 'EXPERIMENT REPORT')
        terminalreporter.write(report_text + '\n')
        if getattr(experiments_module, '_RUN_SIMS', False):
            # Report mode only: quick default runs must not overwrite full multi-season results.
            page_path = experiments_module.write_experiment_report_files(os.path.dirname(__file__))
            terminalreporter.write(f'(tabbed report: {page_path}; per-section results in experiment_results/)\n')
        else:
            terminalreporter.write('(quick mode: stored reports untouched; RUN_EXPERIMENTS=1 regenerates them)\n')
