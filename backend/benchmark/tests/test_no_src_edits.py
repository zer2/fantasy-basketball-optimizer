# benchmark/tests/test_no_src_edits.py
import subprocess

def test_src_and_app_untouched():
    # Compare against benchmark-base (this branch's fork point), NOT main:
    # the parent branch already carries zer2's src/ edits; only our changes should be measured.
    out = subprocess.run(['git', 'diff', '--name-only', 'benchmark-base', '--', 'src/', 'app.py'],
                         capture_output=True, text=True).stdout.strip()
    assert out == '', f'Engine files modified: {out}'
