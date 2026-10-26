"""Run every Chapter 2 re-derivation script in order and report the totals."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
scripts = sorted(here.glob("s[0-9][0-9]_*.py"))
failed = []
for s in scripts:
    print(f"\n######## {s.name} ########", flush=True)
    r = subprocess.run([sys.executable, s.name], cwd=here)
    if r.returncode != 0:
        failed.append(s.name)
print("\n" + ("ALL SCRIPTS PASSED" if not failed else f"FAILED: {failed}"))
sys.exit(1 if failed else 0)
