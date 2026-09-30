"""Launch this worktree's UI and report the actual imported modules."""
from pathlib import Path
import subprocess
import sys
from square_demo import ui, providers

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    print('Каталог:',root,flush=True)
    for args in (('branch','--show-current'),('rev-parse','--short','HEAD')):
        print(subprocess.check_output(('git',*args),cwd=root,text=True).strip(),flush=True)
    print('Python:',sys.executable,'\nИнтерфейс:',ui.__file__,'\nРасчёт:',providers.__file__,flush=True)
    raise SystemExit(ui.run())
