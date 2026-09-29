"""Единая точка входа; сведения о реально загруженном коде печатаются при старте."""
from pathlib import Path
import subprocess
import sys

import orbit_study.passive_optimizer as optimizer
import square_demo.ui as interface


def _git_value(*args):
    try:
        return subprocess.check_output(("git",*args),cwd=Path(__file__).resolve().parent,
                                       text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "вне Git"


def launch_identity():
    root=Path(__file__).resolve().parent
    return {"repository":str(root),"branch":_git_value("branch","--show-current"),
            "commit":_git_value("rev-parse","HEAD"),"python":sys.executable,
            "main":str(Path(__file__).resolve()),"interface":str(Path(interface.__file__).resolve()),
            "optimizer":str(Path(optimizer.__file__).resolve()),
            "optimizer_schema":optimizer.VERSION}


if __name__ == "__main__":
    for key,value in launch_identity().items():
        print(f"square_projection {key}: {value}",flush=True)
    raise SystemExit(interface.run())
