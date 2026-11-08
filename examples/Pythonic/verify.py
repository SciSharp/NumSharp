"""Runs each Python program of this folder and its C# port, and compares what the two print.

    .venv\\Scripts\\python verify.py        (Windows)
    .venv/bin/python verify.py            (Linux, macOS)

Run it with the interpreter that has the libraries. The ports embed that same interpreter, so both sides
compute with the same libraries. The exit code is the number of ports whose output is different.
"""
import os
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent

# A Python program of this folder is a .py file one level down: ScikitLearn/classification.py.
# Its port has the name of the file without the ending: dotnet run -- classification.
programs = sorted(
    path for path in here.glob("*/*.py")
    if not path.parent.name.startswith(".") and path.parent.name not in ("bin", "obj")
)

# The variable is binding for the ports: they embed this interpreter and no other one.
environment = dict(os.environ, NUMSHARP_PYTHONNET_PYTHON=sys.executable)

# Build one time, and show what the compiler said only when the build failed.
build = subprocess.run(["dotnet", "build", "--nologo", "--verbosity", "quiet"], cwd=here, capture_output=True, text=True)
if build.returncode != 0:
    print(build.stdout + build.stderr)
    sys.exit(f"dotnet build failed with exit code {build.returncode}")

different = 0
for program in programs:
    expected = subprocess.run([sys.executable, str(program)], capture_output=True, text=True, check=True).stdout.splitlines()
    port = subprocess.run(["dotnet", "run", "--no-build", "--", program.stem], cwd=here, env=environment, capture_output=True, text=True)
    actual = port.stdout.splitlines()

    same = port.returncode == 0 and actual == expected
    different += not same
    print(f"{program.relative_to(here).as_posix():<34} {len(expected):>3} lines   {'same output' if same else 'DIFFERENT'}")
    if not same:
        print(f"  exit code of the port: {port.returncode}")
        for line in port.stderr.splitlines():
            print(f"  ! {line}")
        for number, (python_line, port_line) in enumerate(zip(expected, actual), start=1):
            if python_line != port_line:
                print(f"  line {number}\n    python: {python_line}\n    port:   {port_line}")
        if len(expected) != len(actual):
            print(f"  python printed {len(expected)} lines, the port {len(actual)}")

sys.exit(different)
