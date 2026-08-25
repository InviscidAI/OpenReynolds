"""`python -m openreynolds` — the same CLI without the generated .exe shim.

Windows Smart App Control blocks unsigned launchers like the one pip writes
into Scripts/; python.exe itself is signed, so this route always works. The
prog_name keeps usage and --version reading "openreynolds" either way.
"""

from openreynolds.cli import main

if __name__ == "__main__":
    main(prog_name="openreynolds")
