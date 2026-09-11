# Quarantined: the runs taken before the toolbox path was fixed

Nothing in this directory counts toward the acceptance. It is kept because it is the
evidence for a defect, and deleting the evidence for a defect while reporting the defect
is how a finding becomes an assertion.

**What was wrong.** `openreynolds/cad/brief.py` told the desk its instruments "are in
`/work/.toolbox/`" as literal text, and `openreynolds/cad/check.py` ran `python3
/work/.toolbox/mesh_look.py` through a shell. Both statements are true on the hosted
image and false on `LocalBackend`, which is rooted wherever it was told to be. Nothing
translates `/work` for a shell, or for a `subprocess.run` inside a kernel cell —
`LocalBackend._resolve` is on the path of `put_file`, `get_file` and `stat`, and on
nothing that reaches a command line.

**What it cost, from `desk/T1/steps.json` in here.** Steps 0 to 6 of twenty-seven:

    0  grep: /work/.toolbox/b123d_api.md: No such file or directory
    1  python3: can't open file '/work/.toolbox/b123d_api.py'
    2  ls: cannot access '/work/.toolbox/': No such file or directory
    3  find / -iname '*b123d_api*'      <- a filesystem-wide search, 6.4 s, whose hits
                                           were stale /tmp/pytest-of-qiuzi directories
    4  ls: cannot access '/work/': No such file or directory
    5  ls ../.toolbox                   <- found it
    6  grep ../.toolbox/b123d_api.md    <- back to work

Seven steps, 26% of the budget, and the run ended `stopped='steps'` having never
meshed. The finish check on the same run reported **"the check wrote no readable
answer"** — about a mesh it had never opened, because `mesh_look.py` was not where it
looked. Neither failure raises anything. Both look like ordinary bad luck.

`runs-desk-quarantined/` holds T1 and T2 complete, and T3 as the partial record of the
run that was killed when the defect was found.
