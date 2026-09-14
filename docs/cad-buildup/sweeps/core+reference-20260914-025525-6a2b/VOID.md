# Void, and it bought something: the tool it measured was never reachable.

This sweep stopped at 03:41 with 12 of 26 cases started and **10 complete**, $11.14 spent.
It is void for the ordinary reason that 10 of 26 is not a corpus result. It is kept for
the other one, which is that the addition it was measuring was never delivered to a single
desk.

## What stopped it, since the first guess was wrong

Not a VM crash. The machine stayed up: `journalctl -b -1` runs continuously from
2026-09-13 12:25:58 to 2026-09-14 09:22:03, with journal lines on either side of 03:41 and
**zero shutdown or reboot lines between 02:50 and 04:00**. No OOM kill anywhere in that
boot, no coredump, 62 GB of RAM on a 32-core machine with the sweep using two workers.

What the evidence supports is a process-level termination when the session that launched
it ended: the driver was `nohup`'d but still in the session's process group, and its
`stdout` log -- written to a session scratchpad -- is gone entirely, which is what a
buffered stream looks like when the writer is killed rather than closed.

There *was* one genuine unclean stop, at **09:22:03**: boot `-1` ends mid-kernel-log with
no shutdown sequence and boot `0` begins 09:24:06. That is roughly five and a half hours
after this sweep was already dead, so it did not touch it.

The fix is `setsid`, not resumption: a sweep has to outlive the session that starts it.

## The finding: the desk could not reach what it was handed

`prepare()` copied `b123d_api.md` to `<workspace>/.reference/`. The desk's working
directory is the **case** directory, `<workspace>/t16/`, one level down. The brief says

> `.reference/b123d_api.md` in your working directory is build123d grouped by what each
> thing is for

and from the desk's working directory that path resolves to nothing.

**One run of the ten took the brief at its word.** T16, cell 1:

```python
print(subprocess.run(["grep","-n","-i","spline\\|revolve",
                      ".reference/b123d_api.md"],capture_output=True,text=True).stdout[:2000])
# -- output
/home/qiuzi/.openreynolds-buildup/work/T16-20260914-032741-ad84/t16 []
```

An empty grep result, and its own `os.listdir('.')` printing `[]`. The file existed, the
brief named it, the isolation mechanism worked, and no desk could open it.

The other nine never tried. `grep -c -E '\.reference|b123d_api'` over every `cells.log` in
this sweep returns 0 for T1, T10, T11, T12, T13, T14, T15, T17, T18 and 2 for T16.

This is the failure this repo has already measured once and written a test against --
*"a brief that names a path the workspace does not have is how a desk spends its opening
steps looking for what it was told it had -- seven of twenty-seven"* -- reproduced by the
addition that cites it. The test that existed
(`test_the_reference_the_brief_names_is_the_one_that_gets_copied`) checked that the source
file was on disk in the toolbox, which it was. It never resolved the path the brief gives
against the directory the desk is given.

## What the ten runs therefore do not say

Nothing about whether `b123d_api.md` helps. Nine desks never saw it and the tenth got an
empty string. The numbers below are recorded so they are not mistaken later for a
measurement of this addition:

| case | steps before → after | passed before → after | stopped |
|---|---|---|---|
| T1 | 13 → 9 | yes → yes | done |
| T10 | 27 → 18 | yes → **no** | time |
| T11 | 21 → 20 | yes → yes | done |
| T12 | 27 → 28 | no → **yes** | done |
| T13 | 17 → 27 | yes → **no** | time |
| T14 | 18 → 23 | yes → yes | done |
| T15 | 23 → 13 | no → no | time |
| T16 | 23 → 26 | yes → yes | done |
| T17 | 24 → 27 | yes → **no** | steps |
| T18 | 27 → 24 | yes → yes | done |

220 → 215 cells over the ten, 7/10 → 6/10 passed. With one run per case and ten cases,
four flips in both directions is sampling, and in any case it is measuring the two brief
lines and the probe repairs that rode along -- not the reference, which was not there.

## Corrected before the re-run

- `prepare()` copies into the case directory, which is the desk's working directory.
- `test_the_reference_the_brief_names_is_reachable_from_the_desks_own_directory` resolves
  the brief's own path against the directory the desk is handed, and fails against the
  code that produced this sweep.
- The re-run is launched with `setsid` and an explicit `--baseline`, because
  `--baseline latest` resolves to *this* sweep, which is void.
