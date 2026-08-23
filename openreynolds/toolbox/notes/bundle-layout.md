# One possible `/work` layout

**This is a suggestion, not a convention you are expected to follow.** `/work` is yours, it
persists between sessions, and any structure that makes sense to you is the right one. What
follows is simply one layout that has worked, offered in case a starting point is useful.

```
/work/
  <study-name>/
    notes.md              # what you are doing and why; the thing future-you reads first
    geometry/             # STL/STEP as received, plus whatever you did to canonicalize it
    case/                 # a real OpenFOAM case: 0/ constant/ system/
    runs/
      01-coarse/          # a copy of case/ per attempt, so a comparison is still possible
      02-medium/
    renders/              # PNGs worth keeping or fetching
    results.json          # if you happen to write one, it is picked up at session end
```

Two things about this are worth more than the layout itself.

**Notes on disk outlive the conversation.** A thread gets refreshed when it grows long, and
a session ends when the user closes the terminal. `/work` does not. Anything you would want
to know at the start of the next session — what you tried, what it gave you, what you had
decided not to bother with — is worth a line in a file. Nothing checks whether you wrote
one.

**Keeping a run rather than overwriting it** is what makes "the finer mesh moved the answer
by 4%" a sentence you can say later. Copying a case directory is cheap next to re-running
the solve.

`results.json` is picked up automatically at the end of a session if it happens to exist,
with whatever shape you gave it. There is no schema and nothing requires one.
