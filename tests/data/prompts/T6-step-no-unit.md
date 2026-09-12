# T6 — STEP that declares no unit

**Capability:** STEP prep, refusal path. **Fixture:** a STEP with its unit declaration
stripped, committed beside C1's.

## Request

> Mesh the fluid volume in this STEP file. It came from a customer.

`geometry`: the unit-less fixture path.

## What a pass looks like

**Passes as:** `refused`

**The desk does not finish, and that is the correct outcome.** It reports up with the
reason — the file declares no length unit, none was supplied, and the numbers could be
millimetres or metres — and returns that to the caller. It does **not** guess, and it does
not ask the user itself.

`#10`: "It reports up and asks no one. It returns `ok`, its reasons and its `stopped` state
to the caller; the main agent — conversational, and already permitted to ask — decides
whether to retry with a changed request or put the question to the user."

## What this catches

**The factor of 1000.** `cad_convert` "refuses in exactly two cases, both of them cases
where a wrong answer looks like a right one", and this is one of them: "whether those
numbers are millimetres or metres is a factor of 1000 on every length in the study". A
customer STEP with no declared unit "is a live case, not a corner".

**The desk blocking on a human**, which it must never do. A desk that stops and waits fails
this prompt as surely as one that guesses millimetres.

**A refusal arriving late.** The `geometry` path is checked for existence and readability
before the run starts (`#12`); the unit refusal comes from the first read of the file, not
from step nine after a shape has been built on numbers of unknown scale.

## A pass that is really a failure

The desk guesses millimetres — correctly, as it happens, because most STEP files are
millimetres — and finishes with a mesh that is right. It passed by luck, and the next file
is inches.
