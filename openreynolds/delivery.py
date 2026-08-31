"""Getting the pictures to the person, without asking the agent to remember to.

The complaint that produced this file, five sessions running: the agent renders
thirty pictures, they mirror home automatically, and the person still types "where is
the image?" -- because nothing points at them and they are buried three directories
deep under a doubled study id. In one session the agent produced seven renders and a
gif and never once ran `fetch`. Delivery was the agent's job and the agent, reasonably,
was busy running a simulation.

So delivery stops being the agent's job. The mirror already brings every file home on
its own thread; this rides on that. Each cycle it looks at what just arrived and does
three things, none of which involve the model:

- **surfaces** every new render into one flat `studies/<id>/renders/` folder, so the
  answer to "where is it" is one obvious place rather than a path nobody would guess;
- **assembles** a directory of animation frames into a gif on this machine -- the
  instance renders frames next to the data, the encoder lives here, and a growing
  frame set is re-assembled as it grows, which is exactly the "partial gif of what you
  have so far" a person asked for while a solve was still running;
- **announces** what it did, so the person is told a picture exists rather than left
  to find it.

Nothing here is on the instance and nothing here is a tool the model calls. It is the
same seam as the rest: the agent produces, the harness presents.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import images, video

FRAME_DIR_HINTS = ("frame", "anim", "movie", "shed", "unsteady")
"""Directory names that suggest a sequence to assemble rather than a gallery of
stills. A `renders/` folder of mesh.png + p.png + U.png is not an animation; a
`frames/` or `anim/` folder of numbered pictures is."""

MIN_SEQUENCE = 4
"""Fewer numbered frames than this is a handful of stills, not a film."""

MIN_ASSEMBLE_GROWTH = 3
"""Re-assemble a growing frame set only after it has gained at least this many
frames, so a live render does not trigger an encode every single mirror cycle."""

_NUMBERED = re.compile(r"^(.*?)(\d+)$")


@dataclass
class DeliveryEvent:
    """What one mirror cycle delivered. Empty events are not reported."""

    images: list[Path] = field(default_factory=list)
    """New still renders, copied into the flat renders folder."""
    videos: list[Path] = field(default_factory=list)
    """Animations assembled (or re-assembled, fuller) this cycle."""

    def __bool__(self) -> bool:
        return bool(self.images or self.videos)

    def lines(self) -> list[str]:
        out = []
        for path in self.videos:
            out.append(f"assembled {path.name}")
        n = len(self.images)
        if n:
            out.append(f"{n} new render{'' if n == 1 else 's'}")
        return out


class Gallery:
    """Surfaces and assembles renders as the mirror brings them home.

    Holds the small amount of state that keeps it from doing the same work twice:
    which source images it has already copied, and how many frames each animation
    directory had when it was last assembled. One instance per session, driven from
    the mirror's own thread.
    """

    def __init__(self, files_dir: Path, renders_dir: Path, assemble=None, encoder=None,
                 capture=None):
        self.files_dir = files_dir
        self.renders_dir = renders_dir
        self.capture = capture
        """Where a surfaced render is also sent for keeping, or None.

        Renders used to reach the platform only when the model happened to call
        `fetch` on one -- and delivery had deliberately stopped using `fetch`, so in
        the ordinary path nothing was ever uploaded and a study looked at from
        anywhere else had no pictures at all. This is the same moment a render
        becomes visible locally, so it is the moment to send it. Fire-and-forget,
        like every other capture: it cannot delay or fail a study."""
        self._assemble = assemble or video.assemble
        self._encoder = encoder or video.encoder
        self._delivered: set[str] = set()
        """Source image paths already copied, so a still is surfaced once."""
        self._assembled: dict[str, int] = {}
        """Frame directory -> frame count at last assembly."""

    def ingest(self, report) -> DeliveryEvent:
        """Look at what one sync pulled, surface and assemble, and say what changed.

        Never raises: it runs inside the mirror, and a convenience may not end a
        session. A frame that cannot be copied or a gif that will not encode becomes
        a quiet nothing, and the next cycle tries again."""
        event = DeliveryEvent()
        pulled = list(getattr(report, "pulled", None) or [])
        try:
            self._surface_stills(pulled, event)
            self._assemble_sequences(pulled, event)
        except Exception:  # noqa: BLE001 - presentation may not end a session
            pass
        return event

    # -- stills ----------------------------------------------------------------

    def _surface_stills(self, pulled: list[Path], event: DeliveryEvent) -> None:
        for src in pulled:
            if images.media_type(src.name) is None:
                continue
            key = str(src)
            if key in self._delivered:
                continue
            # A frame that is part of a sequence is delivered as the assembled gif,
            # not as a hundred loose stills flooding the folder.
            if self._is_frame(src):
                continue
            landed = self._copy_flat(src)
            if landed is not None:
                self._delivered.add(key)
                event.images.append(landed)

    def _copy_flat(self, src: Path) -> Path | None:
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        target = self.renders_dir / src.name
        if target.exists() and target.resolve() != src.resolve():
            # Two cases both wrote `p.png`; keep both, named by where they came from.
            target = self.renders_dir / f"{src.parent.name}_{src.name}"
        try:
            shutil.copy2(src, target)
        except OSError:
            return None
        self._keep(target)
        return target

    def _keep(self, path: Path) -> None:
        """Send a surfaced render to the platform, if this session is capturing."""
        if self.capture is None:
            return
        try:
            self.capture.artifact(path, kind="render")
        except Exception:  # noqa: BLE001 - a convenience may not end a session
            pass

    # -- animations ------------------------------------------------------------

    def _assemble_sequences(self, pulled: list[Path], event: DeliveryEvent) -> None:
        if self._encoder() is None:
            return  # no encoder on this machine; `openreynolds video` still explains
        for directory in self._touched_dirs(pulled):
            frames = video.frames_in(directory)
            if not self._is_sequence(directory, frames):
                continue
            # `_is_sequence` already sets the floor: a named anim/frames dir counts
            # from two frames, a bare numbered run needs MIN_SEQUENCE. No second floor.
            last = self._assembled.get(str(directory), 0)
            if last and len(frames) - last < MIN_ASSEMBLE_GROWTH:
                continue
            # What the frames were rendered for, when the instance said so. A gif
            # at DEFAULT_FPS is the fallback for a directory nobody declared -- not
            # the answer for one that asked for webp at 24.
            name, fps = video.intent(directory)
            out = self.renders_dir / (name or f"{directory.name}.gif")
            self.renders_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._assemble(frames, out, fps=fps)
            except video.VideoError:
                continue
            self._assembled[str(directory)] = len(frames)
            self._keep(out)
            event.videos.append(out)

    def _touched_dirs(self, pulled: list[Path]) -> list[Path]:
        dirs = []
        seen = set()
        for path in pulled:
            if images.media_type(path.name) is None:
                continue
            parent = path.parent
            if str(parent) not in seen:
                seen.add(str(parent))
                dirs.append(parent)
        return dirs

    def _is_sequence(self, directory: Path, frames: list[Path]) -> bool:
        """Whether a directory is an animation to assemble, not a gallery of stills.

        Two signals: the directory says so by its name (`frames/`, `anim/`), or the
        pictures do by being a numbered run sharing one stem (`shed_000.png`,
        `shed_001.png`, ...). A handful of differently-named stills is neither."""
        name = directory.name.lower()
        if any(hint in name for hint in FRAME_DIR_HINTS) and len(frames) >= 2:
            return True
        stems = [_NUMBERED.match(f.stem) for f in frames]
        numbered = [m for m in stems if m]
        if len(numbered) < MIN_SEQUENCE:
            return False
        prefixes = {m.group(1) for m in numbered}
        return len(prefixes) == 1

    def _is_frame(self, path: Path) -> bool:
        name = path.parent.name.lower()
        if any(hint in name for hint in FRAME_DIR_HINTS):
            return True
        siblings = video.frames_in(path.parent)
        return self._is_sequence(path.parent, siblings) and len(siblings) >= MIN_SEQUENCE

    # -- reading the folder back -----------------------------------------------

    def newest(self, limit: int = 24) -> list[Path]:
        """What is in the renders folder, newest first."""
        if not self.renders_dir.is_dir():
            return []
        found = [p for p in self.renders_dir.iterdir() if p.is_file()]
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
