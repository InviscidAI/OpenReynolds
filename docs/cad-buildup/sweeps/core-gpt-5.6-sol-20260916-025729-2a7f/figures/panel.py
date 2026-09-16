import sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
from cad_buildup import cases

SD = Path("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/"
          "957c0f71-2b79-452c-b452-7f7bbeb66bda/scratchpad")
CORPUS = cases()

PANELS = [
    ("T10", "blade passage",  "29 cells, stopped on the step budget.\n"
     "Diagnosed its own defect correctly — “a 51.7 µm blade/periodic intersection\n"
     "creates near-zero-volume cells” — then ran out testing phase angles."),
    ("T9",  "gyroscope bearing", "29 cells, stopped on the step budget.\n"
     "checkMesh flagged 580 small-determinant cells at cell 10; the desk cycled six\n"
     "mesh routes without ever clearing it. No property re-measured on the mesh."),
    ("T22", "vented brake disc", "27 cells, stopped on the step budget.\n"
     "Had a clean 36-region mesh by cell 19, then chased a cosmetic winding warning\n"
     "with tooling that made it worse — 72 flipped edges to 929."),
    ("T24", "swirl chamber", "28 cells, stopped on the step budget.\n"
     "Had the right mesh at cell 13, then spent 14 cells rebuilding patch export\n"
     "through a second toolchain and arrived with an identical mesh."),
]

fig = plt.figure(figsize=(21, 17), dpi=100)
fig.patch.set_facecolor("white")
fig.suptitle("core-gpt-5.6-sol-20260916-025729-2a7f — the four failing runs that delivered a mesh",
             fontsize=20, fontweight="bold", y=0.975, color="#1a1a1a")
fig.text(0.5, 0.951, "Every one built the right geometry. All four failed on budget, "
         "spent on export repair or warning-chasing — not on getting the shape wrong.",
         ha="center", fontsize=13.5, color="#555555", style="italic")

for i, (case, short, note) in enumerate(PANELS):
    row, col = divmod(i, 2)
    left = 0.035 + col * 0.485
    bottom = 0.505 - row * 0.468

    ax = fig.add_axes([left, bottom + 0.148, 0.45, 0.245])
    img = np.asarray(Image.open(SD / "renders" / f"{case}-part.png").convert("RGB"))
    # PyVista pads every screenshot with white; crop to the drawn geometry so the
    # panels are comparable in scale rather than in framing.
    ink = np.where((img < 245).any(axis=2))
    if len(ink[0]):
        pad = 8
        r0, r1 = max(ink[0].min() - pad, 0), min(ink[0].max() + pad, img.shape[0])
        c0, c1 = max(ink[1].min() - pad, 0), min(ink[1].max() + pad, img.shape[1])
        img = img[r0:r1, c0:c1]
    ax.imshow(img); ax.axis("off")

    tx = fig.add_axes([left, bottom - 0.028, 0.45, 0.175]); tx.axis("off")
    tx.text(0, 1.0, f"{case} — {short}", fontsize=16, fontweight="bold",
            va="top", color="#1a1a1a", transform=tx.transAxes)
    body = " ".join(CORPUS[case]["request"].replace("**", "").split())
    tx.text(0, 0.885, textwrap.fill(body, 96), fontsize=9.6, va="top",
            color="#333333", linespacing=1.55, family="serif", transform=tx.transAxes)
    tx.text(0, 0.175, note, fontsize=10, va="top", color="#b03030",
            linespacing=1.5, transform=tx.transAxes)

out = SD / "renders" / "failing-four.png"
fig.savefig(out, facecolor="white", bbox_inches=None)
print(out)
