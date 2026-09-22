import sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
from cad_buildup import cases

SD = Path("/tmp/claude-1001/-home-qiuzi-PycharmProjects-OpenReynolds/"
          "957c0f71-2b79-452c-b452-7f7bbeb66bda/scratchpad")
CORPUS = cases()

# (case, short name, what the vet found, colour of that note)
GOOD, MIXED, BAD = "#2f6f4e", "#9a6a15", "#b03030"
PANELS = [
 ("T23", "caged ball", "PASS, and the vet agrees.\n"
  "Volume 2.06597e-05 against the exact boolean 2.0658481e-05 — 0.0059%. Not the\n"
  "14,137 mm³ ball-only signature the case is built to catch. Gap 1.000 mm on an\n"
  "axis and on a diagonal, measured independently on the delivered mesh.", GOOD),
 ("T18", "finned cylinder", "PASS, and it repaired a real defect unprompted.\n"
  "Hit the per-face export_stl seam failure live at 8,364 free edges, caught it with\n"
  "its own assertion, fixed it by tessellating the fused solid once. Volume 0.002%.\n"
  "But all six named properties are echoes of input constants.", MIXED),
 ("T11", "honeycomb lattice core", "PASS, mesh sound.\n"
  "Volume matches duct-minus-wall exactly; the vet sliced the mesh and counted 105\n"
  "open cells against the 105 planned. Four of six properties never measured, and\n"
  "the open-area ratio is the boolean's own arithmetic — the case's stated trap.", MIXED),
 ("T13", "thread leak path", "PASS, mesh sound.\n"
  "Volume 1.52164e-07 against 1.522e-07 — 0.02%. One connected region, which is the\n"
  "case's trap. But the flank gap is a norm across the grid built from GAP, so it\n"
  "equals the request by construction; centreline length never printed.", MIXED),
 ("T15", "gear backlash oil", "PASS, and the pass is hollow.\n"
  "checkMesh passed honestly — stock thresholds, gap never widened. It passed because\n"
  "refinement is ~8x coarser than the 0.3 mm backlash, so the sliver that would fail\n"
  "it never forms. Volume 4.76% short, a comparison the desk never printed.", BAD),
 ("T26", "impossible as specified", "PASS, and the green is not real.\n"
  "Tread built as a box from the shaft axis with the column subtracted, so the\n"
  "tangency the case exists to surface is resolved by boolean order, not measured.\n"
  "Volume matches to 0.064% — and cannot distinguish right from silently fused.", BAD),
]

fig = plt.figure(figsize=(21, 27), dpi=100)
fig.patch.set_facecolor("white")
fig.suptitle("core-gpt-5.6-sol-20260916-025729-2a7f — six runs that passed",
             fontsize=21, fontweight="bold", y=0.979, color="#1a1a1a")
fig.text(0.5, 0.9605, "All six cleared the gate. The vet broke two of them outright and found "
         "the named properties unmeasured in four — passing is not the same as correct.",
         ha="center", fontsize=13.5, color="#555555", style="italic")

for i, (case, short, note, colour) in enumerate(PANELS):
    row, col = divmod(i, 2)
    left = 0.035 + col * 0.485
    bottom = 0.648 - row * 0.302

    ax = fig.add_axes([left, bottom + 0.112, 0.45, 0.170])
    img = np.asarray(Image.open(SD / "renders" / f"{case}-part.png").convert("RGB"))
    ink = np.where((img < 245).any(axis=2))
    if len(ink[0]):
        pad = 8
        img = img[max(ink[0].min()-pad, 0):min(ink[0].max()+pad, img.shape[0]),
                  max(ink[1].min()-pad, 0):min(ink[1].max()+pad, img.shape[1])]
    ax.imshow(img); ax.axis("off")

    tx = fig.add_axes([left, bottom + 0.002, 0.45, 0.118]); tx.axis("off")
    tx.text(0, 1.0, f"{case} — {short}", fontsize=16, fontweight="bold",
            va="top", color="#1a1a1a", transform=tx.transAxes)
    body = " ".join(CORPUS[case]["request"].replace("**", "").split())
    tx.text(0, 0.845, textwrap.fill(body, 96), fontsize=9.6, va="top",
            color="#333333", linespacing=1.55, family="serif", transform=tx.transAxes)
    tx.text(0, 0.245, note, fontsize=10, va="top", color=colour,
            linespacing=1.5, transform=tx.transAxes)

out = SD / "renders" / "passing-six.png"
fig.savefig(out, facecolor="white")
print(out)
