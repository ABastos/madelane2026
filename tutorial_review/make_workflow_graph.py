"""Render the MaDeLaNe 2024-2026 workflow coverage map from inventory.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
inv = json.loads((HERE / "inventory.json").read_text())
YEARS = {2024: "#2a78d6", 2025: "#eb6834", 2026: "#1baf7a"}
INK, INK2, MUTED, SURF, CHIP = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#f1f0ec"

stages = inv["stages"]
cols = {s["id"]: i for i, s in enumerate(stages)}
by_stage = {s["id"]: [w for w in inv["workflows"] if w["stage"] == s["id"]] for s in stages}
nrows = max(len(v) for v in by_stage.values())

CW, GAP, CH, VGAP = 2.0, 0.32, 0.86, 0.16
W = len(stages) * CW + (len(stages) - 1) * GAP
H = 1.25 + nrows * (CH + VGAP) + 0.55
fig = plt.figure(figsize=(W * 1.02, H * 1.02), facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-.1, W + .1); ax.set_ylim(H, 0); ax.axis("off")

top = 0.25
for s in stages:
    x = cols[s["id"]] * (CW + GAP)
    ax.add_patch(FancyBboxPatch((x, top), CW, 0.78, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=INK, ec="none"))
    ax.text(x + CW / 2, top + .3, s["label"], ha="center", va="center", color="white", fontsize=12.5, weight="bold")
    ax.text(x + CW / 2, top + .58, s["sub"], ha="center", va="center", color="#c3c2b7", fontsize=8.5)
    if cols[s["id"]] < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + CW + .02, top + .39), (x + CW + GAP - .02, top + .39),
                                     arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.4))
    for r, w in enumerate(by_stage[s["id"]]):
        y = top + 1.0 + r * (CH + VGAP)
        ran = w.get("ran", False)
        ax.add_patch(FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="white" if ran else CHIP, ec=INK if ran else "#dddcd7",
                                    lw=1.6 if ran else .8))
        ax.text(x + .1, y + .2, w["topic"], ha="left", va="center", fontsize=8.6, color=INK, weight="bold",
                wrap=True)
        ax.text(x + .1, y + .45, w["lead"], ha="left", va="center", fontsize=7.4, color=INK2)
        ax.text(x + .1, y + .66, w["tools"], ha="left", va="center", fontsize=7, color=MUTED, style="italic")
        for k, (yr, c) in enumerate(YEARS.items()):
            cx, cy = x + CW - .48 + k * .17, y + .66
            if yr in w["years"]:
                ax.add_patch(plt.Circle((cx, cy), .062, fc=c, ec="white", lw=1.2))
            else:
                ax.add_patch(plt.Circle((cx, cy), .055, fc="none", ec="#cfcec8", lw=.8))
        if ran:
            ax.text(x + CW - .58, y + .66, "RAN", ha="right", va="center", fontsize=6.8, color=INK, weight="bold")

ly = H - .3
n = {y: sum(y in w["years"] for w in inv["workflows"]) for y in YEARS}
lx = 0
for yr, c in YEARS.items():
    ax.add_patch(plt.Circle((lx + .1, ly), .07, fc=c, ec="none"))
    ax.text(lx + .25, ly, f"{yr}  ({n[yr]} workflows)", va="center", fontsize=9.5, color=INK2)
    lx += 2.1
ax.add_patch(FancyBboxPatch((lx, ly - .14), .32, .28, boxstyle="round,pad=0,rounding_size=.04", fc="white", ec=INK, lw=1.6))
ax.text(lx + .45, ly, "RAN = re-run in Python for this review", va="center", fontsize=9.5, color=INK2)

out = HERE / "outputs"; out.mkdir(exist_ok=True)
fig.savefig(out / "workflow_map.png", dpi=170, facecolor=SURF)
fig.savefig(out / "workflow_map.svg", facecolor=SURF)
print(n, "stages:", {k: len(v) for k, v in by_stage.items()})
