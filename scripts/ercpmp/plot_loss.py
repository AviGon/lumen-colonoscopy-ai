"""Plot training and test loss per epoch from a train_log.csv.
Usage: python plot_loss.py <model dir> [title]   -> writes <model dir>/loss_curves.png"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_dir = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else "Training and test loss"
rows = list(csv.DictReader(open(os.path.join(model_dir, "train_log.csv"))))
ep = [int(r["epoch"]) for r in rows]
train = [float(r["train_loss"]) for r in rows]
test = [float(r["test_loss"]) for r in rows]

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"Training loss": "#2a78d6", "Test loss": "#eb6834"}  # categorical slots 1 and 2, validated

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

for (name, color), ys in zip(SERIES.items(), (train, test)):
    ax.plot(ep, ys, color=color, lw=2, label=name, solid_capstyle="round", zorder=3)
    ax.plot(ep[-1], ys[-1], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ax.annotate(f"{name}  {ys[-1]:.2f}", (ep[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=9.5)

best = min(range(len(test)), key=test.__getitem__)
ax.plot(ep[best], test[best], "o", ms=8, mfc=SURFACE, mec=SERIES["Test loss"], mew=2, zorder=4)
ax.annotate(f"lowest test loss {test[best]:.2f} (epoch {ep[best]})", (ep[best], test[best]),
            xytext=(0, 14), textcoords="offset points", ha="center", color=INK_2, fontsize=9)

ax.set_xlim(0.5, ep[-1] + 5.5); ax.set_ylim(0, max(train + test) * 1.08)
ax.set_xticks([e for e in ep if e == 1 or e % 5 == 0])
ax.set_xlabel("Epoch", color=MUTED); ax.set_ylabel("Loss", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
leg = ax.legend(loc="upper right", frameon=False, labelcolor=INK_2)
ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
ax.text(0, 1.02, "Lower is better. The final epoch (25) is the saved model; test data was never used to choose it.",
        transform=ax.transAxes, color=MUTED, fontsize=9)
fig.tight_layout()
out = os.path.join(model_dir, "loss_curves.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
