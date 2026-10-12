"""Build model-comparison tables and charts into <project>/artifacts.

Run from the project root after the training notebook:   python src/make_comparison.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))
from data_preprocessing import load_raw_mnist, scale_pixels  # noqa: E402
from inference import DigitRecognizer  # noqa: E402

# palette: first three categorical slots (validated all-pairs), recessive ink/grid
COL = {"SVM": "#2a78d6", "MLP": "#eb6834", "CNN": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODELS = ["SVM", "MLP", "CNN"]
KEYS = {"SVM": "svm", "MLP": "mlp", "CNN": "cnn"}
HW = {"SVM": "CPU", "MLP": "GPU", "CNN": "GPU"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

res = json.loads((ART / "final_results.json").read_text())
hist = json.loads((ART / "training_histories.json").read_text())

# ---------------------------------------------------------------- predictions on the test set
_, _, x_test_raw, y_test = load_raw_mnist(ROOT / "data")
x_test = scale_pixels(x_test_raw)
rec = DigitRecognizer()
pred = {}
for m in MODELS:
    pred[m] = np.array([rec.predict(KEYS[m], img)[0] for img in x_test])

# ---------------------------------------------------------------- table 1: overall comparison
rows = []
for m in MODELS:
    r = res[m]
    rows.append({
        "Model": m, "Hardware": HW[m],
        "Test accuracy (%)": round(100 * r["test_acc"], 2),
        "Validation accuracy (%)": round(100 * r["val_acc"], 2),
        "Train-subset accuracy (%)": round(100 * r["train_acc"], 2),
        "Macro precision (%)": round(100 * r["macro_precision"], 2),
        "Macro recall (%)": round(100 * r["macro_recall"], 2),
        "Macro F1 (%)": round(100 * r["macro_f1"], 2),
        "Test errors (of 10,000)": r["test_errors"],
        "Training time (s)": round(r["train_time_s"], 1),
        "Inference (ms / image)": round(r["inference_ms_per_image"], 4),
        "Stored values / parameters": r["complexity"],
        "File size (MB)": round(r["file_mb"], 2),
        "Best epoch": r["best_epoch"] if r["best_epoch"] is not None else "-",
    })
overall = pd.DataFrame(rows).set_index("Model")
overall.to_csv(ART / "comparison_overall.csv")

# ---------------------------------------------------------------- table 2: per-digit F1
pc = {}
for m in MODELS:
    p, r, f, _ = precision_recall_fscore_support(y_test, pred[m], labels=range(10), zero_division=0)
    pc[m] = dict(precision=p, recall=r, f1=f)
per_digit = pd.DataFrame({f"{m} F1 (%)": np.round(100 * pc[m]["f1"], 2) for m in MODELS})
per_digit.index.name = "Digit"
per_digit["Best model"] = [MODELS[int(np.argmax([pc[m]["f1"][d] for m in MODELS]))] for d in range(10)]
per_digit.to_csv(ART / "comparison_per_digit.csv")

# ---------------------------------------------------------------- table 3: most frequent confusions
conf_rows = []
cms = {m: confusion_matrix(y_test, pred[m], labels=range(10)) for m in MODELS}
for m in MODELS:
    cm = cms[m].copy(); np.fill_diagonal(cm, 0)
    for flat in np.argsort(cm, axis=None)[::-1][:5]:
        t, p_ = divmod(int(flat), 10)
        conf_rows.append({"Model": m, "True digit": t, "Predicted as": p_, "Count": int(cm[t, p_])})
confusions = pd.DataFrame(conf_rows)
confusions.to_csv(ART / "comparison_top_confusions.csv", index=False)

# agreement between models
agree = {
    "All three correct": int(np.sum(np.all([pred[m] == y_test for m in MODELS], axis=0))),
    "All three wrong": int(np.sum(np.all([pred[m] != y_test for m in MODELS], axis=0))),
    "Only CNN correct": int(np.sum((pred["CNN"] == y_test) & (pred["SVM"] != y_test) & (pred["MLP"] != y_test))),
    "CNN wrong, another model correct": int(np.sum((pred["CNN"] != y_test) & ((pred["SVM"] == y_test) | (pred["MLP"] == y_test)))),
}


def save(fig, name):
    fig.savefig(ART / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_models(ax, ys):
    ax.set_yticks(ys, MODELS)
    for t, m in zip(ax.get_yticklabels(), MODELS):
        t.set_color(INK)


# ---------------------------------------------------------------- figure 1: accuracy & errors
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
metrics = [("val_acc", "Validation", "s"), ("test_acc", "Test", "o")]
for i, m in enumerate(MODELS):
    vals = [100 * res[m][k] for k, _, _ in metrics]
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=1)
    for (k, lab, mk), v in zip(metrics, vals):
        ax.scatter(v, i, s=70, marker=mk, color=COL[m], edgecolor=SURF, linewidth=2, zorder=3,
                   label=lab if i == 0 else None)
    ax.annotate(f"{100 * res[m]['test_acc']:.2f}%", (100 * res[m]["test_acc"], i), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
label_models(ax, range(3))
ax.set_ylim(-0.6, 2.6); ax.set_xlim(98.0, 99.3)
ax.set_xlabel("Accuracy / score (%)  — axis zoomed to 98–99.3%")
ax.set_title("Accuracy: CNN leads on validation and test")
leg = ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3)
for h in leg.legend_handles: h.set_color(INK2)
ax.grid(axis="y", visible=False)

ax = axes[1]
errs = [res[m]["test_errors"] for m in MODELS]
bars = ax.barh(range(3), errs, color=[COL[m] for m in MODELS], height=0.55, edgecolor=SURF, linewidth=2)
for i, v in enumerate(errs):
    ax.text(v + 3, i, f"{v}", va="center", fontsize=10, color=INK)
label_models(ax, range(3))
ax.set_xlim(0, 215); ax.set_xlabel("Misclassified test images (of 10,000) — lower is better")
ax.set_title("Test errors")
ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "compare_accuracy.png")

# ---------------------------------------------------------------- figure 2: cost (small multiples)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
panels = [
    ("train_time_s", "Training time (s)", lambda v: f"{v:.1f} s", False),
    ("inference_ms_per_image", "Inference (ms / image, log)", lambda v: f"{v:.4f}" if v < 0.01 else f"{v:.2f}", True),
    ("file_mb", "Saved file size (MB)", lambda v: f"{v:.2f}", False),
    ("complexity", "Stored values / params", lambda v: f"{v / 1e6:.2f}M" if v > 1e6 else f"{v / 1e3:.0f}k", False),
]
for ax, (k, title, fmt, log) in zip(axes, panels):
    vals = [res[m][k] for m in MODELS]
    ax.barh(range(3), vals, color=[COL[m] for m in MODELS], height=0.55, edgecolor=SURF, linewidth=2)
    if log:
        ax.set_xscale("log"); ax.set_xlim(min(vals) / 2, max(vals) * 12)
    else:
        ax.set_xlim(0, max(vals) * 1.35)
    for i, v in enumerate(vals):
        ax.text(v * (1.15 if log else 1) + (0 if log else max(vals) * 0.03), i, fmt(v), va="center", fontsize=9, color=INK)
    label_models(ax, range(3)); ax.invert_yaxis()
    ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
    if k == "complexity":
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Cost: SVM trains on CPU and is the slowest and largest  (lower is better)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout(); save(fig, "compare_cost.png")

# ---------------------------------------------------------------- figure 3: learning curves
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for m in ["MLP", "CNN"]:
    h = hist[m]; ep = np.arange(1, len(h["val_acc"]) + 1)
    axes[0].plot(ep, 100 * np.array(h["val_acc"]), color=COL[m], lw=2, label=m)
    b = h["best_epoch"]
    axes[0].scatter(b, 100 * h["val_acc"][b - 1], s=64, color=COL[m], edgecolor=SURF, linewidth=2, zorder=3)
    axes[0].annotate(f"{m} best: epoch {b}", (b, 100 * h["val_acc"][b - 1]), xytext=(4, 8),
                     textcoords="offset points", fontsize=8, color=INK)
    axes[1].plot(ep, h["train_loss"], color=COL[m], lw=2, label=m)
    axes[1].annotate(m, (ep[-1], h["train_loss"][-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=9, color=INK, va="center")
svm_val = 100 * res["SVM"]["val_acc"]
axes[0].axhline(svm_val, color=COL["SVM"], lw=2, ls="--", label=f"SVM {svm_val:.2f}% (no epochs)")
axes[0].set_ylim(96, 99.3); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation accuracy (%)")
axes[0].set_title("Validation accuracy per epoch")
axes[0].legend(frameon=False, loc="lower right", fontsize=9)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training loss"); axes[1].set_ylim(bottom=0)
axes[1].set_title("Training loss per epoch")
axes[1].legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); save(fig, "compare_learning_curves.png")

# ---------------------------------------------------------------- figure 4: per-digit F1
fig, ax = plt.subplots(figsize=(11, 3.8))
w = 0.26
for j, m in enumerate(MODELS):
    ax.bar(np.arange(10) + (j - 1) * w, 100 * pc[m]["f1"], width=w, color=COL[m], edgecolor=SURF,
           linewidth=1.5, label=m)
ax.set_xticks(range(10), [str(d) for d in range(10)])
ax.set_ylim(97, 100); ax.set_xlabel("Digit"); ax.set_ylabel("F1 score (%) — axis starts at 97%")
ax.set_title("Per-digit F1 on the test set: CNN is best on 9 of 10 digits; 9 is hard for every model")
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "compare_per_digit_f1.png")

# ---------------------------------------------------------------- figure 5: error matrices
seq = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#a9c8ef", "#2a78d6", "#0b3f80"])
vmax = max(int(np.max(cms[m] - np.diag(np.diag(cms[m])))) for m in MODELS)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
for ax, m in zip(axes, MODELS):
    off = cms[m].astype(float); np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap=seq, vmin=0, vmax=vmax)
    for t in range(10):
        for p_ in range(10):
            if t != p_ and cms[m][t, p_] > 0:
                v = cms[m][t, p_]
                ax.text(p_, t, str(v), ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > vmax * 0.5 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False)
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    ax.set_title(f"{m}: {res[m]['test_errors']} errors")
    for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.8, label="Misclassified images")
fig.suptitle("Where each model goes wrong (correct predictions on the diagonal hidden)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
save(fig, "compare_error_matrices.png")

# ---------------------------------------------------------------- markdown report
def md_table(df, index=True):
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and abs(v) >= 1000
                               else str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])

hp = res["hyperparameters"]
hp_df = pd.DataFrame([
    {"Model": "SVM", "Selected hyper-parameters": f"RBF kernel, C={hp['SVM']['C']}, gamma={hp['SVM']['gamma']}"},
    {"Model": "MLP", "Selected hyper-parameters": f"hidden {tuple(hp['MLP']['hidden'])}, lr={hp['MLP']['lr']}, dropout={hp['MLP']['dropout']}, batch={hp['MLP']['batch']}, weight decay={hp['MLP']['wd']}"},
    {"Model": "CNN", "Selected hyper-parameters": f"filters {tuple(hp['CNN']['filters'])}, lr={hp['CNN']['lr']}, dropout={hp['CNN']['dropout']}, batch={hp['CNN']['batch']}, weight decay={hp['CNN']['wd']}"},
])
acc_cols = ["Hardware", "Test accuracy (%)", "Validation accuracy (%)", "Train-subset accuracy (%)",
            "Macro precision (%)", "Macro recall (%)", "Macro F1 (%)", "Test errors (of 10,000)"]
cost_cols = ["Training time (s)", "Inference (ms / image)", "Stored values / parameters", "File size (MB)", "Best epoch"]
agree_df = pd.DataFrame({"Test images": agree}).rename_axis("Outcome")

md = f"""# MNIST model comparison

Models trained {pd.Timestamp.now():%Y-%m-%d} on the 54,000-image training split, evaluated once on the 10,000-image
MNIST test set. MLP and CNN were trained on the GPU (RTX 5070); the SVM uses scikit-learn, which only runs on the CPU.

**Verdict:** the CNN is the best model. It has the highest accuracy (99.07%), the fewest errors (93 vs 149 for the SVM and 187
for the MLP) and the shortest training time (15.6 s).

## 1. Accuracy

{md_table(overall[acc_cols])}

![Accuracy and errors](compare_accuracy.png)

## 2. Cost

{md_table(overall[cost_cols])}

Training time includes only the final model, not the hyper-parameter search. For the SVM, "stored values" counts the numbers held
in its support vectors.

![Cost comparison](compare_cost.png)

## 3. Training curves

![Learning curves](compare_learning_curves.png)

## 4. Per-digit F1 (%)

{md_table(per_digit)}

![Per-digit F1](compare_per_digit_f1.png)

## 5. Most frequent mistakes

{md_table(confusions, index=False)}

![Error matrices](compare_error_matrices.png)

## 6. Agreement between models

{md_table(agree_df)}

## 7. Selected hyper-parameters

{md_table(hp_df, index=False)}
"""
(ART / "model_comparison.md").write_text(md, encoding="utf-8")
print(overall[acc_cols + cost_cols].to_string())
print(per_digit.to_string())
print(agree)
