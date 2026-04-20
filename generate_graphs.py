"""
Generates figures for the paper based on Table 1 results.

TO USE: Replace the placeholder values in RESULTS with actual experiment results.

Figures generated:
  fig1_accuracy_by_model.png     — accuracy across all 5 evaluation sets
  fig2_robustness_score.png      — robustness score per model (distracted / clean)
  fig3_overfiltering_score.png   — over-filtering score per model
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Replace these values with actual results later
# All values are accuracy (%) on each evaluation set.
# Rows = models, Columns = [GSM8K clean, GSM-Plus, GSM-DC, Template, Info-Dense]

RESULTS = {
    "0.5B Clean":   [None, None, None, None, None],
    "0.5B Mixed":   [None, None, None, None, None],
    "0.5B Hard":    [None, None, None, None, None],
    "1.5B Clean":   [None, None, None, None, None],
    "1.5B Mixed":   [None, None, None, None, None],
}

EVAL_SETS = ["GSM8K\n(Clean)", "GSM-Plus\n(OOD)", "GSM-DC\n(OOD)", "Template\n(ID)", "Info-Dense\n(Over-filter)"]

COLORS = {
    "0.5B Clean": "#4C72B0",
    "0.5B Mixed": "#DD8452",
    "0.5B Hard":  "#55A868",
    "1.5B Clean": "#C44E52",
    "1.5B Mixed": "#8172B2",
}


def check_results() -> bool:
    """Return True if all results have been filled in."""
    for model, scores in RESULTS.items():
        if any(s is None for s in scores):
            print(f"[warn] {model} has missing results — using placeholder values for preview.")
            return False
    return True


def fill_placeholder() -> dict:
    """Fill None values with fake data for preview purposes."""
    import random
    random.seed(42)
    filled = {}
    for model, scores in RESULTS.items():
        filled[model] = [
            s if s is not None else round(random.uniform(30, 75), 1)
            for s in scores
        ]
    return filled


def compute_robustness(results: dict) -> dict:
    """
    Robustness score = accuracy on distracted set / accuracy on clean GSM8K.
    Computed for GSM-Plus, GSM-DC, and Template sets.
    """
    robustness = {}
    distracted_sets = [1, 2, 3]  # indices into EVAL_SETS
    set_names = ["GSM-Plus", "GSM-DC", "Template"]

    for model, scores in results.items():
        clean_acc = scores[0]
        if clean_acc and clean_acc > 0:
            robustness[model] = {
                name: round(scores[i] / clean_acc, 3)
                for i, name in zip(distracted_sets, set_names)
            }
    return robustness


def compute_overfiltering(results: dict) -> dict:
    """
    Over-filtering score = distractor-trained accuracy on info-dense
                           / clean-trained accuracy on info-dense.
    Baseline models: 0.5B Clean and 1.5B Clean.
    """
    overfiltering = {}
    baseline_05 = results["0.5B Clean"][4]
    baseline_15 = results["1.5B Clean"][4]

    for model, scores in results.items():
        if "Clean" in model:
            continue
        baseline = baseline_05 if "0.5B" in model else baseline_15
        if baseline and baseline > 0:
            overfiltering[model] = round(scores[4] / baseline, 3)
    return overfiltering


# Figure 1: accuracy across all evaluation sets

def plot_accuracy(results: dict, output_path: str = "fig1_accuracy_by_model.png") -> None:
    """Bar chart showing accuracy of each model on each evaluation set."""
    models = list(results.keys())
    n_models = len(models)
    n_sets = len(EVAL_SETS)
    x = np.arange(n_sets)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, model in enumerate(models):
        scores = results[model]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=model, color=COLORS[model], alpha=0.85)

    ax.set_xlabel("Evaluation Set")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Across Evaluation Sets")
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_SETS)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


# Figure 2: robustness score

def plot_robustness(results: dict, output_path: str = "fig2_robustness_score.png") -> None:
    """
    Bar chart of robustness scores.
    Robustness = accuracy on distracted set / accuracy on clean GSM8K.
    Score of 1.0 = distractors have no effect.
    """
    robustness = compute_robustness(results)
    models = list(robustness.keys())
    set_names = ["GSM-Plus", "GSM-DC", "Template"]
    x = np.arange(len(set_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        scores = [robustness[model][s] for s in set_names]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=model, color=COLORS[model], alpha=0.85)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="No effect (1.0)")
    ax.set_xlabel("Evaluation Set")
    ax.set_ylabel("Robustness Score")
    ax.set_title("Robustness Score by Model\n(distracted accuracy / clean accuracy)")
    ax.set_xticks(x)
    ax.set_xticklabels(set_names)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


# Figure 3: over-filtering score

def plot_overfiltering(results: dict, output_path: str = "fig3_overfiltering_score.png") -> None:
    """
    Bar chart of over-filtering scores.
    Over-filtering = distractor-trained info-dense accuracy / clean-trained info-dense accuracy.
    Score below 1.0 = model over-filters relevant context.
    Score at or above 1.0 = no over-filtering.
    """
    overfiltering = compute_overfiltering(results)
    models = list(overfiltering.keys())
    scores = [overfiltering[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(models, scores, color=[COLORS[m] for m in models], alpha=0.85, width=0.5)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="No over-filtering (1.0)")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Over-filtering Score")
    ax.set_title("Over-filtering Score by Model\n(distractor-trained / clean-trained on Info-Dense)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 1.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {output_path}")


# main

if __name__ == "__main__":
    ready = check_results()
    results = RESULTS if ready else fill_placeholder()

    if not ready:
        print("Running with placeholder data for preview.")
        print("Replace None values in RESULTS with actual experiment results.\n")

    plot_accuracy(results)
    plot_robustness(results)
    plot_overfiltering(results)

    print("\nDone! Generated:")
    print("  fig1_accuracy_by_model.png")
    print("  fig2_robustness_score.png")
    print("  fig3_overfiltering_score.png")
