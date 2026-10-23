"""Plot Macro-F1, MAP, perplexity, and NPMI from poison_budget.py."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_budget(
    input_path="results/poison_budget_curve.json",
    output_path="results/poison_budget_curve.png",
):
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    budgets = payload["budgets"]
    series = {
        "PLSA Macro-F1": ("plsa", "macro_f1"),
        "Tempered PLSA Macro-F1": ("plsa_tempered", "macro_f1"),
        "PLSA NPMI": ("plsa", "npmi_mean"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=180)
    for label, (model, metric) in series.items():
        values = [row[model][metric] for row in payload["runs"]]
        axes[0 if metric == "macro_f1" else 1].plot(budgets, values, marker="o", label=label)
    axes[0].set(xlabel="Poison documents", ylabel="Macro-F1", title="Classification sensitivity")
    axes[1].set(xlabel="Poison documents", ylabel="Mean NPMI", title="Topic coherence sensitivity")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_budget()
