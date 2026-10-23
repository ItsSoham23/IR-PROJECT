"""Plot NPMI values from a poison-budget result file."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_coherence(
    input_path="results/poison_budget_curve.json",
    output_path="results/coherence_curve.png",
):
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    fig, axis = plt.subplots(figsize=(6, 4), dpi=180)
    budgets = payload["budgets"]
    for label, key in (("PLSA", "plsa"), ("Tempered PLSA", "plsa_tempered")):
        axis.plot(budgets, [row[key]["npmi_mean"] for row in payload["runs"]], marker="o", label=label)
    axis.set(xlabel="Poison documents", ylabel="Mean NPMI", title="Topic coherence under poisoning")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_coherence()
