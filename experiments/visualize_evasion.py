"""Plot test-time evasion results."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_evasion(
    input_path="results/evasion_summary.json",
    output_path="results/evasion_curve.png",
):
    with open(input_path, encoding="utf-8") as handle:
        records = json.load(handle)["runs"]
    fig, axis = plt.subplots(figsize=(6, 4), dpi=180)
    axis.plot([row["swaps"] for row in records], [row["macro_f1"] for row in records], marker="o", label="PLSA")
    axis.set(xlabel="Swapped tokens", ylabel="Macro-F1", title="Test-time evasion")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_evasion()
