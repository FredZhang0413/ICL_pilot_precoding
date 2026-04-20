import numpy as np
import matplotlib.pyplot as plt


def plot_mismatch_heatmap() -> None:
    """Plot 4x4 context-query mismatch heatmap from recorded average rates."""
    scenario_labels = [
        "S0 Ultra Dense",
        "S1 Near-Field",
        "S2 Far-Field",
        "S3 Rayleigh",
    ]

    # Rows = query scenario, Columns = context scenario
    rate_matrix = np.array(
        [
            [30.17, 26.37, 26.44, 27.05],
            [22.43, 25.76, 24.47, 22.85],
            [22.59, 25.13, 26.16, 22.69],
            [24.62, 22.49, 22.61, 26.74],
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rate_matrix, cmap="YlGnBu", aspect="equal")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Test Rate")

    ax.set_xticks(np.arange(len(scenario_labels)))
    ax.set_yticks(np.arange(len(scenario_labels)))
    ax.set_xticklabels(scenario_labels)
    ax.set_yticklabels(scenario_labels)

    ax.set_xlabel("Context Scenario")
    ax.set_ylabel("Query Scenario")
    ax.set_title("Mismatch 2D Heatmap (Context vs Query)")

    # Cell annotations
    for i in range(rate_matrix.shape[0]):
        for j in range(rate_matrix.shape[1]):
            ax.text(j, i, f"{rate_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

    # Build percentage matrix by row-wise reference (max value in each row = 100%).
    row_ref = np.max(rate_matrix, axis=1, keepdims=True)
    percent_matrix = rate_matrix / row_ref * 100.0

    print("Row-wise referenced percentage matrix (%):")
    print(np.round(percent_matrix, 2))

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # Most values are concentrated in a high-percent band; tighten color range
    # so relative differences are visually amplified.
    im2 = ax2.imshow(percent_matrix, cmap="YlOrRd", aspect="equal", vmin=80, vmax=100)

    cbar2 = fig2.colorbar(im2, ax=ax2)
    cbar2.set_label("Relative Rate (%)")

    ax2.set_xticks(np.arange(len(scenario_labels)))
    ax2.set_yticks(np.arange(len(scenario_labels)))
    ax2.set_xticklabels(scenario_labels)
    ax2.set_yticklabels(scenario_labels)

    ax2.set_xlabel("Context Scenario")
    ax2.set_ylabel("Query Scenario")
    ax2.set_title("Mismatch 2D Heatmap (Row-wise Referenced %)")

    for i in range(percent_matrix.shape[0]):
        for j in range(percent_matrix.shape[1]):
            ax2.text(j, i, f"{percent_matrix[i, j]:.1f}%", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mismatch_heatmap()
