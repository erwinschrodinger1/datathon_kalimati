import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors


def min_max_graph(dataset, commodity):
    grouped_df = dataset.groupby("Commodity")
    print(grouped_df.get_group(commodity))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        grouped_df.get_group(commodity)["Date"],
        grouped_df.get_group(commodity)["Average"],
        c="brown",
    )
    ax.fill_between(
        grouped_df.get_group(commodity)["Date"],
        grouped_df.get_group(commodity)["Minimum"],
        grouped_df.get_group(commodity)["Maximum"],
        color=mpl_colors.to_rgba("brown", 0.15),
        label="Range (Min-Max)",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"{commodity} Average with Min-Max Range")
    ax.legend()

    return fig
