import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt


def seasonality_graph(dataset, commodity):
    print(dataset.describe())
    monthly_avg = (
        dataset.groupby("Commodity").resample("ME", on="Date")["Average"].mean()
    )
    print(monthly_avg.describe())
    return monthly_avg.loc[commodity]
