import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt


def seasonality_graph(dataset, commodity):
    monthly_avg = (
        dataset.groupby("Commodity").resample("ME", on="Date")["Average"].mean()
    )
    return monthly_avg.loc[commodity]
