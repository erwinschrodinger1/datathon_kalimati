import streamlit as st
import pandas as pd
from src.seasonality_graph import seasonality_graph
from src.dataset_imputation import dataset_imputation
from src.prediction import prediction


@st.cache_resource()
def load_data():
    dataset = pd.read_csv("./kalimati_tarkari_dataset_cleaned.csv")
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    # dataset = dataset_imputation(dataset.copy())
    return dataset


data = load_data()

st.markdown("# Analysis of Kalimati Tarkari Bazaar")

st.markdown("## Overall Seasonality Analysis")


selected_commodity = st.selectbox(
    label="List of Commodities", options=data["Commodity"].unique()
)

submit = st.button("Submit")

if submit:
    st.markdown(f"### Seasonality Graph for {selected_commodity} Commodity")
    st.line_chart(
        seasonality_graph(data.copy(), selected_commodity),
        x_label="Date",
        y_label="Average Price",
    )

    st.markdown(f"### Predicted Graph for {selected_commodity} Commodity")

    with st.spinner("Training Model..."):
        st.line_chart(
            prediction(data.copy(), selected_commodity),
        )
