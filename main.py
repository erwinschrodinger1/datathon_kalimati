import streamlit as st
import pandas as pd
from src.seasonality_graph import seasonality_graph
from src.dataset_imputation import dataset_imputation
from src.prediction import prediction
from src.min_max_graph import min_max_graph
from src.llm_description import generate_description


@st.cache_resource()
def load_data():
    dataset = pd.read_csv("./data_clustered_cleaned.csv")
    raw_dataset = pd.read_csv("./kalimati_tarkari_dataset_cleaned.csv")
    raw_dataset["Date"] = pd.to_datetime(raw_dataset["Date"])
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    # dataset = dataset_imputation(dataset.copy())
    return raw_dataset, dataset


raw_data, data = load_data()

st.markdown("# Analysis of Kalimati Tarkari Bazaar")

st.markdown("## Overall Seasonality Analysis")

st.markdown("### Range of min")

selected_commodity = st.selectbox(
    label="List of Commodities", options=data["Commodity"].unique()
)

submit = st.button("Submit")

if submit:
    st.markdown("## Description based on dataset")
    with st.spinner("Generating Description..."):
        st.markdown(f"{generate_description(data, selected_commodity)}")
    st.markdown(f"### Price Graph for {selected_commodity} Commodity")
    st.pyplot(min_max_graph(data.copy(), selected_commodity))
    st.markdown(
        f"### Seasonality Graph for {selected_commodity} Commodity averged over months"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.header("Raw Data")
        st.line_chart(
            seasonality_graph(raw_data.copy(), selected_commodity),
            x_label="Date",
            y_label="Average Price",
        )
    with col2:
        st.header("Imputed Data")
        st.line_chart(
            seasonality_graph(data.copy(), selected_commodity),
            x_label="Date",
            y_label="Average Price",
        )

    st.markdown(f"### Predicted Graph for {selected_commodity} Commodity")

    with st.spinner("Training Model..."):
        st.line_chart(
            x=data["Date"],
            y=prediction(data.copy(), selected_commodity),
            x_label="Date",
            y_label="Average Price",
        )
