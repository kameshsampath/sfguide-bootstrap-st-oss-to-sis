import streamlit as st

# import pandas to read the our data file
import pandas as pd

st.title("🤖 Machine Learning App")

st.write("Welcome to world of Machine Learning with Streamlit.")

with st.expander("Data"):
    st.write("**Raw Data**")
    # read the csv file
    df = pd.read_csv("data/penguins_cleaned.csv")
    df
    # define and display
    st.write("**X**")
    x = df.drop("species", axis=1)
    x

    st.write("**y**")
    y = df.species
    y

with st.expander("Data"):
    st.scatter_chart(
        df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

# Ineractivity
# Columns:
# 'species', 'island', 'bill_length_mm', 'bill_depth_mm',
# 'flipper_length_mm', 'body_mass_g', 'sex'
with st.sidebar:
    st.header("Input Features")
    islands = df.island.unique().astype(str)
    island = st.selectbox(
        "Island",
        islands,
    )
    gender = st.radio(
        "Gender",
        ("Male", "Female"),
    )
    min, max, mean = (
        df.bill_length_mm.min(),
        df.bill_length_mm.max(),
        df.bill_length_mm.mean().round(2),
    )
    bill_length_mm = st.slider(
        "Bill Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    min, max, mean = (
        df.bill_depth_mm.min(),
        df.bill_depth_mm.max(),
        df.bill_depth_mm.mean().round(2),
    )
    bill_depth_mm = st.slider(
        "Bill Depth(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    min, max, mean = (
        df.filpper_length_mm.min(),
        df.filpper_length_mm.max(),
        df.filpper_length_mm.mean().round(2),
    )
    filpper_length_mm = st.slider(
        "Filpper Length(mm)",
        min_value=min,
        max_value=max,
        value=mean,
    )
    min, max, mean = (
        df.filpper_length_mm.min(),
        df.filpper_length_mm.max(),
        df.filpper_length_mm.mean().round(2),
    )
    body_mass_g = st.slider(
        "Body Mass(g)",
        min_value=min,
        max_value=max,
        value=mean,
    )
