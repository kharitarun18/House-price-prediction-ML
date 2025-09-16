import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load Models
# -----------------------
xgb_model = joblib.load("C:/Users/LSPC/OneDrive/Desktop/projects/HousePrice/xgb_house_model.pkl")

# (Optional) If you have more models, load them here
# rf_model = joblib.load("C:/Users/LSPC/OneDrive/Desktop/projects/HousePrice/rf_house_model.pkl")

# -----------------------
# Load Dataset
# -----------------------
data = pd.read_csv("C:/Users/LSPC/Downloads/realest.csv")

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using real dataset examples or custom inputs.")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Choose Model", ["XGBoost"])  # Add "RandomForest" if available

# Example selector
col1, col2 = st.sidebar.columns([2, 1])
example_index = col1.selectbox(
    "Choose an example house",
    options=list(data.index),
    format_func=lambda x: f"Example {x+1}"
)
if col2.button("🎲 Random Example"):
    example_index = np.random.choice(data.index)

example_row = data.loc[example_index]

# -----------------------
# Layout (Inputs / Results)
# -----------------------
input_col, result_col = st.columns([2, 2])

with input_col:
    st.subheader("📝 Input Features")
    bedroom = st.slider("Bedrooms", 0, 10, int(example_row['Bedroom']))
    space = st.number_input("Space (sq ft)", 0.0, 10000.0, float(example_row['Space']))
    room = st.slider("Rooms", 0, 15, int(example_row['Room']))
    lot = st.number_input("Lot Size", 0.0, 20000.0, float(example_row['Lot']))
    tax = st.number_input("Tax Amount", 0.0, 100000.0, float(example_row['Tax']))
    bathroom = st.slider("Bathrooms", 0, 10, int(example_row['Bathroom']))
    garage = st.slider("Garage Size", 0, 5, int(example_row['Garage']))
    condition = st.slider("Condition (1=Poor, 10=Excellent)", 1, 10, int(example_row['Condition']))

    features = np.array([[bedroom, space, room, lot, tax, bathroom, garage, condition]])

    if st.button("🔮 Predict Price"):
        if model_choice == "XGBoost":
            model = xgb_model
        # elif model_choice == "RandomForest":
        #     model = rf_model

        prediction = model.predict(features)[0]

        with result_col:
            st.subheader("📊 Prediction Result")
            st.success(f"💰 Predicted House Price: ₹ {round(prediction):,}")

            avg_price = data['Price'].mean() if 'Price' in data.columns else prediction
            st.write(f"📉 Dataset Average Price: ₹ {round(avg_price):,}")

            # Bar chart: Predicted vs Average
            fig, ax = plt.subplots()
            ax.bar(["Predicted", "Average"], [prediction, avg_price], color=["green", "blue"])
            ax.set_ylabel("Price (₹)")
            st.pyplot(fig)

# -----------------------
# Extra Insights
# -----------------------
st.subheader("📑 Selected Example Data")
st.table(example_row[['Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage', 'Condition']])

st.subheader("📈 Dataset Insights")
col3, col4 = st.columns(2)

with col3:
    st.write("### Summary Statistics")
    st.write(data.describe())

with col4:
    if 'Price' in data.columns:
        st.write("### Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Price'], kde=True, ax=ax)
        st.pyplot(fig)

# Correlation Heatmap
if st.checkbox("🔎 Show Feature Correlation Heatmap"):
    numeric_features = data.select_dtypes(include=[np.number])
    corr = numeric_features.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

