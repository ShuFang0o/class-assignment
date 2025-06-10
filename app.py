import streamlit as st
import pandas as pd
from joblib import load
from src.inference import get_prediction, get_top_features  # You will define these

# Load model and feature names
model = load("airbnb_occupancy_model.joblib")
feature_names = load("airbnb_features.joblib")

# App State Initialization
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

# Sidebar UI
def app_sidebar():
    st.sidebar.header("Listing Information")

    # Example key inputs
    room_type = st.sidebar.selectbox("Room Type", ['Entire home/apt', 'Private room', 'Shared room'])
    accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)
    bathrooms = st.sidebar.slider("Bathrooms", 0, 5, 1)
    beds = st.sidebar.slider("Beds", 1, 10, 2)
    review_scores_rating = st.sidebar.slider("Review Score Rating", 0.0, 100.0, 90.0)

    def get_input_features():
        input_data = {
            'room_type': room_type,
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'beds': beds,
            'review_scores_rating': review_scores_rating,
            # Add more fields here as needed
        }
        return input_data

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Predict"):
        st.session_state['input_features'] = get_input_features()
    if col2.button("Reset"):
        st.session_state['input_features'] = {}

# Main body
def app_body():
    st.markdown("<h2>🏡 Airbnb Occupancy Rate Prediction</h2>", unsafe_allow_html=True)

    if st.session_state['input_features']:
        user_input = pd.DataFrame([st.session_state['input_features']])
        prediction = get_prediction(model, user_input)

        st.subheader("📈 Predicted Occupancy Rate")
        st.success(f"{prediction:.2f}%")

        # Top 15 features
        st.subheader("⭐ Top 15 Important Features")
        top_features_df = get_top_features(model, feature_names, top_n=15)
        st.dataframe(top_features_df)

# Main entry
def main():
    app_sidebar()
    app_body()

if __name__ == "__main__":
    main()