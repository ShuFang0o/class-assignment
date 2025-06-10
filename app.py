import streamlit as st
from src.inference import get_prediction, get_top_features

# App State Initialization
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

# Sidebar UI
def app_sidebar():
    st.sidebar.header("Listing Details")

    # Inputs
    room_type_str = st.sidebar.selectbox("Room Type", ['Entire home/apt', 'Hotel Room', 'Private room', 'Shared room'])
    room_type_map = {
        'Entire home/apt': 1,
        'Hotel Room': 2,
        'Private room': 3,
        'Shared room': 4
    }
    room_type = room_type_map[room_type_str]

    accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)
    bathrooms = st.sidebar.slider("Bathrooms", 0, 5, 1)
    beds = st.sidebar.slider("Beds", 1, 10, 2)
    minimum_nights = st.sidebar.slider("minimum_nights", 0, 365,30)
    maximum_nights = st.sidebar.slider("maximum_nights", 0, 365,30)
    review_scores_rating = st.sidebar.slider("Review Score Rating", 0.0, 100.0, 90.0)

    bed_linens = st.sidebar.checkbox('Bed linens')
    bed_linens_encoded = int(bed_linens)
    tv = st.sidebar.checkbox('TV')
    tv_encoded = int(tv)
    self_checkin = st.sidebar.checkbox('Self check-in')
    self_checkin_encoded = int(self_checkin)
    air_conditioning = st.sidebar.checkbox('Air Conditioning')
    air_conditioning_encoded = int(air_conditioning)
    hot_water = st.sidebar.checkbox('Hot Water')
    hot_water_encoded = int(hot_water)
    refrigerator = st.sidebar.checkbox('Refrigerator')
    refrigerator_encoded = int(refrigerator)
    wifi = st.sidebar.checkbox('Wifi')
    wifi_encoded = int(wifi)
    bidet = st.sidebar.checkbox('Bidet')
    bidet_encoded = int(bidet)
    elevator = st.sidebar.checkbox('Elevator')
    elevator_encoded = int(elevator)

    price_range_str = st.sidebar.selectbox("Price Range", ['0-100', '101-200', '201-300', '301-500', '501-1000', '>1000'])
    price_range_map = {
        '0-100': 0,
        '101-200': 1,
        '201-300': 2,
        '301-500': 3,
        '501-1000': 4,
        '>1000': 5
    }
    price_range = price_range_map[price_range_str]



    def get_input_features():
        input_data = {
            'room_type': room_type,
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'beds': beds,
            'minimum_nights': minimum_nights,
            'maximum_nights': maximum_nights,
            'review_scores_rating': review_scores_rating,
            'Bed linens': bed_linens,
            'TV': tv,
            'Self check-in': self_checkin,
            'Air conditioning': air_conditioning,
            'Hot water': hot_water,
            'Refrigerator': refrigerator,
            'Wifi': wifi,
            'Bidet': bidet,
            'Elevator': elevator,
            'price_range_coded': price_range
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
    title = '<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;"><b>🏡 Welcome to Airbnb Occupancy Rate Predictor</b></p>'
    st.markdown(title, unsafe_allow_html=True)

    default_msg = '**Predicted occupancy rate:** {}'

    if st.session_state['input_features']:
        input_data = st.session_state['input_features']
        prediction = get_prediction(**input_data)

        st.success(default_msg.format(f"{prediction:.2f}%"))

        # Show top 5 important features
        st.markdown("<h4 style='margin-top: 30px;'>⭐ Top 5 Important Features</h4>", unsafe_allow_html=True)
        top_features_df = get_top_features(top_n=5)
        st.dataframe(top_features_df)

    return None

# Main entry
def main():
    app_sidebar()
    app_body()

if __name__ == "__main__":
    main()