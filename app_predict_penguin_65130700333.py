
import streamlit as st
import pickle

# Load the model and encoders
with open('model_penguin_65130700333.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Create a title for the app
st.title("Penguin Species Prediction")

# Create input fields for user input
species = st.selectbox("Species", species_encoder.classes_)
island = st.selectbox("Island", island_encoder.classes_)
sex = st.selectbox("Sex", sex_encoder.classes_)
bill_length_mm = st.number_input("Bill Length (mm)")
bill_depth_mm = st.number_input("Bill Depth (mm)")
flipper_length_mm = st.number_input("Flipper Length (mm)")
body_mass_g = st.number_input("Body Mass (g)")

# Create a button to trigger the prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = [[species, island, sex, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g]]

    # Encode categorical features
    input_data_encoded = [[species_encoder.transform([species])[0],
                           island_encoder.transform([island])[0],
                           sex_encoder.transform([sex])[0],
                           bill_length_mm,
                           bill_depth_mm,
                           flipper_length_mm,
                           body_mass_g]]

    # Make the prediction
    prediction = model.predict(input_data_encoded)

    # Display the prediction
    st.write("Predicted Species:", species_encoder.inverse_transform(prediction)[0])

