import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Year'] = pd.to_datetime(df['Crop_Year'], format='%Y')
    df = df.sort_values(['State', 'Crop', 'Year'])
    df = df.rename(columns={
        'Crop': 'Crop Name',
        'Annual_Rainfall': 'Rain',
        'Yield': 'Crop Yield'
    })
    return df

# Create and train the model
@st.cache_resource
def create_and_train_model(df):
    le = LabelEncoder()
    df['State_encoded'] = le.fit_transform(df['State'])
    df['Crop_Name_encoded'] = le.fit_transform(df['Crop Name'])
    
    X = df[['State_encoded', 'Crop_Name_encoded', 'Rain', 'Pesticide', 'Fertilizer']]
    y = df['Crop Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le

# Make predictions
def make_prediction(model, le, state, crop_name, rain, pesticide, fertilizer):
    state_encoded = le.transform([state])[0]
    crop_name_encoded = le.transform([crop_name])[0]
    
    input_data = [[state_encoded, crop_name_encoded, rain, pesticide, fertilizer]]
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Crop Yield Prediction")
    
    # Load data and train model
    df = load_and_preprocess_data('crop_yield.csv')
    model, le = create_and_train_model(df)
    
    # User inputs
    state = st.selectbox("Select State", df['State'].unique())
    crop_name = st.selectbox("Select Crop", df['Crop Name'].unique())
    rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0)
    pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, max_value=1000.0, value=100.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=1000.0, value=200.0)

    if st.button("Predict"):
        prediction = make_prediction(model, le, state, crop_name, rain, pesticide, fertilizer)
        
        st.subheader("Prediction Results")
        st.write(f"Predicted Crop Yield: {prediction:.2f}")

        # Plot feature importance
        feature_importance = model.feature_importances_
        features = ['State', 'Crop Name', 'Rain', 'Pesticide', 'Fertilizer']
        fig, ax = plt.subplots()
        ax.bar(features, feature_importance)
        ax.set_ylabel("Feature Importance")
        ax.set_title("Feature Importance for Crop Yield Prediction")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
