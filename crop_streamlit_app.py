import streamlit as st
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pytorch_lightning as pl

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('crop_yield.csv')
    df['Year'] = pd.to_datetime(df['Crop_Year'], format='%Y')
    df = df.sort_values(['State', 'Crop', 'Year'])
    df['time_idx'] = df.groupby(['State', 'Crop']).cumcount()
    df = df.rename(columns={
        'Crop': 'Crop Name',
        'Annual_Rainfall': 'Rain',
        'Yield': 'Crop Yield'
    })
    return df

# Create and train TFT model
def create_and_train_model(df):
    max_prediction_length = 6
    max_encoder_length = 24
    training_cutoff = df['time_idx'].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Crop Yield",
        group_ids=["State", "Crop Name"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["State", "Crop Name"],
        time_varying_known_reals=["time_idx", "Rain", "Pesticide", "Fertilizer"],
        time_varying_unknown_reals=["Crop Yield"],
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),
    )

    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, training.to_dataloader(train=True, batch_size=64))

    return model, training

# Streamlit app
st.title('Crop Yield Prediction with TFT')

# Load data
df = load_data()

# Train model (in practice, you'd want to save and load the trained model)
model, training = create_and_train_model(df)

# User input
st.sidebar.header('Input Parameters')
state = st.sidebar.selectbox('State', df['State'].unique())
crop = st.sidebar.selectbox('Crop', df['Crop Name'].unique())
rain = st.sidebar.slider('Annual Rainfall (mm)', 0, 5000, 1000)
pesticide = st.sidebar.slider('Pesticide (kg/ha)', 0, 100, 50)
fertilizer = st.sidebar.slider('Fertilizer (kg/ha)', 0, 500, 200)

# Make prediction
if st.button('Predict Yield'):
    # Create a sample input
    sample_input = pd.DataFrame({
        'State': [state],
        'Crop Name': [crop],
        'time_idx': [df['time_idx'].max() + 1],
        'Rain': [rain],
        'Pesticide': [pesticide],
        'Fertilizer': [fertilizer],
        'Crop Yield': [np.nan]  # We don't know the yield, that's what we're predicting
    })

    # Convert to the format expected by the model
    encoder_data = training.transform(sample_input)
    prediction = model.predict(encoder_data)

    st.success(f'Predicted Crop Yield: {prediction[0][0]:.2f} kg/ha')

st.write('Note: This is a simplified model for demonstration purposes. For accurate predictions, you would need a more comprehensive dataset and a fully trained model.')
