import pandas as pd
import streamlit as st
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Year'] = pd.to_datetime(df['Crop_Year'], format='%Y')
    df = df.sort_values(['State', 'Crop', 'Year'])
    df['time_idx'] = df.groupby(['State', 'Crop']).cumcount()
    df = df.rename(columns={
        'Crop': 'Crop Name',
        'Annual_Rainfall': 'Rain',
        'Yield': 'Crop Yield'
    })
    return df

# Create and train the model
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
        target_normalizer=GroupNormalizer(groups=["State", "Crop Name"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='cpu',
        enable_model_summary=True,
        gradient_clip_val=0.1,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return tft, validation

# Make predictions
def make_prediction(model, dataset, state, crop_name, rain, pesticide, fertilizer):
    # Create a sample input
    sample_input = dataset.get_sample(1)
    
    # Update the sample with user inputs
    sample_input["State"] = [state]
    sample_input["Crop Name"] = [crop_name]
    sample_input["Rain"][:, 0] = rain
    sample_input["Pesticide"][:, 0] = pesticide
    sample_input["Fertilizer"][:, 0] = fertilizer

    # Make prediction
    prediction = model.predict(sample_input)
    return prediction.numpy().flatten()

# Streamlit app
def streamlit_app(model, dataset, df):
    st.title("Crop Yield Prediction")

    # User inputs
    state = st.selectbox("Select State", df['State'].unique())
    crop_name = st.selectbox("Select Crop", df['Crop Name'].unique())
    rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0)
    pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, max_value=1000.0, value=100.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=1000.0, value=200.0)

    if st.button("Predict"):
        prediction = make_prediction(model, dataset, state, crop_name, rain, pesticide, fertilizer)
        
        st.subheader("Prediction Results")
        st.write(f"Predicted Crop Yield: {prediction[0]:.2f}")

        # Plot the prediction
        fig, ax = plt.subplots()
        ax.plot(range(1, len(prediction) + 1), prediction, marker='o')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Predicted Crop Yield")
        ax.set_title(f"Crop Yield Prediction for {crop_name} in {state}")
        st.pyplot(fig)

# Main function
def main():
    # Load and preprocess data
    df = load_and_preprocess_data('crop_yield.csv')

    # Create and train the model
    model, dataset = create_and_train_model(df)

    # Run Streamlit app
    streamlit_app(model, dataset, df)

if _name_ == "_main_":
    main()
