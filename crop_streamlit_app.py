import streamlit as st
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

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
@st.cache_resource
def create_and_train_model(df):
    max_prediction_length = 1
    max_encoder_length = 20
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

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_model_summary=True,
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

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return model, training

# Plot feature importance
def plot_feature_importance(model, prediction):
    importance = model.interpret_output(prediction, reduction="sum")
    
    fig = go.Figure(go.Bar(
        x=importance.mean(0).tolist(),
        y=importance.columns.tolist(),
        orientation='h'
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    
    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.title("Crop Yield Prediction App")

    # Load data
    df = load_data()

    # Create and train model
    with st.spinner("Training model... This may take a few minutes."):
        model, training = create_and_train_model(df)

    st.success("Model trained successfully!")

    # User input
    st.sidebar.header("Input Parameters")
    state = st.sidebar.selectbox("State", df['State'].unique())
    crop = st.sidebar.selectbox("Crop", df['Crop Name'].unique())
    rain = st.sidebar.slider("Annual Rainfall (mm)", float(df['Rain'].min()), float(df['Rain'].max()), float(df['Rain'].mean()))
    pesticide = st.sidebar.slider("Pesticide (kg/ha)", float(df['Pesticide'].min()), float(df['Pesticide'].max()), float(df['Pesticide'].mean()))
    fertilizer = st.sidebar.slider("Fertilizer (kg/ha)", float(df['Fertilizer'].min()), float(df['Fertilizer'].max()), float(df['Fertilizer'].mean()))

    # Make prediction
    if st.button("Predict Yield"):
        # Prepare input data
        last_data = df[(df['State'] == state) & (df['Crop Name'] == crop)].iloc[-1]
        input_data = pd.DataFrame({
            'State': [state],
            'Crop Name': [crop],
            'time_idx': [last_data['time_idx'] + 1],
            'Rain': [rain],
            'Pesticide': [pesticide],
            'Fertilizer': [fertilizer],
            'Crop Yield': [np.nan]
        })

        # Add historical data
        history = df[(df['State'] == state) & (df['Crop Name'] == crop)].iloc[-20:]
        input_data = pd.concat([history, input_data]).reset_index(drop=True)

        # Make prediction
        prediction = model.predict(input_data, mode="raw")
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"Predicted Crop Yield: {prediction.output[0][0][0]:.2f} kg/ha")

        # Plot feature importance
        st.subheader("Feature Importance")
        plot_feature_importance(model, prediction)

if __name__ == "__main__":
    main()
