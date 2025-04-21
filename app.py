
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
st.set_page_config(
    page_title="Cognitive Score Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load Artifacts ---
# Ensure these paths are correct relative to where app.py is run
# When deploying on Streamlit Cloud, these files should be in the root of the GitHub repo
PREPROCESSOR_PATH = 'cognitive_score_preprocessor.joblib'
MODEL_PATH = 'cognitive_score_gbr_model.joblib'
ORIGINAL_COLUMNS = ['Age', 'Gender', 'Sleep_Duration', 'Stress_Level', 'Diet_Type', 'Daily_Screen_Time', 'Exercise_Frequency', 'Caffeine_Intake', 'Reaction_Time', 'Memory_Test_Score'] # Must match order used for training preprocessor
EXERCISE_ORDER = ['Low', 'Medium', 'High'] # Used for selectbox options

# Caching the loading process for efficiency
@st.cache_resource
def load_artifacts():
    """Loads the preprocessor and model."""
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        return preprocessor, model
    except FileNotFoundError:
        st.error(f"Error: Model or Preprocessor file not found. Ensure '{PREPROCESSOR_PATH}' and '{MODEL_PATH}' are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the artifacts: {e}")
        return None, None

preprocessor, model = load_artifacts()

# --- App Title and Description ---
st.title("🧠 Cognitive Score Predictor")
st.markdown("""
Enter the required details to predict an individual's cognitive score based on lifestyle and performance factors.
*This model is based on historical data and provides an estimate.*
""")
st.divider()

# --- User Input Section ---
st.header("Input Parameters")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Using number_input for flexibility, could use sliders too
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    sleep_duration = st.number_input("Sleep Duration (hours/day)", min_value=4.0, max_value=10.0, value=7.0, step=0.1, format="%.1f")
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5, step=1)
    daily_screen_time = st.number_input("Daily Screen Time (hours/day)", min_value=1.0, max_value=12.0, value=5.0, step=0.1, format="%.1f")
    reaction_time = st.number_input("Reaction Time (ms)", min_value=200.0, max_value=600.0, value=400.0, step=0.1, format="%.1f") # Based on original data range

with col2:
    gender = st.selectbox("Gender", options=['Female', 'Male', 'Other'], index=0)
    diet_type = st.selectbox("Diet Type", options=['Non-Vegetarian', 'Vegetarian', 'Vegan'], index=0)
    exercise_frequency = st.selectbox("Exercise Frequency", options=EXERCISE_ORDER, index=1) # Default 'Medium'
    caffeine_intake = st.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=500, value=250, step=1) # Based on original data range
    memory_test_score = st.number_input("Memory Test Score (40-100)", min_value=40, max_value=100, value=70, step=1) # Based on original data range

# --- Prediction Logic ---
if st.button("Predict Cognitive Score", type="primary") and preprocessor is not None and model is not None:
    # 1. Create DataFrame from input (MUST match the column order used for training the preprocessor)
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Sleep_Duration': [sleep_duration],
        'Stress_Level': [stress_level],
        'Diet_Type': [diet_type],
        'Daily_Screen_Time': [daily_screen_time],
        'Exercise_Frequency': [exercise_frequency],
        'Caffeine_Intake': [caffeine_intake],
        'Reaction_Time': [reaction_time],
        'Memory_Test_Score': [memory_test_score]
    }, columns=ORIGINAL_COLUMNS) # Enforce the exact column order

    st.write("---")
    st.subheader("Processing Input:")
    #st.dataframe(input_data) # Optionally display raw input

    try:
        # 2. Preprocess the input using the loaded preprocessor
        input_processed = preprocessor.transform(input_data)
        #st.write("Processed Input Shape:", input_processed.shape) # Debugging line

        # 3. Make prediction using the loaded model
        prediction = model.predict(input_processed)
        predicted_score = prediction[0] # Get the single prediction value

        # 4. Display the prediction
        st.subheader("✨ Predicted Cognitive Score")
        # Format the score nicely
        st.metric(label="Score (0-100)", value=f"{predicted_score:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

elif preprocessor is None or model is None:
    st.warning("Model artifacts could not be loaded. Prediction unavailable.")

# --- Optional: Add Footer or More Info ---
st.divider()
st.caption("Disclaimer: This prediction is based on a machine learning model and should be considered indicative, not definitive medical advice.")

# --- End of app.py ---
