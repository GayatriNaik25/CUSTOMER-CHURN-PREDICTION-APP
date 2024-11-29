import pandas as pd
import joblib
import base64
import streamlit as st  # Importing Streamlit

# Load the default trained model
model = joblib.load(r"D:\DATA SCIENCE CLASS NOTES\ml\Customer_churn_project\final_gb_classifier.pkl")

# Function to add background from a local image path
def add_bg_from_local(image_path):
    # Read the image and encode it in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Use the encoded image in CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image (replace with your own image path)
add_bg_from_local(r"C:\Users\new\Pictures\images.jpg")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Return preprocessed DataFrame
    return df

# Streamlit UI
st.title("Customer Churn Prediction")

# Collect user inputs
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.radio("Partner", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dependents = st.radio("Dependents", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
phone_service = st.radio("Phone Service", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
multiple_lines = st.radio("Multiple Lines", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.radio("Online Security", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
online_backup = st.radio("Online Backup", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
device_protection = st.radio("Device Protection", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
tech_support = st.radio("Tech Support", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
streaming_tv = st.radio("Streaming TV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
streaming_movies = st.radio("Streaming Movies", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", value=0.0, min_value=0.0)
total_charges = st.number_input("Total Charges", value=0.0, min_value=0.0)
tenure_group = st.number_input("Tenure Group", value=0, min_value=0)

# Define the models (corrected paths with double backslashes)
models = {
    "Logistic Regression": r"C:\Users\new\Downloads\logistic_regression_model.pkl",
    "Decision Tree Classifier": r"C:\Users\new\Downloads\decision_tree_model.pkl",
    "Random Forest Classifier": r"C:\Users\new\Downloads\random_forest_model (1).pkl",
    "AdaBoost Classifier": r"C:\Users\new\Downloads\adaboost_model.pkl",
    "Gradient Boosting Classifier": r"C:\Users\new\Downloads\gradient_boost_model.pkl",
    "XGBoost Classifier": r"C:\Users\new\Downloads\xgboost_model.pkl",
    "Final Gradient Boosting Classifier": r"C:\Users\new\VS\RESUME PROJCT\Customer_Churn_Project\final_gb_classifier.pkl",
    "Final XGBoosting Classifier": r"C:\Users\new\Downloads\final_xgb_model.pkl"
}

# Select the model
selected_model = st.selectbox("Choose a model", list(models.keys()))

# Load the selected model
model = joblib.load(models[selected_model])

# Make prediction when the button is clicked
if st.button("Predict"):
    # Create dictionary from user inputs
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }
    
    # Preprocess the input data
    processed_data = preprocess_input(user_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.markdown(
            """
            <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
                <strong style="color: #d8000c; font-size: 16px;">❌ The customer is likely to churn.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #ccffcc; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
                <strong style="color: #4caf50; font-size: 16px;">✅ The customer is likely to stay.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
