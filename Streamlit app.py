import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import base64
import streamlit as st
import seaborn as sns

# Function to add background from a local image path
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
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

# Add background image
add_bg_from_local(r"C:\Users\new\Pictures\ppp.webp")

# Load the default trained model and dataset
model = joblib.load(r"C:\Users\new\Project\Data Science\Customer_Churn_Project\final_gb_classifier.pkl",'rb')
data = pd.read_csv(r"C:\Users\new\Project\Data Science\Customer_Churn_Project\Telco-Customer-Churn.csv")

# Preprocess input data
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    return df

# Streamlit UI
st.title("Customer Retention Analysis")

st.markdown("""
    <div style="text-align:center;">
    <div>
""", unsafe_allow_html=True)

st.subheader("📝 Description:")

st.write("This application uses machine learning to predict if a telecom customer is likely to stay or churn based on their service and account details,The app also displays helpful visualizations of churn trends across the customer base.")
st.subheader("📋 Dataset Preview :")
st.dataframe(data.head())

#Collect user inputes in the sidebar
st.sidebar.title("Customer Input Features")

gender = st.sidebar.selectbox("👤 Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
senior_citizen = st.sidebar.selectbox("🎂 Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.sidebar.selectbox("💍 Partne", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dependents = st.sidebar.selectbox("👶 Dependents", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
phone_service = st.sidebar.selectbox("📱 Phone Service", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
multiple_lines = st.sidebar.selectbox("📜Multiple Lines", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
internet_service = st.sidebar.selectbox("📶 Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.sidebar.selectbox("💻 Online Security", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
online_backup = st.sidebar.selectbox("☁️ Online Backup", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
device_protection = st.sidebar.selectbox("🛡️ Device Protection", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
tech_support = st.sidebar.selectbox("🛠️ Tech Support", [0, 1, 2], format_func=lambda x: "Yes" if x == 1 else "No" if x == 0 else "No internet service")
streaming_tv = st.sidebar.selectbox("📺 Streaming TV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
streaming_movies = st.sidebar.selectbox("🎥 Streaming Movies", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
contract = st.sidebar.selectbox("📝 Contract Type", ['Month-to-month', 'One year', 'Two year'])
paperless_billing =st.sidebar.selectbox("📬 Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
payment_method = st.sidebar.selectbox("💳 Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.sidebar.number_input("💸 Monthly Charge", value=0.0, min_value=0.0)
total_charges = st.sidebar.number_input("💸Total Charges", value=0.0, min_value=0.0)
tenure_group = st.sidebar.number_input("⏳ Tenure Group", value=0, min_value=0)

def plot_churn_vs_not_churn(churn_counts):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="pastel")
    plt.title("Total Customers: Churn vs Not Churn")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")
    plt.xticks(ticks=[0, 1], labels=["Not Churn", "Churn"])
    st.pyplot(plt)

# Pie Chart
def plot_churn_pie_chart(churn_counts):
    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=["Not Churn", "Churn"], autopct="%1.1f%%", startangle=90, colors=["#FFB6C1", "#99c5c4"])
    ax.axis("equal")
    st.pyplot(fig)

# Prediction
if st.sidebar.button("Predict"):
    try:
        # Collect user input into a dictionary
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
        
        # Make predictions
        prediction = model.predict(processed_data)
        
        # Display the result in the sidebar
        if prediction[0] == 1:
            st.sidebar.markdown("❌ <b>The customer is likely to churn.</b>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("✅ <b>The customer is likely to stay.</b>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Churn vs Not Churn BarChart")
    plot_churn_vs_not_churn(data['Churn'].value_counts())

with col2:
    st.subheader("Churn vs Not Churn PieChart")
    plot_churn_pie_chart(data['Churn'].value_counts())
