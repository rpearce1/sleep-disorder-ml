import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('Sleep_health_and_lifestyle_dataset_cleaned.csv')

# Set page title and icon
st.set_page_config(page_title="Sleep Tracking App", page_icon=":sleeping_accommodation:")

# Sidebar Tabs
selected_tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Sleep Tracking App"])

# Data Overview Section
if selected_tab == "Data Overview":
    # Display basic information about the dataset
    st.write("### Dataset Overview:")
    st.write(df.head())

     # Display an image
    image_path = 'https://github.com/rpearce1/sleep-disorder-ml/blob/main/Figures/decision_tree2.png?raw=true'  # Replace with the actual path to your image
    st.image(image_path, caption='Your Image Caption', use_column_width=True)

    # Display correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

    # Use a simple linear regression model for insights into sleep duration
    X = df[['Quality of Sleep']]
    y = df['Sleep Duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions using the model
    y_pred = model.predict(X_test)

    # Scatter plot of actual vs. predicted sleep duration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Sleep Duration')
    ax.scatter(X_test, y_pred, color='red', label='Predicted Sleep Duration')
    ax.set_title('Actual vs. Predicted Sleep Duration')
    ax.set_xlabel('Quality of Sleep')
    ax.set_ylabel('Sleep Duration')
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)
# Sleep Tracking App Section
elif selected_tab == "Sleep Tracking App":
    # User Profile Input Section
    st.sidebar.header("User Profile Setup")
    name = st.sidebar.text_input("Name")
    age = st.sidebar.number_input("Age", min_value=1, max_value=150)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    sleep_condition = st.sidebar.checkbox("Do you have any existing sleep-related conditions?")

    # Sleep Tracking Input Section
    st.subheader("Sleep Tracking")
    bedtime = st.time_input("Bedtime", value=datetime.now().replace(hour=22, minute=0))
    wake_time = st.time_input("Wake-up Time", value=datetime.now().replace(hour=6, minute=0))
    quality_of_sleep = st.slider("Quality of Sleep", 1, 10, 5)

    # Combine User Input and Sleep Tracking Data
    user_data = {
        'Name': name,
        'Age': age,
        'Gender': gender,
        'Sleep_Condition': sleep_condition,
        'Bedtime': bedtime,
        'Wake_Time': wake_time,
        'Quality_of_Sleep': quality_of_sleep
    }

    # Display combined data
    st.write("### User and Sleep Tracking Data:")
    st.write(user_data)

    # Use a simple linear regression model for insights into sleep duration
    X = df[['Quality of Sleep']]
    y = df['Sleep Duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions using the model
    predicted_sleep_duration = model.predict([[quality_of_sleep]])

    st.write("### Predicted Sleep Duration:")
    st.write(f"The predicted sleep duration is approximately {predicted_sleep_duration[0]:.2f} hours.")
