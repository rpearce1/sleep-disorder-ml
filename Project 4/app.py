# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer 
from lazypredict.Supervised import LazyRegressor
import pickle

# Load the dataset
path = 'c:\\Users\\Gabrellea\\OneDrive\\Documents\\UNC Chapel Hill\\Project 4\\Sleep_health_and_lifestyle_dataset_cleaned.csv'
df = pd.read_csv(path)

# Set page title and icon
st.set_page_config(page_title="Sleep Tracking App", page_icon=":sleeping_accommodation:")

# Sidebar Tabs
selected_tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Sleep Tracking App", "Tableau Visualization"])

# Define lr outside of the if conditions
lr = LinearRegression()

# Data Overview Section
if selected_tab == "Data Overview":
    # Display basic information about the dataset
    st.write("### Dataset Overview:")
    st.write(df.head(3))
    st.write(df.info())

    # Categorize columns into numeric and categorical
    categoric = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    numeric = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Daily Steps', 'BP Systolic', 'BP Diastolic']

    # Display summary statistics for numeric columns
    st.write("### Summary Statistics for Numeric Columns:")
    st.write(df[numeric].describe().round(2))

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[numeric] = pd.DataFrame(imputer.fit_transform(df[numeric]), columns=numeric)

    # Drop rows with any remaining missing values
    df = df.dropna()

    # Display missing values after imputation
    st.write("### Missing Values After Imputation:")
    st.write(df.isnull().sum().sum())

    # Explore value counts for categorical columns
    st.write("### Value Counts for Categorical Columns:")
    st.write(df['Gender'].value_counts())
    st.write(df['Occupation'].value_counts())
    st.write(df['BMI Category'].value_counts())

    # Modify 'BMI Category' values
    df.loc[df['BMI Category'] == 'Normal Weight', 'BMI Category'] = 'Normal'
    st.write("### Modified 'BMI Category' Values:")
    st.write(df['BMI Category'].value_counts())

    # Explore value counts for 'Sleep Disorder'
    st.write("### Value Counts for 'Sleep Disorder':")
    st.write(df['Sleep Disorder'].value_counts())

    # Mapping for Sleep Disorder Measure
    sleep_disorder_mapping = {'No Disorder': 1, 'Sleep Apnea': 2, 'Insomnia': 3}

    # Create 'Sleep Disorder Measure' column based on 'Sleep Disorder'
    df['Sleep Disorder Measure'] = df['Sleep Disorder'].map(sleep_disorder_mapping)

    # Display the updated DataFrame
    st.write("### Updated DataFrame:")
    st.write(df[['Sleep Disorder', 'Sleep Disorder Measure']].head(10))
    st.write(df['Sleep Disorder Measure'].value_counts())

    # Drop unnecessary columns
    X = df.drop(['Person ID'], axis=1)
    y = df[['Stress Level']]

    # Label encode categorical columns and scale numeric columns
    le = LabelEncoder()
    scaler = StandardScaler()

    for column in categoric:
        X[column] = le.fit_transform(X[column])

    for column in numeric:
        X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lazy Predict using regression models
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Display model performance metrics
    st.write("### Model Performance Metrics:")
    st.write(models.sort_values(by=['RMSE', 'Time Taken']))

    # Train a Linear Regression model
    lr.fit(X_train, y_train)

    # Save the trained Linear Regression model
    pickle.dump(lr, open('c:\\Users\\Gabrellea\\OneDrive\\Documents\\UNC Chapel Hill\\Project 4\\model.pkl', 'wb'))

    # Save the Streamlit app
    st.write("### Streamlit App Saved Successfully!")

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
    interruptions = st.slider("Number of Interruptions", 0, 10, 2)

    # Calculate amount of sleep
    sleep_duration = (datetime.combine(datetime.today(), wake_time) - datetime.combine(datetime.today(), bedtime)).seconds / 3600

    # Combine User Input and Sleep Tracking Data
    user_data = {
        'Name': name,
        'Age': age,
        'Gender': gender,
        'Sleep_Condition': sleep_condition,
        'Bedtime': bedtime,
        'Wake_Time': wake_time,
        'Interruptions': interruptions,
        'Sleep_Duration': sleep_duration
    }

    # Display combined data
    st.write("### User and Sleep Tracking Data:")
    st.write(user_data)

    # Save the Streamlit app
    st.write("### Streamlit App Saved Successfully!")

# Tableau Visualization Section
elif selected_tab == "Tableau Visualization":
    # Display Tableau visualization using HTML
    tableau_link = "https://public.tableau.com/views/SleepHealthandLifestyle_17010133805390/Story1?:language=en-US&:display_count=n&:origin=viz_share_link"
    st.write("### Tableau Visualization:")
    st.markdown(f'<iframe src="{tableau_link}" width="1000" height="600"></iframe>', unsafe_allow_html=True)
    
    
