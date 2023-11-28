# sleep-disorder-ml

![image](https://github.com/rpearce1/sleep-disorder-ml/assets/130908954/7c07e310-b296-416c-b503-b97d509ea2c7)

## Sleep Health and Lifestyle Data App

A Streamlit web application for tracking sleep health and lifestyle was created, incorporating machine learning analysis and Tableau visualizations.

## Overview

The Sleep Health and Lifestyle Data App allows users to:

- Create accounts and set up profiles.
- Collect basic information about users, such as age, gender, and any existing sleep-related conditions.
- Input daily sleep patterns, including bedtime, wake-up time, and interruptions during the night.
- Explore machine learning models' predictions on stress levels based on user data.
- Visualize sleep and health-related data using Tableau.

## Usage

## Setup

 ```bash

# 1. Clone the Repository

git clone https://github.com/yourusername/sleep-health-lifestyle-app.git

# 2. Navigate to the project directory:

cd sleep-health-lifestyle-app

    Install the required dependencies:

pip install -r requirements.txt

# 3. Run the App:

streamlit run app.py

 *Visit the provided local URL (usually http://localhost:8501) in your web browser to access the app.

## Features

#Data Overview:
Display basic information about the dataset.

Explore summary statistics and handle missing values.

#Machine Learning:

Train machine learning models to predict stress levels based on user data.

# User Profiles and Sleep Tracking:

Allow users to input and track sleep-related data.

Calculate sleep duration based on bedtime and wake-up time.

# Tableau Visualization:

Embed Tableau visualizations for comprehensive data exploration.

## Directory Structure
-app.py: Main Streamlit app script.
-requirements.txt: List of Python dependencies.
-sleep_health_and_lifestyle_dataset_cleaned.csv: Sample dataset.
-model.pkl: Trained machine learning model (created during app runtime).
-sleep_tracking_image.jpg: Image for the Sleep Tracking App section.

## Credits:
-Streamlit
-Tableau
-Scikit-Learn
-LazyPredict




this is a group project from UNC-VIRT-DATA-PT-06-2023-U-LOLC. 

authors: 

Robert Pearce
Gabrellea Norman
Spencer Bowman

The purpose of this project is to view a large dataset and help determine what factors of one's life can lead to potential sleep disorders, such as Sleep Apnea or Insomnia. (synthetic data for educational and application purposes)

Dataset: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

There are 374 rows. Columns include variables about gender, stress level, occupation, blood pressure (systolic and diastolic), physical activity, age, and BMI. Also tracked are sleep duration and sleep quality. 

