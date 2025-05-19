import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Telecom Churn Prediction App")

# File Upload
uploaded_file = st.file_uploader("Upload your telecom churn CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded and read successfully!")

        # Feature engineering: Convert 'area code' to string
        if 'area code' in df.columns:
            df['area code'] = df['area code'].astype(str)

        # Drop non-numeric columns (or encode if preferred)
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Split data into features and target
        if 'churn' in df_encoded.columns:
            X = df_encoded.drop('churn', axis=1)
            y = df_encoded['churn']
        else:
            st.error("Target column 'churn' not found.")
            st.stop()

        # Save training column names
        feature_names = X.columns

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train XGBoost model
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)

        # Predict
        y_pred = xgb.predict(X_test)

        # Accuracy
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

        # Live prediction
        st.subheader("Make a Live Prediction")
        user_input = {}

        # Reconstruct original columns from df (not encoded)
        for col in df.columns:
            if col == 'churn':
                continue  # Skip target variable
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                user_input[col] = st.selectbox(f"{col}", options=df[col].unique())
            else:
                user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([user_input])
            input_df_encoded = pd.get_dummies(input_df)

            # Align with training features
            input_df_encoded = input_df_encoded.reindex(columns=feature_names, fill_value=0)

            # Optional: Warn if any columns were missing
            missing_cols = set(feature_names) - set(input_df_encoded.columns)
            if missing_cols:
                st.warning(f"Note: Missing columns in input were set to 0: {missing_cols}")

            prediction = xgb.predict(input_df_encoded)[0]
            result = "Churned" if prediction == 1 else "Not Churned"
            st.success(f"Predicted Outcome: **{result}**")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
