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

        # Preview the data
        st.subheader("Data Preview:")
        st.dataframe(df.head())

        # Show null value summary
        st.subheader("Null Value Summary:")
        st.write(df.isnull().sum())

        # Show dataset shape
        st.write(f"Total rows: {df.shape[0]}")

        # Correlation heatmap
        st.subheader("Correlation Heatmap:")
        numeric_df = df.select_dtypes(include=np.number)
        correlation_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap="coolwarm")
        st.pyplot(fig)

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

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train XGBoost model
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)

        # Predict
        y_pred = xgb.predict(X_test)

        # Model performance
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Live prediction
        st.subheader("Make a Live Prediction")
        user_input = {}
        for col in X.columns:
            dtype = X[col].dtype
            if np.issubdtype(dtype, np.number):
                user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))
            else:
                user_input[col] = st.selectbox(f"{col}", options=list(df[col].unique()))

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)

            # Align with training columns
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            prediction = xgb.predict(input_df)[0]
            result = "Churned" if prediction == 1 else "Not Churned"
            st.success(f"Predicted Outcome: **{result}**")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
