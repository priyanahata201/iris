import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import pandas as pd

# App Title
st.title("Diabetes Dataset Viewer")

# Upload CSV
uploaded_file = st.file_uploader("Upload your diabetes CSV file", type=["csv"])

# Read and display the data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded and read successfully!")
        st.subheader("Data Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
    # Replace zeros in certain columns with the median (excluding columns where 0 is valid)
    columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_fix:
    df[col] = df[col].replace(0, df[col].median())

    # Feature and target split
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file to proceed.")
