
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Iris Flower Classification App")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose an iris dataset CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)

    st.subheader("Dataset Preview")
    st.write(df.head())

    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    st.subheader("Make a Prediction")
    sepal_length = st.number_input("Sepal Length", value=5.1)
    sepal_width = st.number_input("Sepal Width", value=3.5)
    petal_length = st.number_input("Petal Length", value=1.4)
    petal_width = st.number_input("Petal Width", value=0.2)

    if st.button("Predict"):
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=X.columns)
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Species: **{prediction}**")

else:
    st.info("Please upload a CSV file containing the Iris dataset.")
