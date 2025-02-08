import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Set Streamlit title
st.title("Logistic Regression Streamlit App")

# Upload dataset
uploaded_file = st.file_uploader("C:\Users\Telang\Desktop\excelr\Titanic_train.csv", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target variable
    target = st.selectbox("Select Target Variable (y)", df.columns)

    # Select features
    features = st.multiselect("Select Features (X)", df.columns)

    if st.button("Train Model"):
        if target and features:
            X = df[features]
            y = df[target]

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            st.write(f"### Model Accuracy: {acc:.2f}")

            # Confusion Matrix
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            ax.matshow(cm, cmap="coolwarm")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
        else:
            st.warning("Please select both target and feature variables.")

# Run the app with: streamlit run app.py
