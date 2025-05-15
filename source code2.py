import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Fraud Detection Analysis (No Visuals)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Sample")
    st.write(df.head())

    # Encode categorical columns
    label_cols = ['country', 'merchant_type']
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # Features and label
    X = df.drop(['transaction_id', 'is_fraud'], axis=1)
    y = df['is_fraud']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and show results
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)

    st.subheader("Classification Report")
    st.text(report)
