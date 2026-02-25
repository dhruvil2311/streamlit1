import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
 
st.title("Naive Bayes Classification App")
 
st.write("Upload your own CSV file or use Credit.csv")
 
# ------------------------------
# Dataset Selection
# ------------------------------
data_option = st.radio("Choose Data Source", ["Upload CSV", "Use Credit.csv"])
 
df = None
 
if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
 
else:
    try:
        df = pd.read_csv("Credit.csv")
    except:
        st.error("Credit.csv not found in directory.")
 
# ------------------------------
# If dataset loaded
# ------------------------------
if df is not None:
 
    # Remove common ID columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
 
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Dataset shape: {df.shape}")
 
    # ------------------------------
    # Target Selection
    # ------------------------------
    target_column = st.selectbox("Select Target Column", df.columns)
 
    if target_column:
 
        y = df[target_column]
 
        # Classification Detection
        if y.dtype == "object" or y.nunique() <= 15:
            st.success("Detected: Classification Problem")
        else:
            st.error("Target appears continuous. Naive Bayes requires classification.")
            st.stop()
 
        feature_columns = st.multiselect(
            "Select Feature Columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )
 
        if feature_columns:
 
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random State", value=42, step=1)
 
            if st.button("Train Model"):
 
                try:
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()
 
                    # Handle missing values
                    X = X.fillna(X.mode().iloc[0])
 
                    # Encode categorical features
                    for col in X.columns:
                        if X[col].dtype == "object":
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
 
                    # Encode target
                    if y.dtype == "object":
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
 
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=int(random_state)
                    )
 
                    model = GaussianNB()
                    model.fit(X_train, y_train)
 
                    y_pred = model.predict(X_test)
 
                    accuracy = accuracy_score(y_test, y_pred)
 
                    st.subheader("Results")
                    st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")
 
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
 
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
 
                    # Table version
                    cm_df = pd.DataFrame(
                        cm,
                        columns=[f"Pred {i}" for i in range(cm.shape[1])],
                        index=[f"Actual {i}" for i in range(cm.shape[0])]
                    )
                    st.dataframe(cm_df)
 
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))
 
                except Exception as e:
                    st.error(f"Model error: {e}")