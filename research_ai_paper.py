import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🔐 Intrusion Detection System")

# Load dataset
data = pd.read_csv("nsl_kdd_dataset.csv")

st.subheader("📊 Dataset Overview")
st.write(f"Shape: {data.shape}")
st.dataframe(data.head())

# 🔥 STEP 1: Separate features & label FIRST (IMPORTANT)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 🔥 STEP 2: Proper binary conversion (CRITICAL FIX)
y = y.apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

# 🔥 STEP 3: Encode ONLY feature columns
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# Train model (fixed randomness)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


st.markdown("---")

# 🎯 SIDEBAR
st.sidebar.title("📊 Graph Controls")

graph_option = st.sidebar.radio(
    "Select a Graph",
    [
        "Label Distribution",
        "Correlation Heatmap",
        "Confusion Matrix",
        "Feature Importance",
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Count Plot",
        "Distribution Plot"
    ]
)

# 🎨 MAIN DISPLAY
st.header(f"📈 {graph_option}")

# GRAPH DISPLAY

if graph_option == "Label Distribution":
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    ax.set_xticklabels(["Normal", "Attack"])
    st.pyplot(fig)
    st.write("This graph shows the distribution of normal and attack traffic, helping identify class imbalance in the dataset.")

elif graph_option == "Correlation Heatmap":
    numeric_data = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write("The heatmap represents correlations between features, helping identify relationships and important variables.")

elif graph_option == "Confusion Matrix":
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack'],
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    st.write("Confusion matrix evaluates model performance by comparing actual and predicted classifications.")

elif graph_option == "Feature Importance":
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    st.write("This graph shows which features contribute most to intrusion detection.")

elif graph_option == "Histogram":
    fig = data.hist(figsize=(10,8))
    st.pyplot(plt.gcf())
    st.write("Histograms display feature distributions, helping understand data patterns.")

elif graph_option == "Box Plot":
    fig, ax = plt.subplots()
    sns.boxplot(data=X.iloc[:, :5], ax=ax)
    st.pyplot(fig)
    st.write("Box plots highlight outliers and anomalies in the dataset.")

elif graph_option == "Scatter Plot":
    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:,0], X.iloc[:,1])
    ax.set_title("Scatter Plot")
    st.pyplot(fig)
    st.write("Scatter plot shows relationships between two features.")

elif graph_option == "Count Plot":
    fig, ax = plt.subplots()
    sns.countplot(x=X.iloc[:,0], ax=ax)
    st.pyplot(fig)
    st.write("Count plot shows frequency of feature values.")

elif graph_option == "Distribution Plot":
    fig, ax = plt.subplots()
    sns.kdeplot(X.iloc[:,0], ax=ax)
    st.pyplot(fig)
    st.write("Distribution plot shows the density of feature values.")

st.markdown("---")
st.info("📌 This system uses Machine Learning to detect and analyze network intrusions effectively.")
