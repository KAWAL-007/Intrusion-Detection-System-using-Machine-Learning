import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.title("🔐 Intrusion Detection System using Machine Learning")

# Load dataset
data = pd.read_csv("nsl_kdd_dataset.csv")

st.subheader("📊 Dataset Overview")
st.write(f"Shape: {data.shape}")
st.dataframe(data.head())

# Encode categorical
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Features & Label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Binary classification
y = y.apply(lambda x: 0 if x == 0 else 1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {acc:.2f}")

st.markdown("---")

# 🎯 SIDEBAR (LEFT PANEL)
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

# 🎨 MAIN DISPLAY AREA
st.header(f"📈 {graph_option}")

# GRAPH DISPLAY

if graph_option == "Label Distribution":
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)
    st.write("This graph shows the distribution of normal and attack traffic. It helps identify class imbalance, which is important for ensuring the model does not become biased toward one class.")

elif graph_option == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write("The heatmap represents correlations between features. Strong correlations indicate relationships, while weaker ones show independence, helping in feature selection and improving model performance.")

elif graph_option == "Confusion Matrix":
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
    st.write("Confusion matrix compares predicted and actual values. It helps evaluate model accuracy and shows how well the system detects intrusions and normal traffic.")

elif graph_option == "Feature Importance":
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances)
    st.pyplot(fig)
    st.write("Feature importance graph shows which features contribute most to predictions. It helps optimize the model by focusing on key attributes.")

elif graph_option == "Histogram":
    fig, ax = plt.subplots()
    data.hist(ax=ax)
    st.pyplot(fig)
    st.write("Histograms display the distribution of feature values, helping identify patterns, skewness, and data spread useful for machine learning.")

elif graph_option == "Box Plot":
    fig, ax = plt.subplots()
    sns.boxplot(data=data.iloc[:, :5], ax=ax)
    st.pyplot(fig)
    st.write("Box plots highlight outliers and anomalies in data. These unusual values can indicate potential intrusions or abnormal behavior.")

elif graph_option == "Scatter Plot":
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,0], data.iloc[:,1])
    st.pyplot(fig)
    st.write("Scatter plot shows relationships between two features and helps visualize clustering between normal and attack data.")

elif graph_option == "Count Plot":
    fig, ax = plt.subplots()
    sns.countplot(x=data.iloc[:,0], ax=ax)
    st.pyplot(fig)
    st.write("Count plot shows frequency of categorical values, helping understand which values occur most frequently.")

elif graph_option == "Distribution Plot":
    fig, ax = plt.subplots()
    sns.kdeplot(data.iloc[:,0], ax=ax)
    st.pyplot(fig)
    st.write("Distribution plot shows probability density of a feature, helping understand how data is spread across values.")

st.markdown("---")
st.info("📌 This system uses Machine Learning to detect and analyze network intrusions effectively.")