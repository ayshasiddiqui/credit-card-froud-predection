import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* ==========================
   MAIN BACKGROUND
========================== */

.stApp{
background:linear-gradient(135deg,#ffffff,#f8f5ff,#efe6ff);
}

.main{
background:transparent;
color:#4b0082;
}

/* ==========================
   SIDEBAR
========================== */

section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#6a0dad,#8a2be2,#b266ff);
}

section[data-testid="stSidebar"] *{
color:white !important;
}

/* ==========================
   TITLES
========================== */

.title{

font-size:50px;

font-weight:900;

color:#6a0dad;

text-align:center;

margin-bottom:5px;

}

.subtitle{

font-size:20px;

text-align:center;

color:#7b2cbf;

margin-bottom:25px;

}

/* ==========================
   METRIC CARDS
========================== */

.metric-card{

background:white;

padding:22px;

border-radius:18px;

box-shadow:0px 10px 25px rgba(106,13,173,.18);

border-left:6px solid #8a2be2;

transition:.3s;

}

.metric-card:hover{

transform:translateY(-6px);

box-shadow:0px 18px 35px rgba(106,13,173,.30);

}

/* ==========================
   BUTTONS
========================== */

.stButton>button{

background:linear-gradient(90deg,#7b2cbf,#c77dff);

color:white;

border:none;

border-radius:12px;

font-size:18px;

font-weight:bold;

height:48px;

width:100%;

transition:.3s;

}

.stButton>button:hover{

background:linear-gradient(90deg,#6a0dad,#9d4edd);

transform:scale(1.03);

}

/* ==========================
   DOWNLOAD BUTTON
========================== */

.stDownloadButton>button{

background:#8a2be2;

color:white;

border-radius:10px;

}

/* ==========================
   DATAFRAME
========================== */

[data-testid="stDataFrame"]{

background:white;

border-radius:15px;

padding:10px;

}

/* ==========================
   METRIC
========================== */

[data-testid="metric-container"]{

background:white;

padding:15px;

border-radius:15px;

box-shadow:0px 5px 20px rgba(106,13,173,.15);

}

/* ==========================
   HEADINGS
========================== */

h1,h2,h3,h4{

color:#6a0dad;

}

/* ==========================
   SUCCESS
========================== */

.stSuccess{

background:#f3e8ff;

color:#4b0082;

}

/* ==========================
   ERROR
========================== */

.stError{

background:#ffe5ec;

color:#c1121f;

}

/* ==========================
   INFO
========================== */

.stInfo{

background:#ede9fe;

color:#5a189a;

}

/* ==========================
   PLOTLY CHARTS
========================== */

.js-plotly-plot{

background:white;

padding:10px;

border-radius:20px;

box-shadow:0px 8px 20px rgba(106,13,173,.12);

}

/* ==========================
   SCROLLBAR
========================== */

::-webkit-scrollbar{

width:8px;

}

::-webkit-scrollbar-thumb{

background:#8a2be2;

border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💳 Fraud Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 EDA", "📈 Model Performance", "🔍 Prediction", "📋 Dataset", "ℹ️ About"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered fraud detection dashboard built with Streamlit</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h2>{len(df):,}</h2>
            <p>Total Transactions</p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        fraud = df[df["Class"] == 1].shape[0]
        st.markdown(f'''
        <div class="metric-card">
            <h2>{fraud:,}</h2>
            <p>Fraud Cases</p>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        legit = df[df["Class"] == 0].shape[0]
        st.markdown(f'''
        <div class="metric-card">
            <h2>{legit:,}</h2>
            <p>Legitimate Cases</p>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        fraud_rate = (fraud / len(df)) * 100
        st.markdown(f'''
        <div class="metric-card">
            <h2>{fraud_rate:.2f}%</h2>
            <p>Fraud Rate</p>
        </div>
        ''', unsafe_allow_html=True)

    st.write("")

    st.markdown("### 📌 Project Overview")
    st.write(
        """
        This dashboard analyzes credit card transactions and identifies potentially fraudulent activities.
        The project includes exploratory data analysis, model performance evaluation, and real-time prediction capabilities.
        """
    )

    # Class distribution chart
    class_counts = df["Class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Class"] = class_counts["Class"].map({0: "Legitimate", 1: "Fraud"})

    fig = px.pie(
        class_counts,
        names="Class",
        values="Count",
        hole=0.6,
        title="Transaction Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# EDA PAGE
# -----------------------------
elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    # Amount Distribution
    st.subheader("💰 Transaction Amount Distribution")
    fig = px.histogram(
        df,
        x="Amount",
        nbins=100,
        title="Distribution of Transaction Amounts"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time Distribution
    st.subheader("⏱ Transaction Time Distribution")
    fig = px.histogram(
        df,
        x="Time",
        nbins=100,
        title="Distribution of Transaction Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    # Fraud vs Legit Amount
    st.subheader("💳 Fraud vs Legit Transaction Amount")
    fig = px.box(
        df,
        x="Class",
        y="Amount",
        color="Class",
        title="Amount Distribution by Class"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    # Example metrics (replace with actual notebook metrics)
    accuracy = 99.92
    precision = 91.3
    recall = 84.7
    f1 = 87.9

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy}%")
    col2.metric("Precision", f"{precision}%")
    col3.metric("Recall", f"{recall}%")
    col4.metric("F1 Score", f"{f1}%")

    st.write("")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        text="Value",
        title="Performance Metrics"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Fraud Prediction")

    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    time = st.number_input("Transaction Time", min_value=0.0, value=10000.0)

    st.write("### PCA Features (V1 - V5 example)")

    v1 = st.slider("V1", -30.0, 30.0, 0.0)
    v2 = st.slider("V2", -30.0, 30.0, 0.0)
    v3 = st.slider("V3", -30.0, 30.0, 0.0)
    v4 = st.slider("V4", -30.0, 30.0, 0.0)
    v5 = st.slider("V5", -30.0, 30.0, 0.0)

    if st.button("🚀 Predict", use_container_width=True):

        # Simple demo logic
        risk_score = abs(v1) + abs(v2) + abs(v3) + abs(v4) + abs(v5)

        if risk_score > 25 or amount > 5000:
            st.error("🚨 Fraudulent Transaction Detected!")
            st.progress(90)
        else:
            st.success("✅ Legitimate Transaction")
            st.progress(20)

# -----------------------------
# DATASET PAGE
# -----------------------------
elif page == "📋 Dataset":

    st.title("📋 Dataset Explorer")

    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="creditcard.csv",
        mime="text/csv"
    )

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 🎯 Objective
    Detect fraudulent credit card transactions using machine learning and data analysis.

    ### 🛠 Technologies Used
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Seaborn
    - Scikit-learn

    ### 📊 Dataset
    - European credit card transactions dataset
    - Highly imbalanced fraud detection problem

    ### 🚀 Features
    - Interactive dashboard
    - EDA visualizations
    - Model performance analysis
    - Real-time prediction interface
    - Dataset explorer
    """)

    st.success("✨ Built with Streamlit ❤️ | Developed by Aysha Siddiqui")