import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Transformer Health Index Analysis", layout="wide")

st.title("Transformer Health Index Analysis & Prediction")
st.markdown("""
This application analyzes transformer health data using the methodology described in the project instructions.
It converts the continuous **Health Index** into a classification problem (Normal vs. Fault) and uses AI models to predict the status.
""")

# Note on Methodology
with st.expander("Methodology & Instructions Reference"):
    st.markdown("""
    **Based on `instruction.md`:**
    *   **Data-Driven AI**: Using ML instead of just empirical rules.
    *   **Imbalance Handling**: Using **SMOTE** to generate synthetic fault records.
    *   **Feature Engineering**: Using DGA ratios (feature crossovers).
    *   **Models**: 
        *   **GBDT Check**: Gradient Boosting (Feature Interactions).
        *   **Hybrid/SVM**: Support Vector Machine (Decision Boundaries).
    *   **Evaluation**: Prioritizing **F1-Score** and **Recall**.
    """)

# 1. Load Data
@st.cache_data
def load_data():
    # Attempt to load the file from the likely location
    file_path = 'health_index/Health index1.csv'
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path.")
        return None

df_raw = load_data()

if df_raw is not None:
    # Sidebar: Configuration
    st.sidebar.header("Configuration")
    
    # Threshold Definition
    st.sidebar.subheader("Fault Definition")
    threshold = st.sidebar.slider(
        "Health Index Threshold (Prediction Cutoff)", 
        min_value=0.0, max_value=100.0, value=50.0, 
        help="Transformers with Health Index > Threshold are labeled as 'Fault'."
    )
    
    # Model Selection
    model_type = st.sidebar.selectbox("Select AI Model", ["Gradient Boosting (GBDT)", "Support Vector Machine (SVM)"])
    
    use_smote = st.sidebar.checkbox("Use SMOTE (Fix Data Imbalance)", value=True, help="Generates synthetic examples for the minority class.")

    # 2. Preprocessing & Feature Engineering
    df = df_raw.copy()
    
    # Feature Engineering: DGA Ratios (Instruction 3.1)
    # Ratios are often more indicative than raw values
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    df['Ratio_CH4_H2'] = df['Methane'] / (df['Hydrogen'] + epsilon)
    df['Ratio_C2H2_C2H4'] = df['Acethylene'] / (df['Ethylene'] + epsilon)
    df['Ratio_C2H4_C2H6'] = df['Ethylene'] / (df['Ethane'] + epsilon)
    # GBDT captures feature crossovers naturally, but explicit ratios help SVM
    
    # Define Target
    df['Status'] = (df['Health index'] > threshold).astype(int)
    # Map 0 -> Normal, 1 -> Fault for display
    status_map = {0: 'Normal', 1: 'Fault'}
    df['Status_Label'] = df['Status'].map(status_map)

    # 3. Data Split
    # Selected features: DGA gases + Oil properties + Derived Ratios
    # Removing 'Health index' (Target Source) and 'Life expectation' (Correlated Target) and 'Status' labels from X
    drop_cols = ['Health index', 'Life expectation', 'Status', 'Status_Label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['Status']

    st.subheader("1. Data Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transformers", len(df))
    with col2:
        fault_count = y.sum()
        st.metric("Fault Count (Status=1)", fault_count, f"{(fault_count/len(df))*100:.1f}%")
        
    fig_dist = px.histogram(df, x="Health index", nbins=50, title="Health Index Distribution")
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handling Imbalance (SMOTE) (Instruction 3.1)
    if use_smote and y_train.nunique() > 1:
        # Check if we have at least some samples of both classes to run SMOTE
        if y_train.value_counts().min() >= 2:
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            st.sidebar.success(f"SMOTE Applied: Train size went from {len(X_train)} to {len(X_train_bal)}")
        else:
            st.sidebar.warning("Not enough samples in minority class to apply SMOTE.")
            X_train_bal, y_train_bal = X_train, y_train
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # Scaling (Important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Training
    st.subheader(f"2. Model Training: {model_type}")
    
    if model_type == "Gradient Boosting (GBDT)":
        # Instruction 2.A: GBDT best for feature crossovers
        model = GradientBoostingClassifier(random_state=42)
    else:
        # Instruction 2.B: SVM part of hybrid approaches
        model = SVC(probability=True, random_state=42)
        
    model.fit(X_train_scaled, y_train_bal)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # 5. Evaluation (Instruction 3.3 Prioritize F1 and Recall)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("F1-Score", f"{f1:.2%}", help="Harmonic mean of precision and recall. Important for imbalanced data.")
    col3.metric("Recall", f"{rec:.2%}", help="Ability to catch actual faults. Critical for failure analysis.")
    
    # Visualizations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Normal', 'Fault'], y=['Normal', 'Fault'])
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_viz2:
         # Feature Importance (only for GBDT)
        if model_type == "Gradient Boosting (GBDT)":
            st.write("**Feature Importance**")
            feat_imp = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(10)
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance plot available for GBDT model.")

    # 6. Full Dataset Prediction Results
    st.subheader("3. Prediction Results")
    st.markdown("Below is the original dataset with an added **Predicted Status** column based on the trained model.")
    
    # Predict on entire dataset (create scaled version first)
    X_full_scaled = scaler.transform(X)
    df['Predicted_Status_Code'] = model.predict(X_full_scaled)
    df['Predicted_Status'] = df['Predicted_Status_Code'].map(status_map)
    df['Health_Index_Label'] = df['Status_Label'] # Rename for clarity
    
    # Highlight mismatches
    df['Prediction_Correct'] = df['Status'] == df['Predicted_Status_Code']
    
    # Reorder columns to put results upfront
    result_cols = ['Health index', 'Health_Index_Label', 'Predicted_Status', 'Prediction_Correct']
    df_display = df[result_cols + [c for c in df.columns if c not in result_cols]]
    
    st.dataframe(df_display.style.apply(lambda x: ['background-color: #ffcdd2' if v == False else '' for v in x], subset=['Prediction_Correct']), use_container_width=True)
    
    st.download_button(
        label="Download Predictions as CSV",
        data=df_display.to_csv(index=False).encode('utf-8'),
        file_name='health_index_predictions.csv',
        mime='text/csv',
    )
    
else:
    st.warning("Please ensure 'Health index1.csv' is in the 'health_index' folder.")
