
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1f77b4; text-align: center; }
    .prediction-high { background: linear-gradient(135deg, #ff6b6b, #ee5a5a); color: white; padding: 2rem; border-radius: 15px; text-align: center; }
    .prediction-medium { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; padding: 2rem; border-radius: 15px; text-align: center; }
    .prediction-low { background: linear-gradient(135deg, #1dd1a1, #10ac84); color: white; padding: 2rem; border-radius: 15px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Load model (with caching)
@st.cache_resource
def load_artifacts():
    return {
        'model': joblib.load('churn_model.pkl'),
        'scaler': joblib.load('scaler.pkl'),
        'encoders': joblib.load('label_encoders.pkl'),
        'num_cols': joblib.load('numerical_cols.pkl'),
        'cat_cols': joblib.load('categorical_cols.pkl'),
        'features': joblib.load('feature_names.pkl')
    }

try:
    art = load_artifacts()
    model, scaler, encoders = art['model'], art['scaler'], art['encoders']
    num_cols, cat_cols, features = art['num_cols'], art['cat_cols'], art['features']
except:
    st.error("Model files not found. Please train and save models first.")
    st.stop()

# Header
st.markdown('<div class="main-header">🛡️ ChurnGuard AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Day 83: Interactive Churn Prediction App</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Customer Profile")

    with st.form("churn_form"):
        with st.expander("👤 Demographics", expanded=True):
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with st.expander("📞 Account Info", expanded=True):
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        with st.expander("📺 Services", expanded=True):
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
            backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
            protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            tech = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

        with st.expander("💰 Financial", expanded=True):
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 5.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure, 10.0)

        submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

with col2:
    st.subheader("📊 Results")

    if submitted:
        # Prepare data
        input_df = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [1 if senior == "Yes" else 0],
            'Partner': [partner], 'Dependents': [dependents],
            'tenure': [tenure], 'PhoneService': [phone],
            'MultipleLines': [lines], 'InternetService': [internet],
            'OnlineSecurity': [security], 'OnlineBackup': [backup],
            'DeviceProtection': [protection], 'TechSupport': [tech],
            'StreamingTV': [tv], 'StreamingMovies': [movies],
            'Contract': [contract], 'PaperlessBilling': [paperless],
            'PaymentMethod': [payment], 'MonthlyCharges': [monthly],
            'TotalCharges': [total]
        })

        # Preprocess
        proc = input_df.copy()
        for col in cat_cols:
            if col in proc.columns and col in encoders:
                try:
                    proc[col] = encoders[col].transform(proc[col].astype(str))
                except:
                    proc[col] = 0
        proc[num_cols] = scaler.transform(proc[num_cols])
        proc = proc[features]

        # Predict
        pred = model.predict(proc)[0]
        probs = model.predict_proba(proc)[0]
        churn_prob = probs[1] if len(probs) > 1 else probs[0]

        # Display results
        if churn_prob >= 0.7:
            risk_class, risk_text = "prediction-high", "HIGH RISK 🔴"
            color = "#ff6b6b"
        elif churn_prob >= 0.4:
            risk_class, risk_text = "prediction-medium", "MEDIUM RISK 🟡"
            color = "#feca57"
        else:
            risk_class, risk_text = "prediction-low", "LOW RISK 🟢"
            color = "#1dd1a1"

        st.markdown(f'<div class="{risk_class}"><h2>{risk_text}</h2><h1>{churn_prob:.1%}</h1><p>Churn Probability</p></div>', unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=churn_prob*100, number={'suffix': "%"},
            domain={'x': [0,1], 'y': [0,1]},
            gauge={'axis': {'range': [0,100]}, 'bar': {'color': color},
                   'steps': [{'range': [0,30], 'color': '#d4edda'},
                            {'range': [30,70], 'color': '#fff3cd'},
                            {'range': [70,100], 'color': '#f8d7da'}]}
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Churn" if pred == 1 else "Stay")
        c2.metric("Confidence", f"{max(probs):.1%}")
        c3.metric("Monthly Value", f"${monthly:.0f}")

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        if churn_prob >= 0.7:
            st.error("🚨 **CRITICAL**: Immediate retention intervention required!")
            st.markdown("• Offer 20-30% discount or loyalty rewards\n• Schedule executive call within 48h\n• Assign dedicated retention specialist")
        elif churn_prob >= 0.4:
            st.warning("⚠️ **MEDIUM RISK**: Include in proactive retention campaign")
            st.markdown("• Send satisfaction survey with incentive\n• Offer flexible payment options\n• Share product usage tips")
        else:
            st.success("✅ **LOW RISK**: Maintain excellent service quality")
            st.markdown("• Enroll in loyalty program\n• Send occasional updates\n• Celebrate milestones")
    else:
        st.info("👈 Fill the form and click Predict to see results")
        st.markdown("### 📝 Try This High-Risk Profile:")
        st.markdown("- Month-to-month contract, Fiber optic\n- Electronic check payment\n- Low tenure (5 months), No tech support\n- Monthly charges: $89")
