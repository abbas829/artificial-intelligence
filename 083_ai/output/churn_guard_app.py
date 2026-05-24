
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 🎨 PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ChurnGuard AI | Customer Retention Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 🎨 CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .churn-risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    .churn-risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9f43);
        color: white;
    }
    .churn-risk-low {
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 🏠 SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=100)
    st.title("🛡️ ChurnGuard AI")
    st.markdown("---")
    st.markdown("**Day 83 of 369-day AI Learning Path**")
    st.markdown("Transforming ML models into business-ready applications.")
    st.markdown("---")

    # Navigation
    st.subheader("📍 Navigation")
    page = st.radio("", 
        ["🏠 Predict Churn", "📊 Batch Predictions", "🔍 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    show_shap = st.toggle("Show SHAP Explanations", value=True)
    show_recommendations = st.toggle("Show Business Recommendations", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)

    st.markdown("---")
    st.info("💡 **Tip:** Adjust the confidence threshold to control prediction sensitivity.")
    st.markdown("Built with ❤️ using Streamlit")

# =============================================================================
# 📦 LOAD MODEL ARTIFACTS (Cached)
# =============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load and cache all model artifacts for performance."""
    artifacts = {
        'model': joblib.load('churn_model.pkl'),
        'scaler': joblib.load('scaler.pkl'),
        'encoders': joblib.load('label_encoders.pkl'),
        'num_cols': joblib.load('numerical_cols.pkl'),
        'cat_cols': joblib.load('categorical_cols.pkl'),
        'features': joblib.load('feature_names.pkl')
    }
    return artifacts

# Load artifacts
try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    num_cols = artifacts['num_cols']
    cat_cols = artifacts['cat_cols']
    features = artifacts['features']
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("Please ensure model artifacts are in the app directory.")
    model_loaded = False
    st.stop()

# =============================================================================
# 🏠 PAGE 1: SINGLE CUSTOMER PREDICTION
# =============================================================================
if page == "🏠 Predict Churn":

    # Header
    st.markdown('<div class="main-header">🛡️ ChurnGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Customer Churn Risk & Get Actionable Retention Strategies</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    # =============================================================================
    # 📋 LEFT COLUMN: INPUT WIDGETS
    # =============================================================================
    with col1:
        st.subheader("📋 Customer Profile")
        st.markdown("Enter customer details below to predict churn risk.")

        with st.form("customer_form"):
            # Create input sections with expanders for better organization

            with st.expander("👤 Demographics", expanded=True):
                # These fields map to our dataset features
                # Adjust based on your actual dataset columns

                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
                partner = st.selectbox("Has Partner", ["No", "Yes"], key="partner")
                dependents = st.selectbox("Has Dependents", ["No", "Yes"], key="dependents")

            with st.expander("📞 Account Information", expanded=True):
                tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
                    key="payment"
                )

            with st.expander("📺 Services Subscribed", expanded=True):
                phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone")
                multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"], key="lines")
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
                online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"], key="security")
                online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"], key="backup")
                device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"], key="protection")
                tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"], key="techsupport")
                streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"], key="tv")
                streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"], key="movies")

            with st.expander("💰 Financial Details", expanded=True):
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0, key="monthly")
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=10.0, key="total")

            # Submit button
            submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

    # =============================================================================
    # 📊 RIGHT COLUMN: PREDICTIONS & RESULTS
    # =============================================================================
    with col2:
        st.subheader("📊 Prediction Results")

        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })

            # Preprocess input
            input_processed = input_data.copy()

            # Encode categorical variables
            for col in cat_cols:
                if col in input_processed.columns and col in encoders:
                    try:
                        input_processed[col] = encoders[col].transform(input_processed[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        input_processed[col] = 0

            # Scale numerical features
            input_processed[num_cols] = scaler.transform(input_processed[num_cols])

            # Ensure column order matches training
            input_processed = input_processed[features]

            # Make prediction
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            churn_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]

            # Determine risk level
            if churn_probability >= confidence_threshold:
                risk_level = "HIGH 🔴"
                risk_class = "churn-risk-high"
                risk_color = "#ff6b6b"
            elif churn_probability >= confidence_threshold * 0.7:
                risk_level = "MEDIUM 🟡"
                risk_class = "churn-risk-medium"
                risk_color = "#feca57"
            else:
                risk_level = "LOW 🟢"
                risk_class = "churn-risk-low"
                risk_color = "#1dd1a1"

            # Display prediction box
            st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h2 style="margin:0;">Churn Risk: {risk_level}</h2>
                    <h1 style="margin:0.5rem 0; font-size: 4rem;">{churn_probability:.1%}</h1>
                    <p style="margin:0; font-size: 1.1rem;">Probability of churning</p>
                </div>
            """, unsafe_allow_html=True)

            # Gauge chart for probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = churn_probability * 100,
                number = {'suffix': "%", 'font': {'size': 40}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#ccc",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
                st.markdown('</div>', unsafe_allow_html=True)
            with metric_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{max(probabilities):.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with metric_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Risk Score", f"{churn_probability:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # =============================================================================
            # 🔍 SHAP EXPLAINABILITY
            # =============================================================================
            if show_shap:
                st.markdown("---")
                st.subheader("🔍 Why This Prediction?")
                st.markdown("SHAP values explain which features pushed the prediction toward churn.")

                try:
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_processed)

                    # Handle binary classification SHAP values
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1][0]  # Churn class
                    else:
                        shap_vals = shap_values[0]

                    # Create feature importance DataFrame
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'SHAP_Value': shap_vals,
                        'Abs_SHAP': np.abs(shap_vals)
                    }).sort_values('Abs_SHAP', ascending=True)

                    # Plot SHAP bar chart
                    fig_shap = px.bar(
                        feature_importance.tail(10),  # Top 10 features
                        x='SHAP_Value',
                        y='Feature',
                        orientation='h',
                        color='SHAP_Value',
                        color_continuous_scale=['#1dd1a1', '#feca57', '#ff6b6b'],
                        title="Top 10 Features Driving This Prediction",
                        labels={'SHAP_Value': 'Impact on Churn Prediction', 'Feature': ''}
                    )
                    fig_shap.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # Show top drivers
                    top_drivers = feature_importance.tail(3)
                    st.markdown("**🔑 Key Drivers:**")
                    for _, row in top_drivers.iterrows():
                        direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
                        st.markdown(f"• **{row['Feature']}**: {direction} churn risk (impact: {row['SHAP_Value']:.3f})")

                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {str(e)}")

            # =============================================================================
            # 💡 BUSINESS RECOMMENDATIONS
            # =============================================================================
            if show_recommendations:
                st.markdown("---")
                st.subheader("💡 Business Recommendations")

                recommendations = []

                if churn_probability >= confidence_threshold:
                    recommendations.extend([
                        "🚨 **URGENT**: Assign dedicated retention specialist immediately",
                        "💰 Offer 20-30% discount or loyalty reward program",
                        "📞 Schedule executive outreach call within 48 hours",
                        "🎁 Provide premium service upgrade at no cost for 3 months"
                    ])
                elif churn_probability >= confidence_threshold * 0.7:
                    recommendations.extend([
                        "⚠️ **MEDIUM RISK**: Include in proactive retention campaign",
                        "📧 Send personalized satisfaction survey",
                        "💳 Offer flexible payment plan options",
                        "🤝 Invite to customer advisory board or feedback session"
                    ])
                else:
                    recommendations.extend([
                        "✅ **LOW RISK**: Maintain excellent service quality",
                        "🌟 Enroll in loyalty rewards program",
                        "📱 Send occasional product updates and new features",
                        "🎉 Celebrate customer milestones (anniversaries, usage achievements)"
                    ])

                # Add feature-specific recommendations
                if contract == "Month-to-month":
                    recommendations.append("📋 **Action**: Offer contract upgrade with incentives (lower monthly rate for longer commitment)")
                if tenure < 12:
                    recommendations.append("🆕 **Action**: New customer - ensure smooth onboarding and early success")
                if monthly_charges > 80:
                    recommendations.append("💸 **Action**: High-value customer - prioritize retention investment")

                for rec in recommendations:
                    st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)

                # Estimated revenue at risk
                annual_value = monthly_charges * 12
                revenue_at_risk = annual_value * churn_probability
                st.info(f"💵 **Estimated Annual Revenue at Risk:** ${revenue_at_risk:,.2f} (based on ${annual_value:,.2f} annual value × {churn_probability:.1%} churn probability)")

        else:
            # Show placeholder when no prediction yet
            st.info("👈 Fill out the customer profile form and click **Predict Churn Risk** to see results here!")

            # Show sample prediction
            st.markdown("### 📝 Sample Customer Profile")
            sample_data = {
                'Feature': ['Tenure', 'Monthly Charges', 'Contract', 'Internet Service', 'Payment Method'],
                'Value': ['12 months', '$65.00', 'Month-to-month', 'Fiber optic', 'Electronic check']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)
            st.caption("This profile typically shows ~75% churn risk. Try it out!")

# =============================================================================
# 📊 PAGE 2: BATCH PREDICTIONS
# =============================================================================
elif page == "📊 Batch Predictions":
    st.markdown('<div class="main-header">📊 Batch Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file to predict churn for multiple customers at once</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📁 Upload Customer CSV", type=['csv'])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(batch_df):,} customer records")

        with st.expander("🔍 Preview Data"):
            st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Analyzing customers..."):
                # Preprocess batch data
                batch_processed = batch_df.copy()

                for col in cat_cols:
                    if col in batch_processed.columns and col in encoders:
                        # Handle unseen categories
                        batch_processed[col] = batch_processed[col].astype(str).apply(
                            lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
                        )
                        batch_processed[col] = encoders[col].transform(batch_processed[col])

                batch_processed[num_cols] = scaler.transform(batch_processed[num_cols])
                batch_processed = batch_processed[features]

                # Predict
                batch_predictions = model.predict(batch_processed)
                batch_probabilities = model.predict_proba(batch_processed)[:, 1]

                # Add predictions to original dataframe
                results_df = batch_df.copy()
                results_df['Churn_Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in batch_predictions]
                results_df['Churn_Probability'] = batch_probabilities
                results_df['Risk_Level'] = results_df['Churn_Probability'].apply(
                    lambda x: 'HIGH 🔴' if x >= confidence_threshold else ('MEDIUM 🟡' if x >= confidence_threshold * 0.7 else 'LOW 🟢')
                )

                # Display results
                st.subheader("📋 Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Customers", len(results_df))
                col2.metric("High Risk", len(results_df[results_df['Risk_Level'] == 'HIGH 🔴']))
                col3.metric("Medium Risk", len(results_df[results_df['Risk_Level'] == 'MEDIUM 🟡']))
                col4.metric("Low Risk", len(results_df[results_df['Risk_Level'] == 'LOW 🟢']))

                # Distribution chart
                fig_dist = px.pie(
                    results_df, 
                    names='Risk_Level', 
                    title='Churn Risk Distribution',
                    color='Risk_Level',
                    color_discrete_map={'HIGH 🔴': '#ff6b6b', 'MEDIUM 🟡': '#feca57', 'LOW 🟢': '#1dd1a1'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    else:
        st.info("📤 Upload a CSV file with customer data to get started.")
        st.markdown("**Expected columns:** " + ", ".join(features[:10]) + "...")

# =============================================================================
# 🔍 PAGE 3: MODEL INSIGHTS
# =============================================================================
elif page == "🔍 Model Insights":
    st.markdown('<div class="main-header">🔍 Model Performance & Insights</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌟 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig_imp = px.bar(
            importance_df.head(15),
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title="Top 15 Most Important Features"
        )
        fig_imp.update_layout(height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.subheader("📊 Model Information")
        st.markdown(f"""
        **Model Type:** {type(model).__name__}

        **Parameters:**
        - Estimators: {model.n_estimators}
        - Max Depth: {model.max_depth}
        - Min Samples Split: {model.min_samples_split}
        - Class Weight: {model.class_weight}

        **Dataset:**
        - Features: {len(features)}
        - Numerical: {len(num_cols)}
        - Categorical: {len(cat_cols)}
        """)

        st.subheader("⚡ Quick Stats")
        st.markdown(f"""
        - **Features Used:** {len(features)}
        - **Training Samples:** ~{len(X_train):,}
        - **Model Size:** Random Forest with {model.n_estimators} trees
        """)

# =============================================================================
# ℹ️ PAGE 4: ABOUT
# =============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About ChurnGuard AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### 🎯 Mission
    ChurnGuard AI helps telecom companies predict and prevent customer churn using 
    advanced machine learning techniques, making AI accessible to business users.

    ### 🛠️ Built With
    - **Streamlit** - Web app framework
    - **Scikit-learn** - Machine learning models
    - **SHAP** - Model explainability
    - **Plotly** - Interactive visualizations
    - **Pandas & NumPy** - Data processing

    ### 📚 Learning Path
    This app is part of the **369-day Python & AI Learning Path**:
    - Day 81: Data Exploration & Feature Engineering
    - Day 82: Model Training & Evaluation
    - **Day 83: Building Interactive Web App** ⭐ You are here!
    - Day 84: Advanced Deployment & Monitoring

    ### 👨‍💻 Author
    Created as part of a comprehensive AI & ML learning journey.
    """)

    st.info("💡 **Pro Tip:** This app demonstrates how to productionize ML models for business stakeholders. The key is making complex predictions interpretable and actionable!")
