"""
SmartRent - AI Rent Predictor Web App
A Streamlit application for predicting apartment rent in Chennai
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="SmartRent - AI Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
# Cache the model loading (loads once, stays in memory)
@st.cache_resource
def load_models():
    """
    Load the trained model and locality encoder.
    
    Returns:
        tuple: (model, encoder)
    """
    try:
        # Load trained model
        with open('models/rent_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load locality encoder
        with open('models/locality_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        return model, encoder
    
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run 'python src/train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load models
model, locality_encoder = load_models()

# ================== HEADER ==================
st.title("🏠 SmartRent - AI Rent Predictor")
st.markdown("### Predict apartment rent in Chennai using Machine Learning")
st.markdown("---")

# Info banner
st.info("💡 **How it works:** Enter apartment details in the sidebar, then click 'Predict Rent' to get an AI-powered estimate!")

# ================== SIDEBAR - INPUT FEATURES ==================
st.sidebar.header("📋 Enter Apartment Details")
st.sidebar.markdown("Fill in all the fields below:")

# Square Feet Input
sq_ft = st.sidebar.number_input(
    "🏢 Square Feet",
    min_value=300,
    max_value=3000,
    value=800,
    step=50,
    help="Size of the apartment in square feet"
)

# BHK Input
bhk = st.sidebar.selectbox(
    "🛏️ BHK (Bedrooms)",
    options=[1, 2, 3, 4],
    index=1,  # Default to 2 BHK
    help="Number of bedrooms"
)

# Floor Input
floor = st.sidebar.slider(
    "🏗️ Floor Number",
    min_value=0,
    max_value=20,
    value=5,
    help="0 = Ground floor"
)

# Locality Input
locality = st.sidebar.selectbox(
    "📍 Locality",
    options=sorted(locality_encoder.classes_),
    index=0,
    help="Select the area in Chennai"
)

# Furnished Input
furnished_options = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Fully Furnished": 2
}
furnished_label = st.sidebar.selectbox(
    "🛋️ Furnishing Status",
    options=list(furnished_options.keys()),
    index=1,  # Default to Semi-Furnished
    help="Level of furnishing"
)
furnished = furnished_options[furnished_label]

# Parking Input
parking = st.sidebar.number_input(
    "🚗 Parking Spaces",
    min_value=0,
    max_value=5,
    value=1,
    step=1,
    help="Number of car parking spots"
)

# Age Input
age_years = st.sidebar.slider(
    "🏛️ Building Age (years)",
    min_value=0,
    max_value=30,
    value=5,
    help="How old is the building?"
)

# Spacing
st.sidebar.markdown("---")

# Predict Button
predict_button = st.sidebar.button("🔮 Predict Rent", type="primary")

# ================== MAIN AREA - PREDICTION ==================
if predict_button:
    # Validation
    if sq_ft < 300 or sq_ft > 3000:
        st.error("⚠️ Square feet must be between 300 and 3000!")
        st.stop()
    
    # Rest of prediction code...
    
    # Show loading spinner
    with st.spinner("🤖 AI is predicting..."):
        
        # Prepare input data
        # Encode locality
        locality_encoded = locality_encoder.transform([locality])[0]
        
        # Create input DataFrame (must match training features)
        input_data = pd.DataFrame({
            'sq_ft': [sq_ft],
            'bhk': [bhk],
            'floor': [floor],
            'locality_encoded': [locality_encoded],
            'furnished': [furnished],
            'parking': [parking],
            'age_years': [age_years]
        })
        
        # Make prediction
        predicted_rent = model.predict(input_data)[0]
        
        # Round to nearest 100
        predicted_rent = round(predicted_rent / 100) * 100
    
    # ================== DISPLAY RESULTS ==================
    
    # Main prediction result
    st.success(f"## 💰 Predicted Monthly Rent: ₹{predicted_rent:,.0f}")
    
    st.markdown("---")

    #Safer Move for Predict Button Function
    st.session_state.predicted_rent = predicted_rent

    if "predicted_rent" in st.session_state:
        rent = st.session_state.predicted_rent

    # ================== PRICE COMPARISON ==================
    st.markdown("---")
    st.subheader("📈 Price Comparison")

    comparison_data = {
    'Feature': ['Your Apartment', 'Budget Option', 'Luxury Option'],
    'Rent (₹)': [
        predicted_rent,
        predicted_rent * 0.7,
        predicted_rent * 1.3
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.bar_chart(comparison_df.set_index('Feature'))


    # Create results summary
    results_df = pd.DataFrame({'Feature': ['Square Feet', 'BHK', 'Floor', 'Locality', 'Furnished', 'Parking', 'Age', 'Predicted Rent'],
    'Value': [sq_ft, bhk, floor, locality, furnished_label, parking, age_years, f"₹{predicted_rent:,.0f}"]})

    # Download Button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Report",
    data=csv,
    file_name='rent_prediction.csv',
    mime='text/csv'
    )

    
    # ================== DETAILED BREAKDOWN ==================
    st.subheader("📊 Detailed Breakdown")
    
    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        per_sqft = predicted_rent / sq_ft
        st.metric(
            label="Per Sq Ft",
            value=f"₹{per_sqft:.0f}",
            delta=None
        )
    
    with col2:
        range_low = predicted_rent * 0.9
        range_high = predicted_rent * 1.1
        st.metric(
            label="Expected Range",
            value=f"₹{range_low:,.0f} - ₹{range_high:,.0f}",
            delta="±10%"
        )
    
    with col3:
        yearly_rent = predicted_rent * 12
        st.metric(
            label="Yearly Cost",
            value=f"₹{yearly_rent/100000:.2f}L",
            delta=None
        )
    
    st.markdown("---")
    
    # ================== FEATURE DETAILS ==================
    st.subheader("🔍 Your Apartment Details")
    
    # Create 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **Size:** {sq_ft} sq ft
        - **Bedrooms:** {bhk} BHK
        - **Floor:** {floor}
        - **Locality:** {locality}
        """)
    
    with col2:
        st.markdown(f"""
        - **Furnishing:** {furnished_label}
        - **Parking:** {parking} space(s)
        - **Building Age:** {age_years} years
        - **Condition:** {'New' if age_years < 5 else 'Moderate' if age_years < 15 else 'Old'}
        """)
    
    # ================== INSIGHTS ==================
    st.markdown("---")
    st.subheader("💡 AI Insights")
    
    # Generate insights based on features
    insights = []
    
    if sq_ft < 600:
        insights.append("🏠 **Compact apartment** - Perfect for singles or couples")
    elif sq_ft > 1200:
        insights.append("🏠 **Spacious apartment** - Great for families")
    
    if furnished == 2:
        insights.append("🛋️ **Fully furnished** - Move-in ready, premium pricing")
    elif furnished == 0:
        insights.append("🛋️ **Unfurnished** - Budget-friendly option")
    
    if floor > 10:
        insights.append("🏗️ **High floor** - Better views, more privacy")
    elif floor == 0:
        insights.append("🏗️ **Ground floor** - Easy access, no elevator needed")
    
    if age_years < 3:
        insights.append("✨ **Nearly new building** - Modern amenities")
    elif age_years > 20:
        insights.append("🏛️ **Established building** - Lower rent, proven structure")
    
    if parking >= 2:
        insights.append("🚗 **Multiple parking** - Great for families with multiple vehicles")
    
    for insight in insights:
        st.info(insight)

else:
    # Show placeholder when no prediction made yet
    st.info("👈 **Get started:** Fill in the apartment details in the sidebar and click 'Predict Rent'")
    
    # Show example
    st.markdown("---")
    st.subheader("📸 Example Prediction")
    st.markdown("""
    **Try these sample inputs:**
    - Square Feet: 1000
    - BHK: 2
    - Floor: 7
    - Locality: Adyar
    - Furnishing: Semi-Furnished
    - Parking: 1
    - Age: 5 years
    
    **Expected Result:** ~₹23,000/month
    """)

# ================== FOOTER ==================
st.markdown("---")
st.markdown("### ℹ️ About This App")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Model Information:**
    - Algorithm: Linear Regression
    - Training Data: 200 Chennai apartments
    - Features Used: 7 key factors
    - Accuracy: ~90% R² Score
    """)

with col2:
    st.warning("""
    **Disclaimer:**
    - Predictions are estimates based on historical data
    - Actual rent may vary based on market conditions
    - Use as a reference, not absolute truth
    - Negotiate based on specific property features
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Python, Scikit-learn, and Streamlit</p>
    <p>🎓 AI/ML Portfolio Project | 2026</p>
</div>
""", unsafe_allow_html=True)