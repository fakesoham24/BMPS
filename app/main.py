import streamlit as st
import pandas as pd
import joblib
import os
import io

# --- Page Configuration --- MUST be the first Streamlit command ---
st.set_page_config(page_title="Bank Term Deposit Prediction", page_icon="🏦", layout="wide")

# Load the trained model pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/best_model.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, '../models/optimal_threshold.pkl')

@st.cache_resource
def load_model_and_threshold():
    model = None
    threshold = 0.5
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(THRESHOLD_PATH):
        threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold

model, optimal_threshold = load_model_and_threshold()

# --- UI Styling ---
st.markdown("""
<style>
    .prediction-box-yes {
        padding: 20px;
        border-radius: 10px;
        background-color: #2e7d32;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
    .prediction-box-no {
        padding: 20px;
        border-radius: 10px;
        background-color: #c62828;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Bank Term Deposit Prediction System")
st.markdown("Predict whether a client will subscribe to a term deposit based on their profile and campaign history.")
st.divider()

if model is None:
    st.error("⚠️ Model not found. Please train the model first by running `python src/train.py`.")
else:
    # --- Input Section ---
    st.subheader("📝 Client Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
        default = st.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
        balance = st.number_input("Yearly Average Balance (in EUR)", value=1000)

    with col2:
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
        contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone', 'unknown'])
        day = st.number_input("Last Contact Day of the Month", min_value=1, max_value=31, value=15)
        month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        
    with col3:
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=200)
        campaign = st.number_input("Number of Contacts during this campaign", min_value=1, max_value=50, value=1)
        pdays = st.number_input("Days since last contact (from previous campaign)", min_value=-1, max_value=999, value=-1, help="-1 or 999 means client was not previously contacted")
        previous = st.number_input("Number of Contacts performed before this campaign", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Outcome of previous campaign", ['failure', 'other', 'nonexistent', 'success', 'unknown'])

    st.divider()
    
    # --- Prediction Logic ---
    if st.button("🚀 Predict Conversion", use_container_width=True):
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }])
        
        try:
            probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.0
            prediction = 1 if probability >= optimal_threshold else 0
            
            # Display Result
            if prediction == 1:
                st.markdown(f'<div class="prediction-box-yes">✅ Yes! The client is LIKELY to subscribe to a term deposit. <br><span style="font-size: 16px;">Probability: {probability:.2%}</span></div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="prediction-box-no">❌ No. The client is UNLIKELY to subscribe to a term deposit. <br><span style="font-size: 16px;">Probability: {probability:.2%}</span></div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
# ==============================
# 🔍 BULK PREDICTION FEATURE
# ==============================

st.divider()
st.header("🔍 Bulk Prediction Scanner")

# ------------------------------
# 1. SAMPLE FILE DOWNLOAD
# ------------------------------
st.subheader("1. Download Sample Templates")

sample_data = pd.DataFrame([{
    'age': 30,
    'job': 'management',
    'marital': 'single',
    'education': 'tertiary',
    'default': 'no',
    'balance': 1000,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'duration': 200,
    'campaign': 1,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown'
}])

col1, col2, col3 = st.columns(3)

# CSV
csv = sample_data.to_csv(index=False).encode('utf-8')
col1.download_button("📄 Download CSV Sample", csv, "sample.csv", "text/csv")

# Excel (use in-memory buffer — avoids writing to disk on every reload)
excel_buffer = io.BytesIO()
sample_data.to_excel(excel_buffer, index=False)
excel_buffer.seek(0)
col2.download_button("📊 Download Excel Sample", excel_buffer, "sample.xlsx")

# JSON
json = sample_data.to_json(orient="records", indent=2)
col3.download_button("📦 Download JSON Sample", json, "sample.json", "application/json")

# ------------------------------
# 2. FILE UPLOAD
# ------------------------------
st.subheader("2. Upload File to Scan")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "json"]
)

if uploaded_file is not None:

    try:
        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        else:
            st.error("Unsupported file format")
            df = None

        if df is not None:
            st.success("✅ File uploaded successfully!")

            # Preview
            st.write("### 📄 File Preview (Top 3 Rows)")
            st.dataframe(df.head(3), use_container_width=True)

            # ------------------------------
            # 3. BULK PREDICTION
            # ------------------------------
            if st.button("🚀 Start Bulk Prediction", use_container_width=True):

                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df)[:, 1]
                        preds = (probs >= optimal_threshold).astype(int)
                    else:
                        preds = model.predict(df)
                        probs = [0]*len(preds)

                    # Add prediction columns
                    df["Prediction"] = preds
                    df["Probability"] = probs

                    st.success("✅ Scanning Complete!")

                    # Show results
                    st.dataframe(df, use_container_width=True)

                    # ------------------------------
                    # 4. DOWNLOAD RESULT
                    # ------------------------------
                    st.subheader("3. Download Results")

                    result_csv = df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        "📥 Download Scanned File",
                        result_csv,
                        "bulk_predictions.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    except Exception as e:
        st.error(f"File Processing Error: {e}")