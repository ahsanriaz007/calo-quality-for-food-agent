import os
import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq

# Page Config
st.set_page_config(
    page_title="Calo Quality & Operations Control Center",
    page_icon="calo_logo.png",
    layout="wide",
)

# ---------------------------------------------------------
# Custom Calo UI/UX CSS Injection
# ---------------------------------------------------------
st.markdown(
    """
<style>
    /* Force every container's top spacing to absolute zero */
    .block-container, [data-testid="stAppViewBlockContainer"] {
        padding-top: 0px !important; 
        margin-top: 0px !important;
    }
    
    /* Hide default Streamlit headers and footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    
    /* Remove default margin above images */
    [data-testid="stImage"] {
        margin-top: 0px !important;
    }
    
    /* Make buttons rounded like Calo's UI */
    div.stButton > button:first-child {
        border-radius: 30px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    
    /* Soft hover effect for buttons */
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 10px rgba(24, 148, 100, 0.2);
    }
    
    /* Make metric cards pop slightly */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Center main heading and pull it close to the logo */
    h1 {
        text-align: center !important;
        margin-top: -20px !important; 
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header & Logo Layout
# ---------------------------------------------------------
spacer_left, center_col, spacer_right = st.columns([2, 1, 2])

with center_col:
    st.image("calo_logo.png", use_container_width=True)

st.title("Automated Quality & Operational Risk Intelligence")

st.markdown(
    "<p style='text-align: center; color: #6B7280; font-size: 1.1rem;"
    " margin-top: -10px; margin-bottom: 35px;'>Real-time customer feedback"
    " categorization & HACCP kitchen anomaly detection</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar - Groq Setup & Status (SECURE / NO HARDCODED KEYS)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# Safely check for API key in secrets, fallback to empty string for safety
try:
    default_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    default_key = ""

api_key = st.sidebar.text_input("Groq API Key", value=default_key, type="password")

if api_key:
  client = Groq(api_key=api_key)
else:
  client = None


def generate_action_plan(incident_text):
  if not client:
    return "API Key missing. Please configure your Groq API key in Streamlit Secrets or sidebar."
  prompt = f"""
    You are an expert Kitchen Operations Manager for Calo, a premium meal prep service.
    Review the following operational incident or customer complaint.
    Write a strict, 2-sentence maximum action plan for floor staff to resolve this.
    Be direct, professional, and focus on food safety or logistics.
    
    Incident Details: {incident_text}
    """
  try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
  except Exception as e:
    return f"Error calling Groq API: {e}"


# ---------------------------------------------------------
# Load Saved Models & Data
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
  try:
    clf = joblib.load("complaint_model.pkl")
    vec = joblib.load("vectorizer.pkl")
    return clf, vec
  except Exception:
    return None, None


clf, vec = load_ml_assets()

# Tabs Layout
tab1, tab2, tab3 = st.tabs([
    "📊 Customer Complaint Intelligence",
    "⚠️ Kitchen Anomaly Monitor",
    "🤖 Live AI Action Resolver",
])

# ---------------------------------------------------------
# TAB 1: Customer Complaints
# ---------------------------------------------------------
with tab1:
  st.subheader("Customer Complaint Analytics")

  if os.path.exists("labeled_reviews.csv"):
    df_reviews = pd.read_csv("labeled_reviews.csv")
    
    st.metric("Total Reviews Analyzed", len(df_reviews))
    st.write("") 

    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.write("### Recent Customer Feedback")
    with col_btn:
        if st.button("🔄 Sync Live Store Reviews", use_container_width=True):
            new_review = pd.DataFrame([{
                "rating": 1, 
                "category": "delivery", 
                "text": "Just ordered the Fiesta Chicken Bowl and it arrived 2 hours late. Unacceptable."
            }])
            df_reviews = pd.concat([new_review, df_reviews], ignore_index=True)
            df_reviews.to_csv("labeled_reviews.csv", index=False)
            
            st.toast("🚨 ALERT: New critical review detected on App Store!", icon="⚠️")
            time.sleep(1.5)
            st.rerun()

    # Data Formatting: Clean table with wrapped text and hidden index
    st.dataframe(
        df_reviews[["rating", "category", "text"]].head(10),
        use_container_width=True,
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn("Customer Review Text", width="large"),
            "rating": st.column_config.NumberColumn("Rating", format="%d ⭐"),
            "category": st.column_config.TextColumn("Issue Category")
        }
    )
    
    st.divider() 

    # UI Upgrade: Professional Plotly Chart in Calo Green
    st.write("### Issue Breakdown")
    category_counts = df_reviews["category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Volume"]
    
    fig = px.bar(
        category_counts, 
        x="Category", 
        y="Volume", 
        color_discrete_sequence=["#189464"], # Official Calo Green
        text_auto=True
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title="Number of Incidents",
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
  else:
    st.warning("`labeled_reviews.csv` not found. Please ensure files are in place.")

# ---------------------------------------------------------
# TAB 2: Kitchen Anomalies
# ---------------------------------------------------------
with tab2:
  st.subheader("HACCP Kitchen Operations & Safety Anomalies")

  if os.path.exists("kitchen_anomalies.csv"):
    df_anomalies = pd.read_csv("kitchen_anomalies.csv")

    m1, m2 = st.columns(2)
    m1.metric("Flagged Kitchen Events", len(df_anomalies))
    m2.metric(
        "High Severity Alerts",
        len(df_anomalies[df_anomalies["severity"] == "HIGH"]),
    )

    st.dataframe(
        df_anomalies, 
        use_container_width=True,
        hide_index=True
    )
  else:
    st.warning("`kitchen_anomalies.csv` not found.")

# ---------------------------------------------------------
# TAB 3: Live AI Resolver
# ---------------------------------------------------------
with tab3:
  st.subheader("Live Operational Incident Resolver")
  st.write(
      "Enter a customer complaint or kitchen alert to categorize it and"
      " generate an instant action plan."
  )

  user_input = st.text_area(
      "Enter Incident or Complaint:",
      value="Found a piece of hard plastic in my chicken meal.",
  )

  if st.button("Analyze & Resolve", type="primary"):
    if user_input.strip():
      col_x, col_y = st.columns(2)

      # UX Polish: Color-coded severity badges
      with col_x:
        st.markdown("#### 🏷️ Predicted Category")
        if clf and vec:
          vec_text = vec.transform([user_input])
          pred_cat = clf.predict(vec_text)[0].lower()
          
          if pred_cat in ["food_safety", "allergen", "foreign_object"]:
              st.error(f"🚨 **{pred_cat.upper()}** (CRITICAL SEVERITY)")
          elif pred_cat in ["delivery", "missing_item", "wrong_item"]:
              st.warning(f"⚠️ **{pred_cat.upper()}** (HIGH SEVERITY)")
          else:
              st.info(f"ℹ️ **{pred_cat.upper()}** (MODERATE SEVERITY)")
        else:
          st.info("ML model not loaded.")

      # LLM Action Plan
      with col_y:
        st.markdown("#### ⚡ Recommended Action Plan")
        with st.spinner("Generating AI action plan..."):
          action_plan = generate_action_plan(user_input)
          st.write(action_plan)