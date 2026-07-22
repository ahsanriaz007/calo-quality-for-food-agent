# Calo Quality & Operations Control Center

An automated, end-to-end quality assurance and operational risk intelligence platform built for [Calo](https://calo.app). This system bridges unstructured customer feedback (App Store/Play Store) with kitchen-level HACCP monitoring, leveraging machine learning classification and large language models (Groq Llama 3) to generate instant, actionable resolutions for floor operations.

🔗 **Live Demo:** [View Live App on Streamlit Cloud](https://your-stream-lit-url-here.streamlit.app) *(Replace with your live link once deployed)*

---

## 🚀 Key Features

1. **Customer Complaint Intelligence (NLP Pipeline):**
   - Automatically ingests customer reviews and classifies them into key operational risk categories (*Food Safety, Delivery, Missing Items, Packaging, etc.*).
   - Features a **Live Store Sync Simulation** that mimics incoming real-time critical feedback, immediately notifying managers via toast alerts and updating analytics.

2. **HACCP Kitchen Operations & Safety Monitor:**
   - Tracks simulated kitchen sensor logs, automatically flagging cold-chain violations (storage temps outside 2–6°C) and cooking threshold discrepancies (HACCP cook temperatures < 63°C).
   - Categorizes alerts by severity (*High vs. Moderate*) to prioritize floor interventions.

3. **Live AI Operational Incident Resolver:**
   - Powered by **Groq (`llama-3.3-70b-versatile`)**, taking any incoming customer complaint or kitchen alert and generating a strict, 2-sentence maximum professional action plan for kitchen floor staff.
   - Includes real-time risk-based color coding (**Critical**, **High**, **Moderate** severity badges).

---

## 🛠️ Tech Stack & Architecture

- **Frontend & UI:** Python, Streamlit (Customized with professional SaaS CSS and Plotly data visualizations).
- **Machine Learning:** `scikit-learn` (Text vectorization and multi-class classification for incoming reviews).
- **Generative AI:** Groq API (`llama-3.3-70b-versatile`) for ultra-low-latency operational action plan generation.
- **Data Engineering:** `pandas`, `joblib` for model persistence and efficient data processing.

---

## 📂 Project Structure

```text
calo-quality-agent/
│
├── app.py                  # Main Streamlit dashboard application
├── complaint_model.pkl     # Trained ML classifier for customer text
├── vectorizer.pkl          # TF-IDF vectorizer for text processing
├── labeled_reviews.csv     # Historical labeled customer feedback dataset
├── kitchen_anomalies.csv   # Flagged HACCP kitchen sensor events
├── .streamlit/             # UI configurations and secure secrets management
│   └── secrets.toml        # Local API key storage (git-ignored)
└── README.md               # Project documentation