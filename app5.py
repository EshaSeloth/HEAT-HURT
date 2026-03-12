import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mysql.connector

st.set_page_config(page_title="Heat Hurt", page_icon="💇", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Nunito:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: #0d0d1a !important;
    color: #e8e8f0 !important;
}
.stApp { background-color: #0d0d1a !important; }

.main-title {
    font-family: 'Abril Fatface', serif !important;
    font-size: 4.7rem !important;
    color: #5a60d6 !important;
    margin: 0 !important;
    line-height: 1 !important;
}
.sub-title {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38b2ac;
    margin: 10px 0 32px 0;
}
.section-label {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38b2ac;
    margin: 20px 0 10px 0;
}
.result-box {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 18px;
}
.result-score {
    font-family: 'Abril Fatface', serif;
    font-size: 5rem;
    line-height: 1;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #5a60d6, #7b5ea7) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 36px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'Nunito', sans-serif !important;
}
.footer {
    text-align: center;
    color: #8888aa;
    font-size: 0.78rem;
    padding: 28px 0 12px 0;
    border-top: 1px solid #2a2a4a;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = mysql.connector.connect(host="localhost", user="root", password="root", database="heathurt")
    users = pd.read_sql("SELECT * FROM users", conn)
    heat  = pd.read_sql("SELECT * FROM heat_usage", conn)
    hair  = pd.read_sql("SELECT * FROM hair_condition", conn)
    conn.close()
    df = users.merge(heat, on="user_id").merge(hair, on="user_id")
    df['hair_type']  = df['hair_type'].map({'straight':1,'wavy':2,'curly':3,'coily':4})
    df['tool_type']  = df['tool_type'].map({'dryer':1,'curler':2,'straightener':3})
    df['usage_freq'] = df['usage_freq'].map({'monthly':1,'weekly':2,'daily':3})
    df = df.drop(columns=['self_reported_condition','report_date'], errors='ignore')
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['damage_score','user_id','condition_id','usage_id'], errors='ignore')
    y = df['damage_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    return clf

df    = load_data()
model = train_model(df)

st.markdown('<div class="main-title">HEAT HURT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-Powered Hair Health Damage Risk Predictor &amp; Analyzer</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Your Hair Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age       = st.number_input("Age", 10, 100, 25)
    hair_type = st.selectbox("Hair Type", ["Straight","Wavy","Curly","Coily"])
    tool_type = st.selectbox("Heat Tool", ["Hair Dryer","Curler","Straightener"])
with col2:
    temperature = st.slider("Temperature (°C)", 100, 230, 180)
    duration    = st.slider("Session Duration (mins)", 5, 75, 30)
    usage_freq  = st.selectbox("How often do you use heat?", ["Monthly","Weekly","Daily"])

months = st.slider("Months using heat tools", 1, 60, 12)
st.markdown("<br>", unsafe_allow_html=True)

hair_map = {"Straight":1,"Wavy":2,"Curly":3,"Coily":4}
tool_map = {"Hair Dryer":1,"Curler":2,"Straightener":3}
freq_map = {"Monthly":1,"Weekly":2,"Daily":3}
score_meta = {
    0: ("#00c48c","No Risk",     "Your hair is healthy! Keep up the good habits."),
    1: ("#6fcf97","Very Mild",   "Very mild risk — just keep an eye on things."),
    2: ("#f2c94c","Mild",        "Mild risk — consider reducing heat frequency."),
    3: ("#f2994a","Moderate",    "Moderate risk — cut down on heat styling."),
    4: ("#eb5757","High Risk",   "High risk — significantly reduce heat usage."),
    5: ("#8e0000","Severe Risk", "Severe risk — stop heat styling immediately!"),
}

if st.button("Predict My Hair Damage Risk"):
    input_df = pd.DataFrame([{
        "age": age, "hair_type": hair_map[hair_type], "tool_type": tool_map[tool_type],
        "temperature_range": temperature, "duration_min": duration,
        "usage_freq": freq_map[usage_freq], "total_usage_months": months
    }])
    prediction        = model.predict(input_df)[0]
    color, label, msg = score_meta[prediction]
    st.markdown(f"""
    <div class="result-box" style="background:{color}18; border:2px solid {color};">
        <div class="result-score" style="color:{color};">{prediction}<span style="font-size:2rem;opacity:0.5;">/5</span></div>
        <div style="font-size:1.3rem;font-weight:700;color:{color};margin-top:6px;">{label}</div>
        <div style="font-size:0.92rem;color:#e8e8f0;margin-top:8px;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Personalised Tips</div>', unsafe_allow_html=True)
    if prediction >= 3:
        st.error("Your hair needs attention:")
        tips = ["Use heat protectant spray before every session","Drop temperature below 180°C","Take 2–3 heat-free days per week","Deep condition weekly"]
    elif prediction >= 1:
        st.warning("Small changes go a long way:")
        tips = ["Always apply a heat protectant","Keep sessions under 20 minutes","Air dry occasionally"]
    else:
        st.success("Your routine looks great!")
        tips = ["Continue using low heat settings","Regular conditioning maintains healthy hair"]
    for t in tips:
        st.write(f"→ {t}")

st.markdown("""
<div class="footer">
    Heat Hurt &nbsp;·&nbsp; ML Hair Damage Predictor &nbsp;·&nbsp; Built with Python, scikit-learn &amp; Streamlit<br><br>
    Made by <strong>Esha Seloth</strong>
</div>
""", unsafe_allow_html=True)
