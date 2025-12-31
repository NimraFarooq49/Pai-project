import streamlit as st
import joblib
import string
import nltk
import os
from nltk.corpus import stopwords

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)

# ================================
# DOWNLOAD STOPWORDS
# ================================
@st.cache_data
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()

# ================================
# LOAD MODEL & VECTORIZER (SAFE)
# ================================
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)

    model_path = os.path.join(base_path, "spam_model.pkl")
    vectorizer_path = os.path.join(base_path, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("❌ Model files not found. Please upload spam_model.pkl and vectorizer.pkl")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_models()

# ================================
# TEXT CLEANING
# ================================
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ================================
# UI
# ================================
st.title("📧 Spam Email / SMS Detection")
st.write("Machine Learning based spam classifier")

user_input = st.text_area(
    "✍️ Enter your message:",
    height=150
)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
            st.write(f"Confidence: {prob[1]*100:.2f}%")
        else:
            st.success("✅ This message is NOT SPAM")
            st.write(f"Confidence: {prob[0]*100:.2f}%")

st.markdown("---")
st.caption("🔒 Messages are not stored | Built with Streamlit")
