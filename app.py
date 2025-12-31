import streamlit as st
import joblib
import string
import nltk
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
# LOAD MODEL & VECTORIZER
# ================================
@st.cache_resource
def load_models():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ================================
# TEXT CLEANING FUNCTION
# ================================
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ================================
# UI
# ================================
st.markdown("""
# 📧 Spam Email / SMS Detection
### 🚀 Machine Learning Based Classifier
Detect whether a message is **Spam** or **Not Spam**
""")

st.divider()

user_input = st.text_area(
    "✍️ Enter Email or SMS text:",
    height=150,
    placeholder="Congratulations! You have won a free prize..."
)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
            st.progress(int(probability[1] * 100))
            st.write(f"**Spam Confidence:** {probability[1] * 100:.2f}%")
        else:
            st.success("✅ This message is **NOT SPAM**")
            st.progress(int(probability[0] * 100))
            st.write(f"**Safe Confidence:** {probability[0] * 100:.2f}%")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("""
**Model:** Logistic Regression  
**Vectorizer:** TF-IDF  
**Accuracy:** ~96%  
**Dataset:** SMS Spam Collection
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by You")

# ================================
# FOOTER
# ================================
st.markdown("""
---
🔒 *Your messages are not stored.*  
🤖 *Built with Streamlit & Machine Learning*
""")
