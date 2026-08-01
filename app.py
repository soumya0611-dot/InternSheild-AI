import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

# Title
st.title("📧 InternShield AI")
st.write("Detect whether an email is Spam or Safe.")

# Input box
email = st.text_area("Paste Email Content Here")

# Button
if st.button("Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        email_vector = tfidf.transform([email])

        prediction = model.predict(email_vector)

        if prediction[0] == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Ham email")