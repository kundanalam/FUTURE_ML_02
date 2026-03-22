import streamlit as st
import pickle
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords

cat_model = pickle.load(open("cat_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Customer Support Analytics Dashboard")


user_input = st.text_area("Enter Customer Issue")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter valid input")
    else:
        cleaned = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned])

        prediction = cat_model.predict(input_vec)[0]
        probs = cat_model.predict_proba(input_vec)[0]

        col1, col2 = st.columns(2)

    
        with col1:
            st.subheader("🎯 Prediction Result")
            st.success(prediction)

            st.subheader("📝 Input Summary")
            st.write(user_input)

        
        with col2:
            st.subheader("📊 Confidence Distribution")

            fig, ax = plt.subplots()
            ax.bar(range(len(probs)), probs)
            ax.set_title("Model Confidence")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Probability")

            st.pyplot(fig)