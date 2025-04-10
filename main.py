
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from textblob import TextBlob

# Load config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Corrected login location: 'main'
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.title("SNOMED CT + ICD-10 Medical Toolkit")

    # Text input
    text = st.text_area("Enter your medical text:")
    
    # Spelling correction suggestion
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    if corrected_text != text:
        st.info(f"Did you mean: `{corrected_text}`?")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("d4data/biobert-nli")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biobert-nli")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    if text:
        results = nlp(text)
        st.subheader("Recognized Medical Terms")
        for r in results:
            st.markdown(f"- **{r['word']}** ({r['entity_group']})")
        
        # Load ICD-10 sample dataset
        df = pd.read_csv("data/icd10_sample.csv")
        st.subheader("ICD-10 Code Mapping (Sample)")
        for r in results:
            matches = df[df['Term'].str.contains(r['word'], case=False)]
            for _, row in matches.iterrows():
                st.markdown(f"- **{row['Term']}** → `{row['ICD10']}`")

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
