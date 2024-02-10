import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
import re

# Load the model from disk
loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    words = re.sub(r"[^a-zA-Z]", " ", text).split()
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in words])

def predict_sentiment(input_text):
    cleaned_text = clean_and_lemmatize(input_text)
    prediction = loaded_model.predict([cleaned_text])
    return prediction[0]

# Set up the Streamlit interface
st.title('Sentiment Analysis Model')
user_input = st.text_area("Enter Text", "Type Here...")
if st.button('Predict Sentiment'):
    result = predict_sentiment(user_input)
    if result == 1:
        st.success('The sentiment is positive!')
    else:
        st.error('The sentiment is negative or neutral.')

# To run the app, use the following command:
# streamlit run app.py
