import streamlit as st
import pandas as pd
from io import StringIO
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu
import numpy as np


train_data = pd.read_csv("train.csv")


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.pexels.com/photos/8961886/pexels-photo-8961886.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

    
    # sidebar for navigation
with st.sidebar:
        
    selected = option_menu('Disaster Tweet Classification',['Tweet Classification',],
                            icons=['activity','heart','person'],
                            default_index=0)
    # Load the saved models

nlp_model = pickle.load(open('nlp_model.sav', 'rb'))


title = st.text_input('Enter your text')
st.write('The Text Entered is -->', title)


data = {
    'text': [title]}

df = pd.DataFrame(data)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(train_data['text'])

x1_test = tfidf_vectorizer.transform(df['text'])
y = nlp_model.predict(x1_test)
if y==0:
    info = "The text is not classified as disaster tweet"
elif y==1:
    info = "The text is classified as disaster tweet"

if st.button('Predict'):
    st.success(info)
    
    