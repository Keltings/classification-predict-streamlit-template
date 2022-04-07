#core packages
import streamlit as st
import altair as alt
import seaborn as sns

#EDA PKGs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer


import re


#utils
import joblib

# Load your raw data
data = pd.read_csv("data/train.csv")

model_lr = joblib.load(open('models\sentiment analysis pipe_lr1.pkl', 'rb'))
model_mnb =joblib.load(open('models\sentiment analysis pipe_mnb.pkl', 'rb'))
#model_pass = joblib.load(open('models\sentiment analysis pipe_pac.pkl', 'rb'))

#fnx
def predict_sentiment(docx):
    result = model_lr.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_lr.predict_proba([docx])
    return results

#fnx
def predict_sentiment(docx):
    result = model_mnb.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_mnb.predict_proba([docx])
    return results 


sentiment_name_dict = {-1 : 'Anti', 0 : 'Neutral', 1 : 'Pro', 2 : 'News'}


def main():
    st.title('Sentiment classifier app')
    menu =["Project Introduction","Team", "Exploratory Data Analysis","Predictions"]
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Predictions':
        st.subheader('Sentiment Text Prediction')
        st.sidebar.subheader('Tweets')
	    
        model_type = st.sidebar.selectbox("Model Type",('Naive Bayes', 'Logistic Regression'))

        if model_type == 'Naive Bayes':
            with st.form(key='sentiment_clf_form'):
                raw_text = st.text_area("Type here")
                submit_text = st.form_submit_button(label='Submit')
        
            if submit_text:
                col1,col2 = st.columns(2)
            
            # apply function here
                prediction = predict_sentiment(raw_text)
                probability = get_predict_proba(raw_text)

                with col1:
                    st.success('Original text')
                    st.write(raw_text)

                    st.success("Prediction")
                    sentiment_name = sentiment_name_dict[prediction]

                    st.write('{}:{}'.format(prediction,sentiment_name))

                    #get the confidence of the prediction
                    st.write('Confidence: {}'.format(np.max(probability)))



                with col2:
                    st.success("Prediction Probability")
                    st.write(probability)
                    #convert the entire probability into a adataframe
                    proba_df = pd.DataFrame(probability, columns=model_mnb.classes_)
                    st.write(proba_df.T)

                    #modify to plot it right
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['sentiments', 'probability']

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiments', y ='probability', color = 'sentiments')
                    st.altair_chart(fig, use_container_width=True)

        else:
            with st.form(key='sentiment_clf_form'):
                raw_text = st.text_area("Type here")
                submit_text = st.form_submit_button(label='Submit')
        
            if submit_text:
                col1,col2 = st.columns(2)
            
            # apply function here
                prediction = predict_sentiment(raw_text)
                probability = get_predict_proba(raw_text)

                with col1:
                    st.success('Original text')
                    st.write(raw_text)

                    st.success("Prediction")
                    sentiment_name = sentiment_name_dict[prediction]

                    st.write('{}:{}'.format(prediction,sentiment_name))

                    #get the confidence of the prediction
                    st.write('Confidence: {}'.format(np.max(probability)))



                with col2:
                    st.success("Prediction Probability")
                    st.write(probability)
                    #convert the entire probability into a adataframe
                    proba_df = pd.DataFrame(probability, columns=model_lr.classes_)
                    st.write(proba_df.T)

                    #modify to plot it right
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['sentiments', 'probability']

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiments', y ='probability', color = 'sentiments')
                    st.altair_chart(fig, use_container_width=True)

    
    if choice == "Project Introduction":
        st.subheader("INTRODUCTION")

        from PIL import Image
        image = Image.open('resources/lpage.jpg')

        st.image(image, caption='This is Climate Change ')
        #image = open("resources/new1.jpg")
        #st.image(image, caption='Sunrise by the mountains')
            
        st.info("ABOUT THE PROJECT")
        # You can read a markdown file from supporting resources folder
        st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint")
        st.markdown("Tesla, a carbon-neutral company, wishes to gauge how their services may be received.")
        st.write( "They would like to determine how people perceive climate change and whether or not they believe it is a real threat.")
        st.write("Tesla contacted our team \"M'Click. They requested that we  build a Classification Model" )
        st.write("We have been tasked to determine how people perceive climate change; if they believe it is a real threat or not.")
        st.write("Data of various tweets and climate change perception of various users have been provided to be analyzed for this task")
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(data[['sentiment', 'message']]) # will write the df to the page


    if choice == "Team":
        st.subheader("ABOUT THE TEAM")
        
        from PIL import Image
        image1 = Image.open('resources/logo.jpg')
        
        st.image(image1, caption='AI at it\'s Peak')
        
        st.info("Who we are")
		# You can read a markdown file from supporting resources folder
        st.markdown("We are a team of young data scientists, popular for the numerous statistical and analytical solutions we offer analytical and AI services to a wide range of corporate institutes and organizations. ")
        from PIL import Image
        image2= Image.open('resources/Teampix.jpg')
        st.image(image2, caption='Our faces')
	    # Building out the Data  Exploratory page          


    if choice == "Exploratory Data Analysis":
        
        st.info("Exploratory Analysis for our data ")
        st.markdown("This process helps us to understand patterns in our data, pinpoint any outliers and indicate relationships between variables. This phase includes descriptive statistics and data visualisations.")	
        from PIL import Image
        image3 = Image.open('resources/expl.jpg')

        st.image(image3, caption='Exploratory Data Analysis')
        df = pd.read_csv('data/train.csv')
        pd.set_option('max_colwidth', -1)
        news=df[(df['sentiment']==2)]
        neutral=df[(df['sentiment']==0)]
        pro=df[(df['sentiment']==1)]
        anti=df[(df['sentiment']==-1)]

		
		#check the top rows in the data
        st.info("Viewing the first 20 entries in the data")
        df_1 = df.head(20)
        st.write(df_1)

		#Count the seperate sentiment groups
        st.info("Numerical counts of the different Sentiment Classes")
        st.write(df.sentiment.value_counts())


		# A sentiment bar graph plot 
        st.info("Graphical Representation of the Sentiment Classes ")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        sns.countplot(x = 'sentiment', data = df, palette="hls")
        plt.title("Distribution of sentiment")
        st.pyplot()
		#st.write(chart1)
	
	
            
        
    

if __name__ == '__main__':
    main()
