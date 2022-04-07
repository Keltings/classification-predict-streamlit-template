"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os





# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
#import plotly.graph_objects as go
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

import re

import warnings
warnings.filterwarnings('ignore')
model_pa = joblib.load(open('resources/Passiveagg_model.pkl', 'rb'))
model_lr =joblib.load(open('resources/logreg_model.pkl', 'rb'))
#model_pass = joblib.load(open('models\sentiment analysis pipe_pac.pkl', 'rb'))

#fnx
def predict_sentiment(docx):
    result = model_pa.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_pa.predict_proba([docx])
    return results

#fnx
def predict_sentiment(docx):
    result = model_lr.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_lr.predict_proba([docx])
    return results 


#sentiment_name_dict = {-1 : 'Anti', 0 : 'Neutral', 1 : 'Pro', 2 : 'News'}

# Vectorizer
news_vectorizer = open("resources/tf-idf.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Belief Analysis")
	st.subheader("Text Classification Analysis of Twitter Sentiments")
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Project Introduction","Team", "Exploratory Data Analysis" ,"Prediction"]
	selection = st.sidebar.selectbox("Main Menu", options)

	# Building out the "Information" page
	if selection == "Project Introduction":
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
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	if selection == "Team":
		st.subheader("ABOUT THE TEAM")
		from PIL import Image
		image = Image.open('resources/logo.jpg')

		st.image(image, caption='AI at it\'s Peak')


		st.info("Who we are")
		# You can read a markdown file from supporting resources folder
		st.markdown("We are a team of young data scientists, popular for the numerous statistical and analytical solutions we offer analytical and AI services to a wide range of corporate institutes and organizations. ")
		

		from PIL import Image
		image = Image.open('resources/Teampix.jpg')

		st.image(image, caption='Our faces')
	# Building out the Data  Exploratory page
	if selection == "Exploratory Data Analysis":
		st.info("Exploratory Analysis for our data ")
		st.markdown("This process helps us to understand patterns in our data, pinpoint any outliers and indicate relationships between variables. This phase includes descriptive statistics and data visualisations.")	
		from PIL import Image
		image = Image.open('resources/expl.jpg')

		st.image(image, caption='Exploratory Data Analysis')
		df = pd.read_csv('resources/train.csv')
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
		plt.title("Distribution of sentiment");
		st.pyplot()
		#st.write(chart1)
	
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with our best Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		model_type = st.sidebar.selectbox("Model Type",('Passive Aggressive', 'Logistic Regression'))


		
		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()


			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_type == 'Passive Aggressive':
				predictor = joblib.load(open(os.path.join("resources/Passiveagg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				

			if model_type == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/logreg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			else:
				print("Choose Model to classifiy Text") 	

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#sentiment_name_dict = {-1 : 'Anti', 0 : 'Neutral',1: 'Pro', 2 : 'News'}	
			sentiment_name_dict = {-1 : 'Anti  and it does not believe in man-made climate change ', 0 : 'Neutral and it neither supports nor refutes the belief of man-made climate change',1: 'Pro and it supports the belief of man-made climate change', 2 : 'News and it is helping to share somefactual news about climate change '}
			st.success("Prediction")
			prediction1 = tuple(prediction)
			sentiment_name = sentiment_name_dict[prediction1[0]]
			#st.write('{}:{}'.format(prediction1,sentiment_name))
			st.write('{}'.format(prediction1))
			st.success("Interpretation")
			st.write("This text is  categorized as: {}".format(sentiment_name))	
			
				
					

		
        


		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
