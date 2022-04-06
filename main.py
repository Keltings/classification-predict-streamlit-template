import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

import re

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')
pd.set_option('max_colwidth', -1)
news=df[(df['sentiment']==2)]
neutral=df[(df['sentiment']==0)]
pro=df[(df['sentiment']==1)]
anti=df[(df['sentiment']==-1)]

df_p = df.head(20)

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
import re

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # remove all numbers with letters attached to them
    #alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    stop = stopwords.words('english')
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmatizer = WordNetLemmatizer()
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


df['message_clean'] = df['message'].apply(review_to_words)

#TEXT FEATURE EXTRACTION
x_value=df['message_clean']
y=df['sentiment']

#Train_Test_Split
X_train, X_test, y_train, y_test = train_test_split(x_value, y,stratify=y,test_size=0.2, random_state=0)

#Bag of Words
vectorizer = CountVectorizer()
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

#Naive Baise
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
from sklearn.metrics import accuracy_score
mnb = MultinomialNB()
mnb.fit(count_train, y_train)
pred = mnb.predict(count_test)
score_1_nb = metrics.accuracy_score(y_test, pred)

from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
passive = PassiveAggressiveClassifier()
passive.fit(count_train, y_train)
pred = passive.predict(count_test)
score_pac = metrics.accuracy_score(y_test, pred)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)
mnb_tf = MultinomialNB()
mnb_tf.fit(tfidf_train_2, y_train)
pred = mnb_tf.predict(tfidf_test_2)
score_tf = metrics.accuracy_score(y_test, pred)

#TFIDF Model
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train, y_train)
pred = pass_tf.predict(tfidf_test)
scor_tfidf = metrics.accuracy_score(y_test, pred)

#TFIDF: Bigrams
tfidf_vectorizer2 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
tfidf_train_2 = tfidf_vectorizer2.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer2.transform(X_test)
pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_2, y_train)
pred = pass_tf.predict(tfidf_test_2)
score_bg = metrics.accuracy_score(y_test, pred)

#TFIDF : Trigrams
tfidf_vectorizer3 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
tfidf_train_3 = tfidf_vectorizer3.fit_transform(X_train)
tfidf_test_3 = tfidf_vectorizer3.transform(X_test)
pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_3, y_train)
pred = pass_tf.predict(tfidf_test_3)
score_tg = metrics.accuracy_score(y_test, pred)



#SECTIONS and Containers
header = st.container()

visual = st.container()

data_Eng = st.container()

model = st.container() 

opt = st.sidebar.selectbox('Select Section', ['Header', 'Visual', 'Model'] )

if opt == 'Header':
    with header:
        st.title('Climate Change Belief Analysis 2022')
        st.subheader('Taking a Breief Overview of our Data')
        st.write(df.head())

elif opt == 'Visual':  
    with visual:
        fig = go.Figure(data = go.Table(
            header = dict(values=list(df_p['sentiment', 'message', 'tweetid'].columns)),
            cells = dict(values=[df_p.sentiments, df_p.message, df_p.tweetid])
        ))
        st.write(df_p)
        sns.countplot(x = 'sentiment', data = df, palette="hls")
        plt.title("Distribution of sentiment");
        st.pyplot()

elif opt == 'Model':
    with model:
        options = ['Naive Bayes', 'ML Model TFIDF']
        model_choice = st.selectbox('Choose Model', options)
        if model_choice == 'Naive Bayes':
            col, cols = st.columns(2)
            col.subheader('Multimorminal_NB Accurac:')
            col.write(score_1_nb)

            col.subheader('Passive Agressive Classifer Accurac:')
            col.write(score_pac)

            col.subheader('Multimorminal_NB_2 Accurac:')
            col.write(score_tf)

        elif model_choice == 'ML Model TFIDF':
            col, cols = st.columns(2)
            col.subheader('Passive Agressive Classifer Accurac:')
            col.write(scor_tfidf)

            col.subheader('TFIDF Bigrams Accurac:')
            col.write(score_bg)

            col.subheader('TFIDF Trigrams Accurac:')
            col.write(score_tg)