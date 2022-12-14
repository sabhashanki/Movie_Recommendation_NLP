import pandas as pd
import numpy as np
from rake_nltk import Rake
import string
from flask import Flask, request, render_template, jsonify
import requests

# NLP model 

# importing dataset
df = pd.read_csv('movies_list.csv')

# Assigning punctuation list
punc = string.punctuation

# Remove Punction from Plot feature
df.Plot = df.Plot.apply(lambda x : "".join([i for i in x if i not in punc]))


rake = Rake()
df['keywords'] = ''

# Extracting keywords from plot feature
for index, row in df.iterrows():
    rake.extract_keywords_from_text(row.Plot)
    df['keywords'][index] = list(rake.get_word_degrees().keys())

# Extracting and removing punctuation from genre, actors and director feature
df.Genre = df.Genre.apply(lambda x : x.split(','))
df.Actors = df.Actors.apply(lambda x : x.split(',')[:3])
df.Director = df.Director.apply(lambda x : x.split(','))

# Removing whitespaces
df.Genre = df.Genre.apply(lambda x : [i.lower().replace(' ','') for i in x])
df.Actors = df.Actors.apply(lambda x : [i.lower().replace(' ','') for i in x])
df.Director = df.Director.apply(lambda x : [i.lower().replace(' ','') for i in x])

df['BOW'] = ''
columns = ['Genre','Director','Actors','keywords']

# Joining genre, actors, keywords into single feature as BOW
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    df.BOW[index] = words

# Removing whitespace in BOW 
df.BOW = df.BOW.str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

# creating vectorization, cosine similaarity score for BOW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vector = CountVectorizer()
vector_matrix = vector.fit_transform(df.BOW)
cosine_sim = cosine_similarity(vector_matrix, vector_matrix)

title = pd.Series(df.Title)

# Function to create recommendation list 
def recommendation(name, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = title[title == name].index[0]
    score = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top5 = list(score.iloc[1:11].index)

    for i in top5:
        recommended_movies.append(list(df['Title'])[i])
    
    return recommended_movies

# Flask Application

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():

    movie_name = request.form['movie']
    recom_movies = list(recommendation(movie_name))

    return render_template('home.html', prediction_text = f'Recommended Movies are : \n {",".join([i for i in recom_movies])}')

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)
