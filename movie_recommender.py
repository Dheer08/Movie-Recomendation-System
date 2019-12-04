import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/results",methods=["POST"])
def recommendations():
	global movie_user_likes
	movie_user_likes = request.form.get("movie")
	movies = recommend()
	if not movies:
		return render_template("Error.html")
	else:
		return render_template("recommendation.html",movies = movies)
#==================================================================================================================================#
df = pd.read_csv('movie_dataset.csv')
def recommend():
	def get_title_from_index(index):
		try:
			return df[df.index == index]["title"].values[0]
		except:
			return 0

	def get_index_from_title(title):
		try:
			return df[df.title == title]["index"].values[0]
		except:
			return -1

	features = ['keywords','cast','genres','director']

	for feature in features:
		df[feature] = df[feature].fillna("")

	def combine_features(row):
		try:
			return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
		except:
			print(row)

	df["combined_features"] = df.apply(combine_features,axis=1)

	cv = CountVectorizer()
	count_matrix = cv.fit_transform(df["combined_features"])

	cosine_similarities = cosine_similarity(count_matrix)
 	
	movie_index = get_index_from_title(movie_user_likes)
	if movie_index == -1:
		return []

	else:
		similiar_movies = list(enumerate(cosine_similarities[movie_index]))

		sorted_similiar_movies = sorted(similiar_movies,key=lambda x:x[1],reverse=True)

		i=0
		movies = []
		for movie in sorted_similiar_movies:
			if i >10:
				break
			elif i==0:
				i=i+1
			else:
				movies.append(get_title_from_index(movie[0]))
				i=i+1

		return movies

"""for movie in sorted_similiar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
print(' Total No of movies :',i)"""

#print(movies)
#==============================================================================================================================#

if __name__ == '__main__':
	app.run(debug=True)
	
