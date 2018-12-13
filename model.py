import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from random import randint
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import PCA


# read movie and rating data, then merve them into one table
# movie = pd.read_csv("data/movie.csv")
# rating = pd.read_csv("data/rating.csv")
# movie_rating = pd.merge(movie, rating, on = "movieId")
# user_ratings = movie_rating.pivot(index = 'userId', columns ='movieId', values = 'rating')
# user_ratings.dropna(axis=0, how='all')
# user_ratings.to_csv("user_ratings.csv")


def metamovie():
    url = "https://raw.githubusercontent.com/onceuponapril/650final/master/movie_metadata.csv"
    movie = pd.read_csv(url, error_bad_lines=False)
    movie_1 = movie.copy()
    movie_titles = []
    for title in movie_1["movie_title"]:
        movie_titles.append(title.strip())

    title_list = []
    for i in range(10):
        id = randint(0, len(movie_titles) - 1)
        if movie_titles[id] not in title_list:
            title_list.append(movie_titles[id])

    return title_list


def content_recommendation(input):
    url = "https://raw.githubusercontent.com/onceuponapril/650final/master/movie_metadata.csv"
    movie = pd.read_csv(url, error_bad_lines=False)
    movie_1 = movie.copy()
    movie_1["plot_keywords"] = movie_1["plot_keywords"].fillna(" ")
    movie_1['director_name'] = movie_1['director_name'].fillna(" ")
    movie_1['actor_1_name'] = movie_1['actor_1_name'].fillna(" ")
    movie_1['actor_2_name'] = movie_1['actor_2_name'].fillna(" ")
    movie_1['actor_3_name'] = movie_1['actor_3_name'].fillna(" ")
    movie_1['content'] = movie_1.apply(lambda x: x['plot_keywords'] + x['director_name'] + x['actor_1_name'] + 
                        x['actor_2_name'] + x['actor_3_name'], 1)
    
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(movie_1["plot_keywords"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    movie_titles = []
    for title in movie_1["movie_title"]:
        movie_titles.append(title.strip())

    movie_1 = movie_1.reset_index(drop=True)
    titles = movie_1["movie_title"]
    indices = pd.Series(movie_1.index, index=movie_titles)

    idx = indices[input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    qualified = movie_1.iloc[movie_indices][
        ["movie_title", "genres", "imdb_score"]
    ].head(10)

    return qualified


def collabmv():
    movie = pd.read_csv("data/movie.csv")
    movie=movie[["movieId", "title"]]
    collabset = []
    for i in range(10):
        id = randint(0, len(movie) - 1)
        col = movie.iloc[id]
        collabset.append(col)
    return collabset


def cf_recommend(movie_id, user_rating):
    
    movie = pd.read_csv("data/movie.csv")
    rating = pd.read_csv("data/rating.csv")

    movie_rating = pd.merge(movie, rating, on="movieId")
    movie_rating = movie_rating[["movieId", "userId", "rating"]]
    id=movie_rating.userId.unique()
    split_threshold = (id.max()-id.min())/6
    demo_data=movie_rating[movie_rating.userId<=split_threshold]


    # user input
    # user_rating = list(map(float, user_rating))
    user_mv = pd.DataFrame(movie_id, columns=["movieId"]).astype(int)
    user_rating = pd.DataFrame(user_rating, columns=["rating"])
    user_rating["rating"].replace('', np.nan, inplace=True)
    user_rating =user_rating["rating"].astype(float).dropna()

    user_input = pd.concat([user_mv, user_rating], axis=1)
    userId = len(demo_data.userId.unique())+1
    user_input["userId"] = userId
    user_pred = pd.concat([demo_data, user_input], axis=0, ignore_index=True)
    pd.to_numeric(user_pred["rating"], errors="coerce")


    # user input into prediction
    ratings = user_pred.pivot(index="userId", columns="movieId", values="rating")
    ratings = ratings.dropna(axis=0, how="all")
    ratings = ratings.fillna(0)


    # # Normalization
    R = ratings.as_matrix()
    user_ratings_mean = np.mean(R, axis=1)
    ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)

    # # prediction
    U, sigma, Vt = svds(ratings_demeaned, k=10)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt
    ) + user_ratings_mean.reshape(-1, 1)
    all_preds = pd.DataFrame(all_user_predicted_ratings, columns=ratings.columns)

    # Get and sort the user's predictions
    sorted_user_predictions = all_preds.iloc[userId-1].sort_values(ascending=False)  # User ID starts at 1

    recommendations = movie[~movie["movieId"].isin(user_input ["movieId"])].merge(
            pd.DataFrame(sorted_user_predictions).reset_index(),
            how="left",
            left_on="movieId",
            right_on="movieId"
        )
    recommendations.columns=["movieId","title","genres", "Predictions"]
    recommendations = recommendations.sort_values("Predictions", ascending=False).iloc[:10, :-1]

    return recommendations