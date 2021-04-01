from CollaborativeRS import CollaborativeRS
from ContentRS import ContentRS
from HybridRS import HybridRS
import TestFunctions as tf
import pandas as pd

# load movies and ratings into DataFrames
movies = pd.read_csv("Data/combinedMovies.csv")
ratings = pd.read_csv("Data/combinedRatings.csv")

# initialise a instance of a HybridRS
hybrid_rs = HybridRS(movies, ratings)

# have the HybridRS predict the rating of user 1 for movie 3034
predicted_rating = hybrid_rs.predict_rating(user_id=1, movie_id=3034, k_num=10, mode=0.5)
print(f"Predicted rating = {predicted_rating}")

# have the HybridRS make a recommendation for user 1
recommendations = hybrid_rs.make_recommendation(1, k_num=10, mode=0.5)

# print the top 10 recommendations
print(recommendations.head(10))

# you can also export the recommendations to a csv file in the Data folder using export()
tf.export(recommendations, "recommendations for user 1")

# create a test sample
test_sample = hybrid_rs.create_test_sample(k_num=10, mode=0.5, num_samples=100, seed=123)

# using the test sample, find the average percentage error and RMSE
percentage_error = tf.p_error_from_sample(test_sample)
print(f"Percentage error = {percentage_error}")

rmse = tf.rmse_from_sample(test_sample)
print(f"RMSE = {rmse}")