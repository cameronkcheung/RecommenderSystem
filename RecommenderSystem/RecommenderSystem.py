import pandas as pd
from abc import ABC, abstractmethod


class RecommenderSystem(ABC):

    @abstractmethod
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """Takes in a movies DataFrame and a ratings DataFrame"""
        self.movies = movies
        self.ratings = ratings

    def make_recommendation(self, user_id: int, k_num: int = "max") -> pd.DataFrame:
        """Makes a DataFrame of movie recommendations using the predict_rating() by the input RecommenderSystem for
        all movies in self.movies and returns a sorted list of recommendations as a DataFrame.

        Args:
            user_id (int): The ID of the user.

            k_num (int): The number of top most similar / highest correlation movies to consider when predicting the
                rating (see CollaborativeRS.predict_rating() and ContentRS.predict_rating()). If k_num == "max" then
                both RSs will consider all movies possible. If the input int is > the max number movies then it will
                default to the max.

         Returns:
            DataFrame: Contains the columns [movieId, title, predicted_rating] sorted descending by predicted_rating."""

        # getting a list of movie IDs of all movies the user has rated
        movie_df = self.get_ratings_by_user_id(user_id)
        watched_movie_ids = movie_df["movieId"].tolist()

        # make the return DataFrame
        column_names = ["movieId", "title", "predicted_rating"]
        return_df = pd.DataFrame(columns=column_names)

        # predict the rating for all unrated movies by the input user
        for i, row in self.movies.iterrows():
            if row["movieId"] not in watched_movie_ids:

                # create the new row and add to return DataFrame
                new_row = {"movieId": row["movieId"],
                           "title": row["title"],
                           "predicted_rating": self.predict_rating(user_id, row["movieId"], k_num)}
                return_df = return_df.append(new_row, ignore_index=True)
                # print(row["title"])

        # sort the return DataFrame descending by predicted_rating
        return_df.sort_values(by=['predicted_rating'], ascending=False, inplace=True)
        return return_df

    @abstractmethod
    def predict_rating(self):
        """Overrode in subclasses"""
        pass

    def get_movie_id_by_title(self, movie_title: str) -> int:
        """Takes in a movie title and returns its corresponding id"""
        movie_id = self.movies.loc[self.movies["title"] == movie_title]
        movie_id = movie_id.iloc[0]["movieId"]
        return movie_id

    def get_rating(self, movie_id: int, user_id: int) -> int:
        """Takes in a movie_id and a user_id and returns the corresponding rating"""
        rating = self.ratings.loc[(self.ratings.movieId == movie_id) & (self.ratings.userId == user_id)]
        rating = rating.iloc[0]["rating"]
        return rating

    def get_ratings_by_user_id(self, user_id: int) -> pd.DataFrame:
        """Gets a DataFrame of all ratings matching the input user id"""
        list_ratings = self.ratings.loc[self.ratings['userId'] == user_id]
        return list_ratings

    def get_ratings_by_movie_id(self, movie_id: int) -> pd.DataFrame:
        """Gets a dataframe of all ratings matching the input movie id"""
        list_ratings = self.ratings.loc[self.ratings['movieId'] == movie_id]
        return list_ratings

    def get_title_by_movie_id(self, movie_id: int) -> str:
        """Takes in a movie id and returns its corresponding title"""
        movie_title = self.movies.loc[self.movies["movieId"] == movie_id]
        movie_title = movie_title.iloc[0]["title"]
        return movie_title

# ------------------------------------------------------------------------------- Test functions

    def create_test_sample(self, k_num: int = "max", num_samples: int = 1, seed: int = None):
        """Creates a random test sample of ratings to be used for evaluation. Works by selecting a number of rating
        at random.

        Args:
            k_num (int): The number of top most similar / highest correlation movies to consider when predicting
                the rating (see CollaborativeRS.predict_rating() and ContentRS.predict_rating()). If k_num == "max" then
                both RSs will consider all movies possible. If the input int is > the max number movies then it will
                default to the max.

            num_samples (int): The number of sample ratings in the output. Defaults to 1.

            seed (int): The seed used in the random selection of the ratings. If no seed is provided then the
                selection will be unseeded.

         Returns:
            DataFrame: A DataFrame of randomly selected ratings containing the rows [userId, movieId, actual_rating,
            predicted_rating]. (unsorted)
        """

        # randomly select the ratings from self.ratings
        if seed is None:
            sample = self.ratings.sample(n=num_samples)
        else:
            sample = self.ratings.sample(n=num_samples, random_state=seed)

        # create output DataFrame
        output_df = pd.DataFrame(columns=["userId", "movieId", "actual_rating", "predicted_rating"])

        # predict the rating for the samples and append to the output DataFrame
        for i, row in sample.iterrows():
            predicted_rating = self.predict_rating(row["userId"], row["movieId"], k_num)

            new_row = {
                "userId": row["userId"],
                "movieId": row["movieId"],
                "actual_rating": row["rating"],
                "predicted_rating": predicted_rating
            }

            output_df = output_df.append(new_row, ignore_index=True)

        return output_df
