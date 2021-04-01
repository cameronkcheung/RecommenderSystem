from CollaborativeRS import CollaborativeRS
from ContentRS import ContentRS
from RecommenderSystem import RecommenderSystem
import pandas as pd


class HybridRS(RecommenderSystem):
    """Attributes:
        movies (DataFrame): A DataFrame of movies containing at least columns ["movieId","title","keywords"].
        ratings (DataFrame): A DataFrame of ratings containing at least columns ["userId","movieId","rating"].
        collaborative (CollaborativeRS): A collaborativeRS initialised using the input movies and ratings.
        content (ContentRS): A ContentRS initialised using the input movies and ratings.
        sparsity (float): The sparsity of the user-movie dataset of the CollaborativeRS.
    """

    def __init__(self, movies, ratings):
        super().__init__(movies, ratings)
        self.collaborative = CollaborativeRS(movies, ratings)
        self.content = ContentRS(movies, ratings)
        self.sparsity = self.calculate_sparsity()

    def calculate_sparsity(self):
        """Returning the sparsity value of the user-movie matrix using the equation by Sarwar et al. (2000):
        1 - (nonzero entries / total entries)"""
        num_of_zeros = (self.collaborative.matrix == 0).sum().sum()
        total_num_values = self.collaborative.matrix.shape[0] * self.collaborative.matrix.shape[1]

        sparsity = 1 - ((total_num_values - num_of_zeros) / total_num_values)
        return sparsity

    def predict_rating(self, user_id: int, movie_id: int, k_num="max", mode: float = None) -> float:
        """Predicts the rating of the input user for the input movie. If the user has already rated the movie then
        this method will not use that rating when predicting the rating. (Overrides predict_rating() from
        RecommenderSystem)

        Args:
            user_id (int): The ID of the user.

            movie_id (int): The ID of the movie.

            k_num (int): The number of top most similar / highest correlation movies to consider when predicting the
                rating (see CollaborativeRS.predict_rating() and ContentRS.predict_rating()). If k_num == "max" then
                both RSs will consider all movies possible. If the input int is > the max number movies then it will
                default to the max.

            mode (float): The method of hybridisation. Argument should be a float between 0 and 1. This arguments
                specifies the percentage of CollaborativeRS's predicted rating used, i.e., if this argument is 0.6, then
                the CollaborativeRS's predicted rating will be weighted to be 60% and the ContentRS's to be 40%. If mode
                is not provided, the sparsity value will be used to calculate the predicted value.

         Returns:
            float: The predicted rating of the input movie by the input user.
        """
        content_rating = self.content.predict_rating(user_id, movie_id, k_num)
        collaborative_rating = self.collaborative.predict_rating(user_id, movie_id, k_num)

        if content_rating is None or collaborative_rating is None:
            return None
        elif mode is None:
            return (collaborative_rating * (1 - self.sparsity)) + (content_rating * self.sparsity)
        else:
            return (collaborative_rating * mode) + (content_rating * (1 - mode))

    def make_recommendation(self, user_id: int, k_num: int = "max", mode: float = None) -> pd.DataFrame:
        """Makes a DataFrame of movie recommendations using the predict_rating() by the input RecommenderSystem for
        all movies in self.movies and returns a sorted list of recommendations as a DataFrame.

        Args:
            user_id (int): The ID of the user.

            k_num (int): The number of top most similar / highest correlation movies to consider when predicting the
                rating (see CollaborativeRS.predict_rating() and ContentRS.predict_rating()). If k_num == "max" then
                both RSs will consider all movies possible. If the input int is > the max number movies then it will
                default to the max.

            mode (float): The method of hybridisation used to calculate the predicted rating. Argument should be a float
                between 0 and 1. This arguments specifies the percentage of CollaborativeRS's predicted rating used.
                (See HybridRS.predict_rating()).

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
                           "predicted_rating": self.predict_rating(user_id, row["movieId"], k_num, mode)}
                return_df = return_df.append(new_row, ignore_index=True)
                #print(row["title"])

        # sort the return DataFrame descending by predicted_rating
        return_df.sort_values(by=['predicted_rating'], ascending=False, inplace=True)
        return return_df

    def create_test_sample(self, k_num: int = "max", mode: float = None, num_samples: int = 1, seed: int = None):
        """Creates a random test sample of ratings to be used for evaluation. Works by selecting a number of rating
        at random.

        Args:
            k_num (int): The number of top most similar / highest correlation movies to consider when predicting
                the rating (see CollaborativeRS.predict_rating() and ContentRS.predict_rating()). If k_num == "max" then
                both RSs will consider all movies possible. If the input int is > the max number movies then it will
                default to the max.

            mode (float): The method of hybridisation used to calculate the predicted rating. Argument should be a float
                between 0 and 1. This arguments specifies the percentage of CollaborativeRS's predicted rating used.
                (See HybridRS.predict_rating()).

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
            predicted_rating = self.predict_rating(row["userId"], row["movieId"], k_num, mode)

            new_row = {
                "userId": row["userId"],
                "movieId": row["movieId"],
                "actual_rating": row["rating"],
                "predicted_rating": predicted_rating
            }

            output_df = output_df.append(new_row, ignore_index=True)

        return output_df
