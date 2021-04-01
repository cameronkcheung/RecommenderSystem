import pandas as pd
from RecommenderSystem import RecommenderSystem

# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


class CollaborativeRS(RecommenderSystem):
    """Attributes:
        movies (DataFrame): A DataFrame that must contain columns ["movieId","title","keywords"].
        ratings (DataFrame): A DataFrame that must contain columns ["userId","movieId","rating"].
        matrix (DataFrame): The user-movie DataFrame. Has movies as columns and users and rows.
    """

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        super().__init__(movies, ratings)
        self.matrix = self.make_matrix()

    def make_matrix(self):
        """Creates a user-movie matrix using the input movies and ratings"""

        joined_ratings = self.ratings.merge(self.movies, on='movieId', how='left')

        # finding average ratings for each movie
        average_ratings = pd.DataFrame(joined_ratings.groupby('title')['rating'].mean())

        # adding total ratings
        average_ratings['total_ratings'] = pd.DataFrame(joined_ratings.groupby('title')['rating'].count())

        # pivoting DataFrame
        movie_matrix = joined_ratings.pivot_table(index='userId', columns='title', values='rating')

        # ----- Normalisation -----
        num_movies = len(movie_matrix.columns)

        # Adding average score column (total score / (number of movies - number of NaN))
        movie_matrix["avg_score"] = movie_matrix.sum(axis=1) / (num_movies - movie_matrix.isna().sum(axis=1))

        # Subtracting avg_score from all values
        movie_matrix.iloc[:, 0:-1] = movie_matrix.iloc[:, 0:-1].sub(movie_matrix.avg_score, axis=0)

        # Filling all NaN with 0
        movie_matrix.fillna(0, inplace=True)

        # Removing avg_score column
        del movie_matrix["avg_score"]

        return movie_matrix

    def predict_rating(self, user_id: int, movie_id: int, k_num: int = "max") -> float:

        """Predicts the rating of the input user for the input movie. If the user has already rated the movie then
        this method will not consider that rating when predicting the rating. Overrides
        RecommenderSystem.predict_rating().

        Args:
            user_id (int): The ID of the user.

            movie_id (int): The ID of the movie.

            k_num (int): The number of top most similar / highest correlation movies to consider when predicting the
            rating. If k_num == "max" then it will consider all movies possible. If the input int is > the max
            number movies then it will default to the max.

         Returns:
            float: The predicted rating of the input movie by the input user.
        """

        # ------------------------------------------------------------------------------- Getting correlations

        # Get the input movie title
        movie_title = self.get_title_by_movie_id(movie_id)

        # filter the movie-user matrix to only movies the user has rated
        temp_matrix = self.matrix.loc[:, self.matrix.loc[user_id].ne(0)]

        # adding the input movie to temp_matrix to be able to perform correlations
        temp_matrix[movie_title] = self.matrix[movie_title]

        # finding correlation between input movie and every movie the user has rated
        total_corr = temp_matrix.corrwith(temp_matrix[movie_title])

        # ------------------------------------------------------------------------------- Cleaning DataFrame

        # renaming new row to "correlation"
        recommendation = pd.DataFrame(total_corr, columns=['correlation'])

        # Removing the input movie
        recommendation.drop([movie_title], inplace=True)

        # Sorting and getting a subset of the movies
        # recommendation = recommendation[recommendation["total_ratings"] > 0]

        # add movieId column to recc
        movie_ids = self.movies[["movieId", "title"]]
        recommendation = recommendation.merge(movie_ids, on="title", how="left")

        # ------------------------------------------------------------------------------- Calculating predicted rating

        # getting dataframe of all the users ratings
        user_ratings = self.get_ratings_by_user_id(user_id)

        # merging the users ratings and the correlation for each movie
        corrs = recommendation[["movieId", "correlation"]]

        user_ratings = user_ratings.merge(corrs, on='movieId', how="left")

        # remove the input movie
        user_ratings.drop(user_ratings[user_ratings['movieId'] == movie_id].index, inplace=True)

        # only consider movies with correlation of >= 0
        # user_ratings.drop(user_ratings[user_ratings['correlation'] < 0].index, inplace=True)

        # sorting by correlation and getting kNumber of top results
        user_ratings.sort_values("correlation", ascending=False, inplace=True)

        if k_num != "max":
            user_ratings = user_ratings.head(k_num)

        # calculating predicted score for the movie

        top_eq = 0
        bottom_eq = 0
        for i, row in user_ratings.iterrows():
            top_eq = top_eq + (row["rating"] * abs(row["correlation"]))
            bottom_eq = bottom_eq + abs(row["correlation"])

        if bottom_eq == 0:
            return None
        else:
            predict_rating = top_eq / bottom_eq
            return predict_rating
