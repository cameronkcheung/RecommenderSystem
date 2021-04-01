from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from RecommenderSystem import RecommenderSystem


class ContentRS(RecommenderSystem):
    """Attributes:
        movies (DataFrame): A DataFrame that must contain columns ["movieId","title","keywords"].
        ratings (DataFrame): A DataFrame that must contain columns ["userId","movieId","rating"].
        matrix (csr_matrix): The movie-word matrix.
    """

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        super().__init__(movies, ratings)
        self.matrix = self.make_matrix()

    def make_matrix(self):
        """ Creates a TF_IDF movie-word matrix using the input movies"""

        # Create TF-IDF vectors and remove stop words
        tfidf = TfidfVectorizer(stop_words='english')

        temp_movies = self.movies

        # Replace NaN with an empty string
        temp_movies['overview'] = temp_movies['overview'].fillna('')

        # Return the transformed TF-IDF matrix
        return tfidf.fit_transform(temp_movies['overview'])

    def get_correlations(self, title: str) -> pd.DataFrame:
        """Returns a DataFrame of correlation between the input movie and all other movies using cosine similarity on
        the movie-word matrix.

        Args:
            title (string): The title of the movie.

         Returns:
            DataFrame: A DataFrame with columns "correlation" and "title".
        """

        # Construct a reverse map of indices and movie titles
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

        # Get the index of the movie that matches the title
        idx = indices[title]

        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(self.matrix, self.matrix)

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Getting the indices for the movies
        movie_indices = [i[0] for i in sim_scores]
        movie_index = self.movies['title'].iloc[movie_indices]

        # Creating the output DataFrame
        out_df = pd.DataFrame(sim_scores, columns=["tempId", "correlation"])

        # Adding the title to the DataFrame and removing the tempId
        out_df["title"] = out_df["tempId"].map(movie_index)
        out_df.drop("tempId", axis=1, inplace=True)

        return out_df

    def predict_rating(self, user_id: int, movie_id: int, k_num: int = "max") -> float:

        """Predicts the rating of the input user for the input movie. If the user has already rated the movie then
        this method will not consider that rating when predicting the rating. (Overrides
        RecommenderSystem.predict_rating())

        Args:
            user_id (int): The ID of the user.

            movie_id (int): The ID of the movie.

            k_num (int): The number of top most similar / highest correlation movies to consider when predicting the
            rating. If k_num == "max" then it will consider all movies possible. If the input int is > the max
            number movies then it will default to the max.

         Returns:
            float: The predicted rating of the input movie by the input user.
        """

        # ------------------------------------------------------------------------------- Preparing DataFrame

        # get movie title
        movie_title = self.get_title_by_movie_id(movie_id)

        # ------------------------------------------------------------------------------- Getting correlations

        # making the correlation DataFrame
        total_corr = self.get_correlations(movie_title)

        # set new index to title
        total_corr.set_index("title", inplace=True)

        # renaming new row to "correlation"
        recommendation = pd.DataFrame(total_corr, columns=['correlation'])

        # ------------------------------------------------------------------------------- Cleaning DataFrame

        # Removing the input movie
        recommendation.drop([movie_title], inplace=True)

        # Averaging the correlations
        recommendation = pd.DataFrame(recommendation.groupby('title')['correlation'].mean())
        recommendation.dropna(inplace=True)

        # add movieId column to recc
        movieIDList = self.movies[["movieId", "title"]]
        recommendation = recommendation.merge(movieIDList, on="title", how="left")

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

        # fix divide by 0 issue when the correlation between two movies is 0
        if bottom_eq == 0:
            return None
        else:
            predict_rating = top_eq / bottom_eq
            return predict_rating
