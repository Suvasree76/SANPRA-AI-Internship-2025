import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, ratings_path, movies_path):
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        self.user_item_matrix = None
        self.item_sim_df = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepares dataframes for recommendation."""
        if "timestamp" in self.ratings_df.columns:
            self.ratings_df = self.ratings_df.drop(columns=["timestamp"])

    def _create_user_item_matrix(self):
        """Creates the user-item matrix."""
        self.user_item_matrix = self.ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

    def _compute_item_similarity(self):
        """Computes the item-item similarity matrix."""
        item_user_matrix = self.user_item_matrix.T
        item_user_filled = item_user_matrix.fillna(0)
        item_sim_matrix = cosine_similarity(item_user_filled)
        self.item_sim_df = pd.DataFrame(item_sim_matrix, index=item_user_filled.index, columns=item_user_filled.index)

    def predict_ratings_item_based(self, target_user_id):
        """Predicts ratings for a target user using item-based CF."""
        user_ratings = self.user_item_matrix.loc[target_user_id].fillna(0)
        all_movie_ids = self.user_item_matrix.columns

        numerator = user_ratings.values.dot(self.item_sim_df.values)
        
        rated_mask = (user_ratings.values > 0)
        denominator = rated_mask.dot(np.abs(self.item_sim_df.values))

        predicted_ratings = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        pred_series = pd.Series(predicted_ratings, index=all_movie_ids)

        already_rated = self.user_item_matrix.loc[target_user_id].dropna().index
        pred_series = pred_series.drop(labels=already_rated, errors="ignore")

        return pred_series.sort_values(ascending=False)

    def recommend_movies(self, target_user_id, n=5):
        """Recommends top N movies for a target user."""
        self._create_user_item_matrix()
        self._compute_item_similarity()
        
        preds = self.predict_ratings_item_based(target_user_id)
        top_n = preds.head(n)

        recs = pd.DataFrame({"movieId": top_n.index, "predicted_rating": top_n.values})
        recs = recs.merge(self.movies_df, on="movieId", how="left")
        return recs

    def get_user_input_and_recommend(self):
        """Gets user input and provides recommendations."""
        print("Welcome! 😀")
        movie_name = input("Enter a movie you'd like to rate (e.g., Inception (2010)): ")

        movie_row = self.movies_df[self.movies_df['title'].str.lower() == movie_name.lower()]

        if movie_row.empty:
            print("Sorry 😢, this movie is not in our database.")
            print("Available movies include:")
            print(self.movies_df['title'].head(20).tolist())
            return

        movie_id = movie_row['movieId'].values[0]
        
        rating = 0
        while rating < 1 or rating > 5:
            try:
                rating = float(input("How would you rate this movie? (1-5): "))
                if rating < 1 or rating > 5:
                    print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")


        new_user_id = self.ratings_df['userId'].max() + 1
        new_rating = pd.DataFrame({"userId": [new_user_id], "movieId": [movie_id], "rating": [rating]})
        self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)

        print(f"\n👌 You rated '{movie_name}' with {rating}. Thank you!")

        print("\n🔮 Based on your rating, we recommend:")
        recommendations = self.recommend_movies(new_user_id)
        print(recommendations[['title', 'predicted_rating']])

def main():
    """Main function to run the movie recommender."""
    recommender = MovieRecommender("mdb/ratings.csv", "mdb/movies.csv")
    recommender.get_user_input_and_recommend()

if __name__ == '__main__':
    main()
