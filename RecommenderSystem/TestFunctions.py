import math


def export(df, file_name):
    """Takes a dataframe and a file name and saves the dataframe to the output folder"""
    df.to_csv("Output/" + file_name + ".csv", encoding='utf-8', index=False)


def percentage_error(pred, actual):
    """Takes in the predicted value and the actual value and returns the percentage error."""
    p_error = 100 * abs(pred - actual) / actual
    return p_error


def root_mean_square_error(list_of_preds_actuals):
    """Tales a list of tuples (predicted_value, actual_value) and calculates root mean square error. Also works with a
    list of lists."""
    square_total = 0

    for i in range(len(list_of_preds_actuals)):
        square_total += (list_of_preds_actuals[i][0] - list_of_preds_actuals[i][1]) ** 2

    return math.sqrt(square_total / len(list_of_preds_actuals))


def p_error_from_sample(sample_df):
    """Calculates average percentage error from a sample"""
    sample_df.dropna(inplace=True)

    count_avg = 0

    for i, row in sample_df.iterrows():
        p_error = percentage_error(row["predicted_rating"], row["actual_rating"])
        count_avg += p_error

    return count_avg / len(sample_df.index)


def rmse_from_sample(sample_df):
    """Calculates the root mean square error from a sample"""
    sample_df.dropna(inplace=True)

    actual_rating_list = sample_df["actual_rating"].tolist()
    predicted_rating_list = sample_df["predicted_rating"].tolist()
    rating_tuples = tuple(zip(predicted_rating_list, actual_rating_list))

    return root_mean_square_error(rating_tuples)


def sparsify_df(df, percentage: float, seed: int = None):
    """Makes the input DataFrame more sparse by removing rows.

    Args:
        df (DataFrame): The DataFrame you want to increase the sparsity of.

        percentage (float): The percentage of rows remaining in the output.

        seed (int): The seed used in the random selection of the ratings. If no seed is provided then the selection will
            be unseeded.

    Returns:
        DataFrame: The input DataFrame with reduced number of rows.

    """
    num_rows = int(len(df.index) * percentage)

    if seed is None:
        return df.sample(num_rows)
    else:
        return df.sample(num_rows, random_state=seed)
