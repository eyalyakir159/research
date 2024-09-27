import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from hurst import compute_Hc
import warnings

# Suppress all warnings
#warnings.filterwarnings("ignore")


# Variance Ratio Test function
def varianceratio(X, lag=2):
    """
    Variance Ratio Test to detect if a series follows a random walk.
    X : time series data
    lag : the lag period (default is 2)
    """
    X = np.asarray(X)
    n = len(X)
    mu = np.mean(X)

    # Calculate the denominator
    b = np.sum((X[1:] - X[:-1]) ** 2) / (n - 1)

    # Calculate the numerator
    t = int(lag)
    a = np.sum((X[t:] - X[:-t]) ** 2) / (n - t)

    # Variance ratio
    vr = a / (t * b)
    return vr


# Function to calculate ADF, KPSS, PP, Hurst Exponent, and Variance Ratio stationarity scores for each feature
def calculate_stationarity_scores(df):
    scores = {'Feature': [], 'ADF_Score': [], 'KPSS_Score': [], 'PP_Score': [], 'Hurst_Score': [], 'VR_Score': []}

    for col in df.columns:
        time_series = df[col].dropna()  # Handle missing values, if any

        # ADF Test
        adf_result = adfuller(time_series)
        adf_score = adf_result[0]  # ADF Statistic

        # KPSS Test
        kpss_result = kpss(time_series, regression='c', nlags='auto')
        kpss_score = kpss_result[0]  # KPSS Statistic

        # Phillips-Perron Test (using arch package)
        pp_result = PhillipsPerron(time_series).stat
        pp_score = pp_result  # PP Statistic

        # Hurst Exponent
        H, _, _ = compute_Hc(time_series, simplified=True)
        hurst_score = H  # Hurst exponent

        # Variance Ratio Test
        vr_score = varianceratio(time_series)

        # Store results in the dictionary
        scores['Feature'].append(col)
        scores['ADF_Score'].append(adf_score)
        scores['KPSS_Score'].append(kpss_score)
        scores['PP_Score'].append(pp_score)
        scores['Hurst_Score'].append(hurst_score)
        scores['VR_Score'].append(vr_score)

    # Convert the scores dictionary into a DataFrame
    return pd.DataFrame(scores)


# Function to calculate the general stationarity score for the whole dataset
def calculate_general_stationarity_score(stationarity_df):
    # Compute mean of ADF, KPSS, PP, Hurst, and Variance Ratio scores
    adf_mean_score = np.mean(stationarity_df['ADF_Score'])
    kpss_mean_score = np.mean(stationarity_df['KPSS_Score'])
    pp_mean_score = np.mean(stationarity_df['PP_Score'])
    hurst_mean_score = np.mean(stationarity_df['Hurst_Score'])
    vr_mean_score = np.mean(stationarity_df['VR_Score'])

    # Calculate a combined score (e.g., mean of all metrics)
    combined_score = (np.abs(adf_mean_score) + kpss_mean_score + np.abs(
        pp_mean_score) + hurst_mean_score + vr_mean_score) / 5

    return adf_mean_score, kpss_mean_score, pp_mean_score, hurst_mean_score, vr_mean_score, combined_score


# List of file paths
paths = [
    "../data/ETT/ETTh1.csv",
    "../data/ETT/ETTm1.csv",
    "../data/weather/weather.csv",
    "../data/exchange_rate/exchange_rate.csv",
    "../data/traffic/traffic.csv",
    "../data/electricity/electricity.csv",
]

# Open the file in write mode (this will overwrite existing content)
with open('stationarity_results.txt', 'w') as file:
    for path in paths:
        # Load the dataset
        df_raw = pd.read_csv(path)

        # Remove 'date' column if it exists and keep the rest as features
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(columns=['date'])

        # Convert the dataset into the appropriate format
        df = df_raw  # Assuming all other columns are features

        # Calculate stationarity scores for each feature
        stationarity_scores = calculate_stationarity_scores(df)

        # Now calculate the general stationarity score for the whole dataset
        adf_mean, kpss_mean, pp_mean, hurst_mean, vr_mean, general_score = calculate_general_stationarity_score(
            stationarity_scores)

        # Write the results to the file
        file.write(f"Results for dataset: {path}\n")
        file.write(f"Mean ADF Score: {adf_mean}\n")
        file.write(f"Mean KPSS Score: {kpss_mean}\n")
        file.write(f"Mean PP Score: {pp_mean}\n")
        file.write(f"Mean Hurst Score: {hurst_mean}\n")
        file.write(f"Mean Variance Ratio Score: {vr_mean}\n")
        file.write(f"General Stationarity Score: {general_score}\n")
        file.write("\n")  # Add a blank line between results for readability

print("Stationarity results have been saved to 'stationarity_results.txt'")
