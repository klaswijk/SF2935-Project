import numpy as np
import pandas as pd
from scipy import stats


categorical_features = ("key",)
binary_features = ("mode",)
numerical_features = (
    "danceability", 
    "energy", 
    "loudness", 
    "speechiness", 
    "acousticness", 
    "instrumentalness", 
    "liveness", 
    "valence", 
    "tempo"
)
all_features = categorical_features + binary_features + numerical_features


def get_unlabeled_data():
    df = pd.read_csv("data/project_unlabeled.csv")
    return df


def get_training_data(outliers="remove"):
    df = pd.read_csv("data/project_train.csv")

    if outliers == "remove":
        df = remove_outliers(df)
    elif outliers == "nan":
        df = remove_outlier_features(df)
    elif outliers != "ignore":
        raise ValueError(f"Unkown argument '{outliers=}'")
    
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    return X, y


def get_test_data():
    df = pd.read_csv("data/project_test.csv")
    return df


def remove_outliers(df, tol=4):
    """
    Remove samples that deviate more than 'tol' standard deviations
    in any feature.
    """
    return df[(np.abs(stats.zscore(df)) < tol).all(axis=1)]


def remove_outlier_features(df, tol=4):
    """
    Sets all feature values that deviate more than 'tol' standard
    deviations to 'np.nan'.
    """
    return df.where(np.abs(stats.zscore(df)) < tol, np.nan)


def append_unlabeled_input(transformer, X):
    """
    Append unlabeled input to the transformer's 'fit' method.
    If 'pass_through' is True, pass the unlabeled data through the
    transformer's 'transform' method.

    Assumes y is not used by the transformer's methods.
    """
    transformer.fit = lambda X_, _: transformer.fit(np.vstack([X_, X]))
    transformer.transform = lambda X_, _: transformer.transform(np.vstack([X_, X]))
    transformer.fit_transform = lambda X_, _: transformer.fit_transform(np.vstack([X_, X]))
    return transformer


def remove_unlabeled_output(transformer, size):
    """
    Remove unlabeled data from the transformer's output. 
    The unlabeled data is assumed to be at the tail end of the output.
    """
    transformer.transform = lambda X, y: transformer.transform(X, y)[:-size]
    transformer.fit_transform = lambda X, y: transformer.fit_transform(X, y)[:-size]
    return transformer