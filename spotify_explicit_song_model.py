import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix

# df_songs contains features of songs and whether they are explicit or not to use as training data
df_songs = pd.read_csv("https://www.dropbox.com/s/hijzbof7nnche09/top_artists_spotify-no_labels.csv?dl=1")
df_songs['Explicit'] = df_songs['Explicit'].astype('int')
features = ["Acousticness", "Danceability", "Duration", "Energy", "Instrumentalness", 
            "Liveness", "Loudness", "Speechiness", "Tempo", "Valence"]
cat_features = ["Mode", "TimeSignature"]

# Setting up preprocessing pipeline. Numerical features are standardized, and categorical features are one-hot encoded
ct = make_column_transformer(
    (StandardScaler(), features),
    (OneHotEncoder(), cat_features),
    remainder="drop"  # Drop other columns that aren't needed
)

# Logistic regression to classify songs as explicit or not
pipeline = make_pipeline(
    ct,
    LogisticRegression(penalty='none', solver='newton-cg')
)

# Train the logistic regression model using df_songs dataset
all_features = features + cat_features
pipeline.fit(df_songs[all_features], df_songs['Explicit'])

# Load test data to test the prediction model
top = pd.read_csv("https://www.dropbox.com/scl/fi/j9yz0nqnm4d0sqafineyb/twenty-one-pilots.csv?rlkey=5sw0m763g4mxwzpu4t529hmc7&dl=1")
top['Explicit'] = top['Explicit'].astype('int')
top_cleaned = top.drop(columns=['Album', 'Artist', 'Name', 'Explicit'])
top_dropped = top_cleaned[all_features].dropna()

predictions = pipeline.predict(top_dropped)

# Compare predicted explicit labels with the actual labels using confusion matrix
conf_matrix = pd.DataFrame(
    confusion_matrix(top['Explicit'], predictions),
    columns=pipeline.classes_,
    index=pipeline.classes_
)
