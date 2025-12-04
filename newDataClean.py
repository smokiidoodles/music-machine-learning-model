from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pandas as pd
import numpy as np


d = pd.read_csv("dataset.csv", skiprows=0)
print(d)

feature_cols = ['track_genre','danceability','energy','acousticness','instrumentalness','tempo','duration_ms','popularity']
non_featured_cols = ['track_name','track_id','explicit','artists','key','album_name', 'mode','speechiness','liveness','valence','time_signature']

target_col = 'popularity'


print("This is to split the two data verisons")
data= d.drop(non_featured_cols, axis='columns')

pd.set_option('display.max_columns', None)
#print(data)


from sklearn.preprocessing import OneHotEncoder


# Flatten lists if present
data['track_genre'] = data['track_genre'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Normalize genre strings
data['track_genre'] = data['track_genre'].fillna('unknown').str.strip().str.lower()

# One-hot encode track_genre
ohe = OneHotEncoder(sparse_output=False)  # categories auto-detected
genre_encoded = ohe.fit_transform(data[['track_genre']])


# Convert to DataFrame with proper column names
genre_df = pd.DataFrame(genre_encoded, columns=ohe.get_feature_names_out(['track_genre']))


# Combine numeric features with one-hot genres
X = pd.concat([data.drop('track_genre', axis=1), genre_df], axis=1)

print(X.head())
print(data.head())