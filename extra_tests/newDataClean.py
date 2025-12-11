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

# Loading dataset
d = pd.read_csv("dataset.csv", skiprows=0)

# Removed columns not used for modeling
non_featured_cols = ['track_name', 'track_id', 'explicit', 'artists', 'key','album_name', 'mode',
                     'speechiness', 'liveness','valence', 'time_signature']

data = d.drop(non_featured_cols, axis='columns')

# Processing track_genre column
data['track_genre'] = data['track_genre'].apply(
    lambda x: x[0] if isinstance(x, list) else x
)

data['track_genre'] = data['track_genre'].fillna('unknown').str.strip().str.lower()

# Encode genre
le = LabelEncoder()
data['track_genre_encoded'] = le.fit_transform(data['track_genre'])


# Features and target
value_to_drop = 'world-music'
data = data.drop(data[data['track_genre'] == value_to_drop].index)
X = data.drop(columns=['track_genre', 'track_genre_encoded'])

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = data['track_genre_encoded']
print(data)