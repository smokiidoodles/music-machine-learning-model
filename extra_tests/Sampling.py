from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading dataset
# ---------------------------
data = pd.read_csv("dataset.csv", skiprows=0)

# Fixing track_genre values
data['track_genre'] = data['track_genre'].fillna('unknown').astype(str).str.strip().str.lower()

# Encoded genre
le = LabelEncoder()
data['track_genre_encoded'] = le.fit_transform(data['track_genre'])


# Prepare features
# ----------------------------
X = data.drop(columns=['track_genre', 'track_genre_encoded'])

# Convert to numeric & replace non-numeric with NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Replace inf, -inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Replace all NaN with 0
X = X.fillna(0)

# Target
y = data['track_genre_encoded']

#--------------------------------
# Taking a Sample of Data ( since the dataset is large )
#--------------------------------

# fraction of data
frac = 0.20   # keeps 20% of each genre class

# proportional stratified sampling
sampled_data = data.groupby('track_genre_encoded', group_keys=False).apply(
    lambda x: x.sample(frac=frac)
).reset_index(drop=True)

print("Original dataset size:", len(data))
print("Sampled dataset size:", len(sampled_data))

# Verify class distribution
print("Original distribution:")
print(data['track_genre_encoded'].value_counts(normalize=True))

print("\nSampled distribution:")
print(sampled_data['track_genre_encoded'].value_counts(normalize=True))
