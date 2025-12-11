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

# Introducing stratified sample size
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y
)

# ---------------------------
# PCA and LDA
# ---------------------------
pca = PCA(n_components=3)
X_r = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=3)
X_r2 = lda.fit_transform(X, y)

print("PCA explained variance:", pca.explained_variance_ratio_)


# ---------------------------
# Plot PCA
# ---------------------------
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                color=color, alpha=0.8, lw=lw, label=f"Class {i}")

plt.title("PCA of Genre Dataset")
plt.legend()


# ---------------------------
# Plot LDA
# ---------------------------
plt.figure()

for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1],
                alpha=0.8, color=color, label=f"Class {i}")

plt.title("LDA of Genre Dataset")
plt.legend()

plt.show()
