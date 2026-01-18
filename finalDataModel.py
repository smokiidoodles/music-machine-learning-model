
# 1. IMPORTS

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

# ------------------------
# LOAD DATA
# ------------------------
data = pd.read_csv("dataset.csv", index_col=0)
#index_col=0 removes the automatic ID column assigned to each song

non_featured_cols = [
    'track_name','track_id','explicit','artists','key','album_name','mode','time_signature'
]
data = data.drop(columns=non_featured_cols, errors='ignore')

# Process track_genre
data['track_genre'] = data['track_genre'].apply(lambda x: x[0] if isinstance(x, list) else x)
data['track_genre'] = data['track_genre'].fillna('unknown').str.lower().str.strip()

# ------------------------
# 141 Genres TO 4 Genres
# ------------------------
def map_genre_4(g):
    g = g.lower()
    if any(k in g for k in ['pop','k-pop','j-pop','c-pop','electro']):
        return 'pop'
    if any(k in g for k in ['rock','alt','indie','garage','classic','metal','doom','thrash']):
        return 'rock'
    if any(k in g for k in ['hip hop','rap','trap']):
        return 'hiphop'
    if any(k in g for k in ['jazz','blues','r&b','soul','ambient','classical','chill']):
        return 'jazz_soothing'
    return None

data['genre_grouped'] = data['track_genre'].apply(map_genre_4)
data = data.dropna(subset=['genre_grouped'])


# Encode target
le = LabelEncoder()
data['genre_encoded'] = le.fit_transform(data['genre_grouped'])

# ------------------------
# FEATURES & TARGET
# ------------------------
X = data.drop(columns=['track_genre','genre_grouped','genre_encoded'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = data['genre_encoded']

# ------------------------
# FEATURE ENGINEERING
# ------------------------
X['energy_acoustic_ratio'] = X['energy'] / (X['acousticness'] + 1e-5)
X['loud_instr'] = X['loudness'] * X['instrumentalness']
X['tempo_bin'] = pd.cut(X['tempo'], bins=3, labels=[0,1,2])
X['duration_bin'] = pd.cut(X['duration_ms'], bins=3, labels=[0,1,2])

# -------------------------------
# TRAIN/TEST SPLIT (stratified)
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Scaled features for KNN and ensemble
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------
# CLASSIFIERS
# --------------------------------
rf = RandomForestClassifier(
    n_estimators=500, max_depth=40, min_samples_leaf=3, class_weight='balanced')

dt = DecisionTreeClassifier(class_weight='balanced')

knn = KNeighborsClassifier(n_neighbors=7)


# CROSS-VALIDATION
# --------------------------------
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'balanced_accuracy': make_scorer(balanced_accuracy_score)
}

cv = ShuffleSplit(n_splits=5, test_size=0.2)

print("\nRandom Forest CV:")
rf_scores = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring)
print("Accuracy:", np.mean(rf_scores['test_accuracy']))
print("Precision:", np.mean(rf_scores['test_precision']))
print("Balanced Accuracy:", np.mean(rf_scores['test_balanced_accuracy']))

print("\nDecision Tree CV:")
dt_scores = cross_validate(dt, X_train, y_train, cv=cv, scoring=scoring)
print("Accuracy:", np.mean(dt_scores['test_accuracy']))
print("Precision:", np.mean(dt_scores['test_precision']))
print("Balanced Accuracy:", np.mean(dt_scores['test_balanced_accuracy']))

print("\nKNN CV:")
knn_scores = cross_validate(knn, X_train_scaled, y_train, cv=cv, scoring=scoring)
print("Accuracy:", np.mean(knn_scores['test_accuracy']))
print("Precision:", np.mean(knn_scores['test_precision']))
print("Balanced Accuracy:", np.mean(knn_scores['test_balanced_accuracy']))

# --------------------------------
# 9. VOTING ENSEMBLE
# --------------------------------
ensemble = VotingClassifier(estimators=[('RF', rf), ('DT', dt), ('KNN', knn)], voting='soft')

ensemble.fit(X_train_scaled, y_train)
y_pred = ensemble.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
bal_acc = balanced_accuracy_score(y_test, y_pred)

print("\nEnsemble Test Accuracy:", acc)
print("Ensemble Test Precision:", prec)
print("Ensemble Test Balanced Accuracy:", bal_acc)

print("\nFIRST 10 ENSEMBLE PREDICTIONS:")
for i in range(10):
    true_label = le.inverse_transform([y_test.iloc[i]])[0]
    pred_label = le.inverse_transform([y_pred[i]])[0]
    print(f"Sample {i+1}: True = {true_label} | Predicted = {pred_label}")
