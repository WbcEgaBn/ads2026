import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ==========================================
# 1. LOAD DATA & PREPARE SPLITS
# (80% training, 20% testing split)
# ==========================================
df = pd.read_csv('Combined_Wine.csv')

# Prepare features (X) and target (y)
#Dropping free_sulfur_dioxide and density as they are highly correlated with other features
X = df.drop(columns=['quality', 'quality_class'], errors='ignore')
y = df['quality_class']

# One-hot encode wine_type
X = pd.get_dummies(X, columns=['wine_type'], drop_first=True)

# Outer train-test split (80% train, 20% test - LOCKED AWAY)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)


# ==========================================
# 2. RECIPE & PREPROCESSING
# ==========================================
# Scaling features based ONLY on the training split to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. K-NEAREST NEIGHBORS (KNN) MODEL TRAINING
# WITH FOLD-LEVEL RESAMPLING & CV TUNING
# ==========================================
print('--- Training and Tuning KNN Model with CV Resampling ---')

# Define values of K to tune
k_values = np.arange(1, 22, 2)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)

mean_cv_scores = []

# Loop through each candidate K value
for k in k_values:
  fold_scores = []

  # K-Fold Cross Validation loop
  for train_idx, val_idx in cv.split(X_train_scaled, y_train):
    # Split into current fold's train and validation sets
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # --- RESAMPLE TRAINING FOLD TO BALANCE CLASSES ---
    # Combine training fold features and target temporarily
    train_fold_df = pd.DataFrame(X_tr)
    train_fold_df['target'] = y_tr.values

    df_low = train_fold_df[train_fold_df['target'] == 'Low']
    df_med = train_fold_df[train_fold_df['target'] == 'Medium']
    df_high = train_fold_df[train_fold_df['target'] == 'High']

    # Oversample minority classes ('Low' and 'High') to match 'Medium' count
    max_len = len(df_med)
    df_low_os = resample(
        df_low, replace=True, n_samples=max_len, random_state=2026
    )
    df_high_os = resample(
        df_high, replace=True, n_samples=max_len, random_state=2026
    )

    train_balanced = pd.concat([df_med, df_low_os, df_high_os])

    X_tr_balanced = train_balanced.drop(columns=['target']).values
    y_tr_balanced = train_balanced['target'].values

    # Train KNN on the balanced training fold
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(X_tr_balanced, y_tr_balanced)

    # Evaluate on the untouched validation fold using Macro F1
    preds = knn.predict(X_val)
    score = f1_score(y_val, preds, average='macro')
    fold_scores.append(score)

  # Average score across all folds for this K value
  mean_cv_scores.append(np.mean(fold_scores))

# Find the best K value
best_k_idx = np.argmax(mean_cv_scores)
best_k = k_values[best_k_idx]
best_score = mean_cv_scores[best_k_idx]


# ==========================================
# 4. KNN AUTOPLOT
# ==========================================
plt.figure(figsize=(8, 4))
plt.plot(
    k_values,
    mean_cv_scores,
    marker='o',
    color='b',
    linewidth=2,
)
plt.title('KNN Hyperparameter Tuning: Neighbors vs. CV Macro F1-Score (Balanced Folds)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Macro F1-Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(
    best_k,
    color='r',
    linestyle='--',
    label=f'Best K = {best_k}',
)
plt.legend()
plt.tight_layout()
plt.show()

print(f'Best KNN Parameter (K): {best_k}')
print(f'Best Cross-Validation Macro F1-Score: {best_score:.4f}\n')


# ==========================================
# 5. SOFTMAX REGRESSION MODEL TRAINING (Placeholder)
# ==========================================
# # softmax reg here
# Group mate task: Implement Multinomial Logistic Regression,
# set up a hyperparameter grid for C (regularization), and use GridSearchCV(cv=cv).


# ==========================================
# 6. NEURAL NETWORK MODEL TRAINING (Placeholder)
# ==========================================
# # neural network here
# Group mate task: Implement MLPClassifier,
# set up a hyperparameter grid for hidden_layer_sizes and alpha, and use GridSearchCV(cv=cv).


# ==========================================
# 7. MODEL COMPARISON & FINAL SELECTION
# ==========================================
# Compare cv results across models here before selecting the best one
# to evaluate against the unseen X_test / y_test data.