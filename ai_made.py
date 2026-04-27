"""
Teen Mental Health Dataset — Seminar-style ML Research
=======================================================
Tasks:
  1. EDA & Visualization
  2. Data Cleaning & Preprocessing
  3. Train/Test Split with class imbalance handling (SMOTE)
  4. Algorithms: Logistic Regression, KNN, Decision Tree, Random Forest
  5. Hyperparameter Tuning (GridSearchCV)
  6. Evaluation: Accuracy, F1, ROC-AUC, Confusion Matrix
  7. Final comparison plot
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nBasic stats:\n{df.describe().round(2)}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"\nClass distribution:\n{df['depression_label'].value_counts()}")
print(f"Imbalance ratio: {df['depression_label'].value_counts()[0] / df['depression_label'].value_counts()[1]:.1f}:1")


# ─────────────────────────────────────────────
# 2. EDA — Figure 1: Distribution Overview
# ─────────────────────────────────────────────
fig1, axes = plt.subplots(3, 3, figsize=(15, 12))
fig1.suptitle("EDA — Feature Distributions", fontsize=16, fontweight='bold', y=1.01)

numeric_cols = ['age', 'daily_social_media_hours', 'sleep_hours',
                'screen_time_before_sleep', 'academic_performance',
                'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level']

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974']

for ax, col, color in zip(axes.flat, numeric_cols, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    mean_val = df[col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f'mean={mean_val:.1f}')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig1_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] fig1_distributions.png")


# ─────────────────────────────────────────────
# EDA — Figure 2: Correlation Heatmap
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 8))
corr = df[numeric_cols + ['depression_label']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_lower = corr.where(~mask)

im = ax.imshow(corr_lower, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)

labels = [c.replace('_', '\n') for c in numeric_cols + ['depression_label']]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=8)

for i in range(corr_lower.shape[0]):
    for j in range(corr_lower.shape[1]):
        val = corr_lower.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < 0.7 else 'white')

ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig2_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] fig2_correlation.png")


# ─────────────────────────────────────────────
# EDA — Figure 3: Group Analysis
# ─────────────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("Group Analysis", fontsize=14, fontweight='bold')

# 3a. Avg stress/anxiety by platform
platform_means = df.groupby('platform_usage')[['stress_level', 'anxiety_level', 'addiction_level']].mean()
x = np.arange(len(platform_means))
width = 0.25
axes[0].bar(x - width, platform_means['stress_level'], width, label='Stress', color='#C44E52')
axes[0].bar(x, platform_means['anxiety_level'], width, label='Anxiety', color='#4C72B0')
axes[0].bar(x + width, platform_means['addiction_level'], width, label='Addiction', color='#55A868')
axes[0].set_xticks(x)
axes[0].set_xticklabels(platform_means.index)
axes[0].set_title("Avg Levels by Platform")
axes[0].legend()
axes[0].spines[['top', 'right']].set_visible(False)

# 3b. Sleep hours by age group
df['age_group'] = pd.cut(df['age'], bins=[12, 14, 16, 19], labels=['13–14', '15–16', '17–19'])
age_sleep = df.groupby('age_group', observed=True)[['sleep_hours', 'daily_social_media_hours']].mean()
x2 = np.arange(len(age_sleep))
axes[1].bar(x2 - 0.2, age_sleep['sleep_hours'], 0.4, label='Sleep hrs', color='#8172B2')
axes[1].bar(x2 + 0.2, age_sleep['daily_social_media_hours'], 0.4, label='Social media hrs', color='#DD8452')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(age_sleep.index)
axes[1].set_title("Sleep vs Social Media by Age Group")
axes[1].legend()
axes[1].spines[['top', 'right']].set_visible(False)

# 3c. Depression label by gender
gender_dep = df.groupby('gender')['depression_label'].mean() * 100
axes[2].bar(gender_dep.index, gender_dep.values, color=['#4C72B0', '#DD8452'], width=0.4, edgecolor='white')
axes[2].set_title("Depression Rate by Gender (%)")
axes[2].set_ylabel("%")
for i, (idx, val) in enumerate(gender_dep.items()):
    axes[2].text(i, val + 0.2, f'{val:.1f}%', ha='center', fontweight='bold')
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig3_group_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] fig3_group_analysis.png")


# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("PREPROCESSING")
print("=" * 55)

df_ml = df.copy()

# Encode categoricals
le = LabelEncoder()
df_ml['gender']                 = le.fit_transform(df_ml['gender'])          # male=1, female=0
df_ml['platform_usage']         = le.fit_transform(df_ml['platform_usage'])  # ordinal not ideal, but sufficient
df_ml['social_interaction_level'] = df_ml['social_interaction_level'].map({'low': 0, 'medium': 1, 'high': 2})

# Drop derived column
df_ml.drop(columns=['age_group'], inplace=True)

# Features / target
X = df_ml.drop(columns=['depression_label'])
y = df_ml['depression_label']

print(f"Features shape: {X.shape}")
print(f"Feature columns: {list(X.columns)}")

# Train/test split — stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train class dist: {dict(y_train.value_counts())}")

# Handle imbalance with SMOTE on TRAIN only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Train class dist: {dict(pd.Series(y_train_res).value_counts())}")

# Scale features (important for LR, KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 4. MODEL TRAINING + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("HYPERPARAMETER TUNING (GridSearchCV, cv=5)")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Logistic Regression ---
lr_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
lr_grid   = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=cv, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train_sc, y_train_res)
print(f"\nLogistic Regression best params: {lr_grid.best_params_}  |  CV F1={lr_grid.best_score_:.3f}")

# --- KNN ---
knn_params = {'n_neighbors': [3, 5, 7, 9, 11, 15], 'weights': ['uniform', 'distance']}
knn_grid   = GridSearchCV(KNeighborsClassifier(), knn_params, cv=cv, scoring='f1', n_jobs=-1)
knn_grid.fit(X_train_sc, y_train_res)
print(f"KNN best params: {knn_grid.best_params_}  |  CV F1={knn_grid.best_score_:.3f}")

# --- Decision Tree ---
dt_params = {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
dt_grid   = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=cv, scoring='f1', n_jobs=-1)
dt_grid.fit(X_train_sc, y_train_res)
print(f"Decision Tree best params: {dt_grid.best_params_}  |  CV F1={dt_grid.best_score_:.3f}")

# --- Random Forest ---
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
rf_grid   = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=cv, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_sc, y_train_res)
print(f"Random Forest best params: {rf_grid.best_params_}  |  CV F1={rf_grid.best_score_:.3f}")


# ─────────────────────────────────────────────
# 5. EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)

models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'KNN':                 knn_grid.best_estimator_,
    'Decision Tree':       dt_grid.best_estimator_,
    'Random Forest':       rf_grid.best_estimator_,
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    results[name] = {'Accuracy': acc, 'F1': f1, 'ROC-AUC': auc, 'cm': cm, 'y_prob': y_prob, 'y_pred': y_pred}
    print(f"\n{name}")
    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, zero_division=0))


# ─────────────────────────────────────────────
# 6. FIGURE 4: Confusion Matrices
# ─────────────────────────────────────────────
fig4, axes = plt.subplots(1, 4, figsize=(18, 4))
fig4.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = res['cm']
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(name, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('fig4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] fig4_confusion_matrices.png")


# ─────────────────────────────────────────────
# 7. FIGURE 5: ROC Curves
# ─────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] fig5_roc_curves.png")


# ─────────────────────────────────────────────
# 8. FIGURE 6: Model Comparison Bar Chart
# ─────────────────────────────────────────────
fig6, ax = plt.subplots(figsize=(10, 5))

model_names = list(results.keys())
metrics     = ['Accuracy', 'F1', 'ROC-AUC']
metric_colors = ['#4C72B0', '#DD8452', '#55A868']
x = np.arange(len(model_names))
width = 0.25

for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric, color=color, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Accuracy / F1 / ROC-AUC', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fig6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] fig6_model_comparison.png")


# ─────────────────────────────────────────────
# 9. FIGURE 7: Feature Importances (RF)
# ─────────────────────────────────────────────
rf_best = rf_grid.best_estimator_
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = list(X.columns)

fig7, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(
    [feature_names[i] for i in indices[::-1]],
    importances[indices[::-1]],
    color='#4C72B0', edgecolor='white'
)
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('fig7_feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] fig7_feature_importances.png")

print("\n" + "=" * 55)
print("ALL DONE")
print("=" * 55)