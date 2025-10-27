from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#load file
DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
df = pd.read_csv(DB_FILEPATH)

df = df.drop(columns=['name','kWh','datetime'])

OUTPUT = 'kWh Normalised'

X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[OUTPUT])
y = df[OUTPUT]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
pca.fit(X)
X_pca = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index = [f'PC{i+1}' for i in range(pca.n_components_)]
                        )


plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
pc_index = 0  # choose PC (0 for PC1, 1 for PC2, etc.)

loadings.iloc[pc_index].plot(kind='bar')
plt.title(f'Feature Contributions to {loadings.index[pc_index]}')
plt.xlabel('Feature')
plt.ylabel('Loading Value')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title("PCA Feature Loadings Heatmap")
plt.xlabel("Features")
plt.ylabel("Principal Components")
plt.tight_layout()
plt.show()








##Kernel approixmation
dr = Nystroem(kernel='rbf',gamma = 0.0002, n_components=10, random_state=510444756, n_jobs=-1)
X_nystroem = dr.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_nystroem, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 6. Evaluate performance ---
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")


print(f"Original shape: {X_scaled.shape}, Reduced shape: {X_nystroem.shape}")

# --- 4. Visualize Nyström components with PCA to 2D ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_nystroem)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='viridis', s=30, alpha=0.7)
plt.title('Nyström Approximation (visualized via PCA to 2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='kWh_normalised')
plt.show()

# --- 5. Train a regression model on the reduced data ---
X_train, X_test, y_train, y_test = train_test_split(X_nystroem, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(f"R² score: {r2_score(y_test, model.predict(X_test)):.3f}")

# --- 6. Estimate column importance via permutation on the original features ---
# Refit using the original scaled features (to attribute importance correctly)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_orig = Ridge(alpha=1.0)
model_orig.fit(X_train_orig, y_train_orig)

result = permutation_importance(model_orig, X_test_orig, y_test_orig, n_repeats=10, random_state=42)
importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

# --- 7. Visualize feature importances ---
plt.figure(figsize=(10,6))
importances.head(25).plot(kind='bar')
plt.title('Top 15 Feature Importances (via Permutation)')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()
