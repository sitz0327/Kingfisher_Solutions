from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#load file
DB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "DBcleaned.csv")
df = pd.read_csv(DB_FILEPATH)

df = df.drop(columns=['name','id','kWh','datetime','maximumPower','month','day','hour','dayOfYear', 'year', 'Cloud Fill Flag', 'Cloud Type', 'Fill Flag'])

OUTPUT = 'kWh Normalised'

X = df.select_dtypes(include=['float64', 'int64'])
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