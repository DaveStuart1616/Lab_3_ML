# %% read data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# 1 area A,
# 2 perimeter P,
# 3 compactness C = 4*pi*A/P^2,
# 4 length of kernel,
# 5 width of kernel,
# 6 asymmetry coefficient
# 7 length of kernel groove.
# 8 target
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]


# %%
df.describe()

#%%
df.info()

#%% 
_ = sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)

# %% LMPLOT Asymmetry v Area
sns.set_style('whitegrid') 
sns.lmplot(x="area", 
    y="asymmetry_coefficient",
    data = df
)
_ = plt.title("Asymmetry Coefficient v Area")

# %% LMPLOT Asymmetry v Area, Hue of Target
sns.set_style('whitegrid') 
sns.lmplot(x="area", 
    y="asymmetry_coefficient",
    hue="target",
    data = df
)
_ = plt.title("Asymmetry Coefficient v Area, by Target")

# %% Perimeter v compactness
sns.set_style('whitegrid') 
sns.lmplot(x="compactness", 
    y="perimeter", 
    data = df
)
_ = plt.title("Perimeter v Compactness")
#%% PAIRPLOT
sns.set_style('whitegrid')
sns.pairplot(df, 
    hue="target",
)

#%% PAIRGRID
sns.set_style('darkgrid')
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)

# %% determine the best numbmer of clusters
from sklearn.cluster import KMeans 
from sklearn.metrics import homogeneity_score


x = df.drop("target", axis=1)
y = df["target"]
inertia = {}
homogeneity = {}
# use kmeans to loop over candidate number of clusters 
# store inertia and homogeneity score in each iteration

for k in range(1, 10):
    km = KMeans(n_clusters=k)
    pred = km.fit_predict(x)
    inertia[k] = km.inertia_
    homogeneity[k] = homogeneity_score(y, pred)

# %% PLOT
ax = sns.lineplot(
    x=list(inertia.keys()),
    y=list(inertia.values()),
    color="blue",
    label="inertia",
    legend=None,
)
ax.set_ylabel("inertia")
ax.twinx()
ax = sns.lineplot(
    x=list(homogeneity.keys()),
    y=list(homogeneity.values()),
    color="red",
    label="homogeneity",
    legend=None,
)
ax.set_ylabel("homogeneity")
ax.figure.legend()

# DBSCAN CLUSTER ANALYSIS
#%%
X = {df["target"]}

# call leaner
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(x)
pred = dbscan.fit_predict(x)

#%%
## metrics
from sklearn.metrics import homogeneity_score
print(f"homogeneity: {homogeneity_score(y, pred)}")

#%% plot
import seaborn as sns
import matplotlib.pyplot as plt

_, axes = plt.subplots(1, 2)
sns.scatterplot(
x=[x[0] for x in X] ,y=[x[1] for x in X], hue=y,ax=axes[0]
)
sns.scatterplot(
x=[x[0] for x in X] ,y=[x[1] for x in X], hue=pred,ax=axes[1]
)
plt.show()
