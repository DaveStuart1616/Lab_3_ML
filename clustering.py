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
_ = sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)


# %% also try lmplot and pairplot
sns.set_style('whitegrid') 
sns.lmplot(x="area", 
    y="asymmetry_coefficient", 
    data = df
)
_ = plt.title("Asymmetry Coefficient v Area")

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


# %% Perimeter v compactness
sns.set_style('whitegrid') 
sns.lmplot(x="compactness", 
    y="perimeter", 
    data = df
)
_ = plt.title("Perimeter v Compactness")
# %%
