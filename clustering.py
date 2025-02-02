# %% read data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None) 
# header as no titles available and tab separated ("\t+"")
# 1 area A,
# 2 perimeter P,
# 3 compactness C = 4*pi*A/P^2,
# 4 length of kernel,
# 5 width of kernel,
# 6 asymmetry coefficient
# 7 length of kernel groove.
# 8 target
df.columns = [      #create df with column names
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

#%%  Scatter Asymmetry, Area v Target
_ = sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)

# %% LMPLOT Asymmetry v Compactness
sns.set_style('whitegrid') 
sns.lmplot(x="compactness", 
    y="asymmetry_coefficient",
    hue="target",
    data = df
)
_ = plt.title("Asymmetry Coefficient v Compactness, By Target")

# %% LMPLOT Asymmetry v Area, Hue of Target
sns.set_style('whitegrid') 
sns.lmplot(x="perimeter", 
    y="length_kernel_groove",
    hue="target",
    data = df
)
_ = plt.title("Perimeter v Kernel Length Groove, by Target")

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
_ = sns.pairplot(df, 
    hue="target",
)

#%% PAIRGRID
sns.set_style('darkgrid')
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
_  = g.map_diag(sns.kdeplot)

# %% determine the best number of clusters
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
_ = ax.figure.legend()

# HEIRARCHICAL METHOD
# %% HEIRARCHICAL CALC HOMOGENEITY

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import homogeneity_score

x = df.drop("target", axis=1)
y = df["target"]
#inertia = {}
homogeneity = {}
# use kmeans to loop over candidate number of clusters 
# store inertia and homogeneity score in each iteration

for k in range(1, 10):
    km = KMeans(n_clusters=k)
    pred = km.fit_predict(x)
    #inertia[k] = km.inertia_
    homogeneity[k] = homogeneity_score(y, pred)

# %% PLOT HEIRARCHICAL HOMOGENEITY
# ax = sns.lineplot(
#     x=list(inertia.keys()),
#     y=list(inertia.values()),
#     color="blue",
#     label="inertia",
#     legend=None,
# )
# ax.set_ylabel("inertia")
# ax.twinx()
ax = sns.lineplot(
    x=list(homogeneity.keys()),
    y=list(homogeneity.values()),
    color="red",
    label="homogeneity",
    legend=None,
)
_ = ax.set_ylabel("homogeneity")
#ax.figure.legend()
#X = {df["target"]}

# %%
