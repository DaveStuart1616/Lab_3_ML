# %% read data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
sns.distplot(train["SalePrice"])

# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
ax = sns.scatterplot(x="SalePrice",  #Price v OverAll Quality
    y="OverallQual",
    data=train
) 
_ = plt.title("Sale Price v Overall Quality Rating")

#%%
sns.distplot(train[train["CentralAir"]=="Y"]["SalePrice"], label="With CA")
sns.distplot(train[train["CentralAir"]=="N"]["SalePrice"], label="W/O CA")
plt.legend()
_ = plt.title("Sales Price v With or Without Central Air (CA)")

# %% Survived w.r.t CentralAir v SalePrice and GrLivArea
g = sns.FacetGrid(train, col="CentralAir", height=6, aspect=.8)
_ = g.map(sns.scatterplot, "SalePrice", "GrLivArea")

#%% Price v Overall Quality Regression Chart
sns.set_style('whitegrid') 
sns.lmplot(x="SalePrice", 
    y="OverallQual", 
    data = train
) 
_ = plt.title("Sale Price v Overall Quality Rating")

#%% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
ax = sns.scatterplot(x="SalePrice",  #Price v Central
    y="CentralAir",
    data=train
) 
_ = plt.title("Sale Price v Central Air")

#%%
ax = sns.barplot(y="SalePrice",  #Price v Yr Built 
    x="YearBuilt",
    orient="v",
    data=train
) 
_ = plt.title("Sale Price v Year Built")

# %% DUMMY REGRESSOR TEST
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# NEURAL NETWORK F_SCORE METHOD REGRESSION METHOD
#%% TRAINING DATASET DATA TRANSFORMATION

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
enc = OneHotEncoder(handle_unknown='ignore')

selected_columns = [
    "GrLivArea", 
    "OverallQual", 
    "YearBuilt",
    "CentralAir",
    "1stFlrSF", 
    "2ndFlrSF", 
    "LotArea",
    "Neighborhood",
    "MasVnrArea"
]
train_x = train[selected_columns]
train_y = train["SalePrice"]

train_x["GrLivArea"] = train_x["GrLivArea"].fillna("unknown")
train_x["MasVnrArea"] = train_x["OverallQual"].fillna("unknown")

ct = ColumnTransformer(
    [
        ("MasVnrArea_fillna", imp, ["MasVnrArea"]),
        ("ohe", enc, ["CentralAir", "Neighborhood"]),  
    ],
    remainder="passthrough"
)

train_x = ct.fit_transform(train_x)

# %% TRAINING SET F-SCORE CALC
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(max_iter=2000, random_state=2020)

mlp.fit(train_x, train_y)

print("Training Set Performance")
print(evaluate(mlp, train_x, train_y))

# %%  TESTING DATASET DATA TRANSFORMATION

selected_columns = [
    "GrLivArea", 
    "OverallQual", 
    "YearBuilt",
    "CentralAir",
    "1stFlrSF", 
    "2ndFlrSF", 
    "LotArea",
    "Neighborhood",
    "MasVnrArea"
]
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x["GrLivArea"] = test_x["GrLivArea"].fillna("unknown")
test_x["OverallQual"] = test_x["OverallQual"].fillna("unknown")

test_x = ct.transform(test_x)

#%% TESTING SET F-SCORE CALC
print("Test Set Performance")
print(evaluate(mlp, test_x, test_y))

# %%
