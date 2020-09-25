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

# %% Survived w.r.t CentralAir / OverallQual / BldgType / etc
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

#%% Price v Overall Quality Regression Chart
#Transform Central Air
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
ct = ColumnTransformer(
    [
        ("CentralAir_t", enc, ["CentralAir"])
    ],
    remainder="passthrough"
)
ct.fit_transform(train)

sns.set_style('whitegrid') 
sns.lmplot(x="SalePrice", 
    y="Central Air", 
    data = train
#plt.title("Sale Price v Central Air")

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood


# %%
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


# %% your solution to the regression problem
