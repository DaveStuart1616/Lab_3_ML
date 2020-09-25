# %% read data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
_ = sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked 
_ = sns.countplot(x="Survived", hue="Sex", data=train)

#%%
_ = sns.countplot(x="Survived", hue="Embarked", data=train)

#%%
_ = sns.countplot(x="Survived", hue="Pclass", data=train)

#%% ALL VARIABLES
g = sns.FacetGrid(train, col="Sex", height=6, aspect=.8)
_ = g.map(sns.barplot, "Pclass", "Survived", "Embarked")


#%% Survived wrt Sex Count
#Female
# df = train
# df_f = df[df["Sex"] == "female"]
# df_fs = df_f[df_f["Survived"] == 1]
# df_fd = df_f[df_f["Survived"] == 0]

# _ = sns.displot(df_fs, x="Age", binwidth=3)
# _ = plt.title("Female Survived")

# #Male
# df_m = df[df["Sex"] == "male"]
# df_ms = df_m[df_m["Survived"] == 1]
# df_md = df_m[df_m["Survived"] == 0]

# _ = sns.displot(df_ms, x="Age", binwidth=3)
# _ = plt.title("Male Survived")


# %% Age distribution ?
_ = sns.displot(train, x="Age", binwidth=3)
_ = plt.title("Age Distribution-All Passengers")

# %% Survived w.r.t Age distribution ?
sns.distplot(train[train["Survived"]==1]["Age"], label="survived")
sns.distplot(train[train["Survived"]==0]["Age"], label="perished")
plt.legend()

#%%
#df_s = train ==
#_ = sns.distplot(df_s["Age"])

#df_s = df_s[df_s["Survived"] == 1] #Those who survived
#_ = sns.displot(df_s, x="Age", binwidth=3) #survivor plot
#_ = plt.title("Survived") 

#df_d = train
#df_d = df_d[df_d["Survived"] == 0] #Those who died
#_ = sns.displot(df_d, x="Age", binwidth=3) #non-survivor plot
#_ = plt.title("Perished")

#%% #Both Histograms above
# _ = sns.distplot(df_s["Age"], #Survivor
#     hist=False,
#     color="blue",
#     label="Survived",
#     #,
# ) 

# _ = sns.distplot(df_d["Age"], #non-survivor plot
#     hist=False,
#     color="green",
#     label="Perished",
#     #,
# ) 
# _ = plt.title("Survived and Perished by Age")
# _ = plt.legend()

# %% SibSp / Parch distribution ?
sns.displot(train, x="SibSp", binwidth=1)
_ = plt.title("Number of Siblings")


# %% Survived w.r.t SibSp / Parch  ?
sns.displot(train, x="Parch", binwidth=1)
_ = plt.title("Parch - Number of Parent/Children")

#DUMMY CLASSIFIER
# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass", "Age"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")

# NEURAL NETWORK F_SCORE METHOD CLASSIFICATION
#%% TRAINING DATASET DATA TRANSFORMATION

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
enc = OneHotEncoder(handle_unknown='ignore')

selected_columns = ["Pclass", "Age", "Embarked", "Sex", "Cabin"]
train_x = train[selected_columns]
train_y = train["Survived"]

train_x["Embarked"] = train_x["Embarked"].fillna("unknown")
train_x["Cabin"] = train_x["Cabin"].fillna("unknown")

ct = ColumnTransformer(
    [
        ("age_fillna", imp, ["Age"]),
        ("ohe", enc, ["Embarked", "Sex", "Cabin"]),  
    ],
    remainder="passthrough"
)

train_x = ct.fit_transform(train_x)

# %% TRAINING SET F-SCORE CALC
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
mlp.fit(train_x, train_y)

print("Training Set Performance")
print(evaluate(mlp, train_x, train_y))

# %%  TESTING DATASET DATA TRANSFORMATION

selected_columns = ["Pclass", "Age", "Embarked", "Sex", "Cabin"]
test_x = test[selected_columns]
test_y = truth["Survived"]

test_x["Embarked"] = test_x["Embarked"].fillna("unknown")
test_x["Cabin"] = test_x["Cabin"].fillna("unknown")

test_x = ct.transform(test_x)

#%% TESTING SET F-SCORE CALC
print("Test Set Performance")
print(evaluate(mlp, test_x, test_y))

# %%
