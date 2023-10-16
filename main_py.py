# %% [markdown]
# <h1>Loading Dataset</h1>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

# %%
# Load the dataset
df  = pd.read_csv("balanced.csv")


# %% [markdown]
# <h1>Exploring the dataset</h1>

# %%
# Check for duplicates
column_data_types = df.dtypes
column_data_types

# %%


# %% [markdown]
# <h1>Model Building</h1>

# %%
y = df['y']
X = df.drop('y', axis=1)

# %% [markdown]
# <h3>Using Gradient Boosting</h3>

# %%
from sklearn.ensemble import GradientBoostingRegressor

# %% [markdown]
# Split to get the Test and Train data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create a Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model using your training data (X_train, y_train)
model.fit(X_train, y_train)

# %% [markdown]
# Predict

# %% [markdown]
# Extract Client Data

# %%
lat = float(input("Lat"))
lon = float(input("Lon"))

# %%
import requests

url = "https://weatherapi-com.p.rapidapi.com/current.json"

querystring = {"q": f"{lat},{lon}"}

headers = {
	"X-RapidAPI-Key": "fd686eb4fbmsh09a45c03eb9a89fp1a3435jsnce9ce4bbab3e",
	"X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

main = response.json()['current']
location = response.json()['location']

main

# %%
location

# %%
import pandas as pd

Max_Temp = main['temp_f']
Min_Temp = main['temp_c']
Rainfall = main['precip_mm'] * 100
Relative_Humidity = main['humidity']
Wind_Speed = main['wind_kph']
Cloud_Coverage = x['cloud']
LATITUDE = location['lat']	
LONGITUDE = location['lon']

data = {
    'Max_Temp': [Max_Temp],
    'Min_Temp': [Min_Temp],
    'Rainfall': [Rainfall],
    'Relative_Humidity': [Relative_Humidity],
    'Wind_Speed': [Wind_Speed],
    'Cloud_Coverage': [Cloud_Coverage],
    'LATITUDE': [LATITUDE],
    'LONGITUDE': [LONGITUDE]
}

X_test = pd.DataFrame(data)


# %%
predictions = model.predict(X_test)

pr = predictions[0]

threshold = 0.5  

# Make the binary prediction
if pr >= threshold:
    prediction = 1  # Flood
    print("Flood ? Yes")
else:
    prediction = 0  # No flood
    print("Flood ? No")




