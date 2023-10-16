
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd


import requests

df  = pd.read_csv("balanced.csv")

y = df['y']
X = df.drop('y', axis=1)

from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model using your training data (X_train, y_train)
model.fit(X_train, y_train)

lat = float(input("Lat"))
lon = float(input("Lon"))



url = "https://weatherapi-com.p.rapidapi.com/current.json"

querystring = {"q": f"{lat},{lon}"}

headers = {
	"X-RapidAPI-Key": "fd686eb4fbmsh09a45c03eb9a89fp1a3435jsnce9ce4bbab3e",
	"X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

main = response.json()['current']
location = response.json()['location']




Max_Temp = main['temp_f']
Min_Temp = main['temp_c']
Rainfall = main['precip_mm'] * 100
Relative_Humidity = main['humidity']
Wind_Speed = main['wind_kph']
Cloud_Coverage = main['cloud']
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




