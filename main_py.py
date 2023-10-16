import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup

# Load the model and data
df = pd.read_csv("balanced.csv")
y = df['y']
X = df.drop('y', axis=1)

from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Create a Streamlit app
st.title("Flood Prediction App")

# Collect user input
lat = st.number_input("Enter Latitude")
lon = st.number_input("Enter Longitude")


# Check if both latitude and longitude have been provided
if lat != 0.0 and lon != 0.0:
    # Make API request to get weather data
    url = "https://weatherapi-com.p.rapidapi.com/current.json"
    querystring = {"q": f"{lat},{lon}"}

    headers = {
        "X-RapidAPI-Key": "fd686eb4fbmsh09a45c03eb9a89fp1a3435jsnce9ce4bbab3e",
        "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)
    main = response.json()['current']
    location = response.json()['location']

    country = location['country']
    st.write("Location: ")
    st.write(location['name'])
    st.write(location['region'])
    st.write(country)
    

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

    # Make predictions
    predictions = model.predict(X_test)
    pr = predictions[0]

    # Display the result
    threshold = 0.5
    if pr >= threshold:
        st.write("Flood Prediction: Yes")
    else:
        st.write("Flood Prediction: No")
else:
    st.write("Enter Lat and Lon")




if lat!= 0.0 and long != 0.0:
  # Create a Streamlit app
  st.title("Most Vulnerable Cities to Flooding USA")
  # Send an HTTP GET request to the URL
  url = "https://insurify.com/insights/most-vulnerable-cities-to-flooding-2023/"
  response = requests.get(url)
  if response.status_code == 200:
      # Parse the HTML content of the page
      soup = BeautifulSoup(response.text, 'html.parser')

      # Find the relevant data on the page (you may need to inspect the HTML structure of the page)
      # In this case, we're looking for h3 elements
      h3_elements = soup.find_all('h3')

      # Extract and display the names using Streamlit
      names = [h3.get_text() for h3 in h3_elements]

      for name in names:
          st.write(name)

  else:
      st.write("Failed to retrieve the web page. Status code:", response.status_code)
else:
    st.write("-----")

