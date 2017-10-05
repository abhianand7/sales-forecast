# This script will collect temperature data for different cities upto 16 days in advance

import requests
import pandas as pd
import datetime
import logging
import json

# for DB connection
from sqlalchemy import create_engine
db_conx = create_engine('mysql+pymysql://root:abhinav@7@localhost:3306/prediction_output', echo=False)

openweather_base_api_url = "http://api.openweathermap.org/data/2.5/weather?"
query_by_city_id = "id={}"
query_by_coord = "lat={0}&lon={1}"
query_by_city_and_country = "q={0},{1}"
api_key = '&appid=2cd4aa684d4d906c992e7d2816210386'

city_list_file = './temp_city.json'

try:
    with open(city_list_file, 'r') as fobj:
        try:
            city_list_data = json.load(fobj)
        except EOFError:
            logging.critical("{}, File Empty".format(city_list_file))
except FileNotFoundError:
    logging.critical("File Not Found, please input correct file path.")
except IOError:
    logging.critical("Please ensure the correct path read access to the file.")


today_date = datetime.datetime.now().date()

data_frame = pd.DataFrame(columns=['region', 'date', 'temp_min', 'temp', 'temp_max', 'humidity'])
count = 0
for sale_region in city_list_data.keys():
    lat = city_list_data[sale_region]['lat']
    lon = city_list_data[sale_region]['lon']
    print(lat, lon)
    if lat and lon:
        weather_data = requests.get(openweather_base_api_url + query_by_coord.format(lat, lon) + api_key)
        weather_data = weather_data.json()

        weather_data = weather_data['main']
        data_frame.loc[count] = [sale_region, today_date, weather_data['temp_min'], weather_data['temp'],
                                 weather_data['temp_max'], weather_data['humidity']]
        count += 1
    else:
        pass


# print(data_frame)
data_frame.to_sql(name='Temperature', con=db_conx, if_exists='append', index=False)


