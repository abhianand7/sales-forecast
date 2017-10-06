import os
import platform
import re
import datetime

import numpy as np
import pandas as pd

# for DB connection
from sqlalchemy import create_engine
db_conx = create_engine('mysql+pymysql://root:abhinav@7@localhost:3306/prediction_output', echo=False)


pattern = re.compile(r'region_([0-9]+)_temp.csv')
date_pattern = re.compile(r'[a-z]{1,4},[0-9]{1,5},.*')

platform = platform.system()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

temperature_dir = os.path.join(BASE_DIR, 'temperature')

data_frame = pd.DataFrame(columns=['region', 'date', 'temp_min', 'temp', 'temp_max', 'humidity'])

count = 0

for filename in os.listdir(temperature_dir):
    region = pattern.search(filename).groups()[0]
    print(filename, region)
    file_path = os.path.join(temperature_dir, filename)
    with open(file_path, 'r') as fobj:
        for line in fobj.readlines():
            line = line.strip().lower()
            line_data = line.split(',')
            if date_pattern.search(line):
                date_str = '-'.join(line_data[:2])
                date_obj = datetime.datetime.strptime(date_str, '%b-%Y').date()
                # print(date_obj)
                continue
            else:
                print(line_data)
                date_obj = date_obj + datetime.timedelta(1)
                if 'na' in line_data:
                    data_frame.loc[count] = [region, date_obj, np.nan, np.nan, np.nan, np.nan]
                else:
                    data_frame.loc[count] = [region, date_obj, line_data[3], line_data[2], line_data[1], np.nan]
            count += 1
            print(date_obj)

data_frame.to_sql(name='Temperature', con=db_conx, if_exists='append')

