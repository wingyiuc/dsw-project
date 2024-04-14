# pip install ydata-profiling
import pandas as pd
from ydata_profiling import ProfileReport

file_path = 'Airbnb_Data.csv'
data = pd.read_csv(file_path)

profile = ProfileReport(data, title='Airbnb Data Profile', explorative=True)
profile.to_file("Airbnb_Data_Profile.html")