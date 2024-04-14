# pip install ydata-profiling
import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: auto profiling
file_path = 'Airbnb_Data.csv'
data = pd.read_csv(file_path)

profile = ProfileReport(data, title='Airbnb Data Profile', explorative=True)
profile.to_file("Airbnb_Data_Profile.html")


# Step 2: Correlation matrix

data = pd.read_csv("Airbnb_Data.csv")
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Map of Airbnb Data')
plt.show()


threshold = 0.8
strong_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
strong_pairs = strong_pairs[strong_pairs.between(threshold, 1, inclusive="left")]
strong_pairs = strong_pairs[strong_pairs != 1] 
print("Highly correlated pairs:")
print(strong_pairs)