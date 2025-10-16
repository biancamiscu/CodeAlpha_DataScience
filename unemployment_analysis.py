import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


# 1.Load the dataset
df = pd.read_csv('Unemployment.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Display the first few rows
print(df.head())

# 2.Check for missing values
print(df.isnull().sum())

# Fill missing values if necessary
df.ffill(inplace=True)

#3.Convert the 'Date' column to datetime format and extract additional time-related features
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

#4.Visualize the unemployment trends over time
# Set the style
sns.set(style="whitegrid")

# Plot the unemployment rate over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df, marker='o')
plt.title('Unemployment rate over time')
plt.xlabel('Date')
plt.ylabel('Unemployment rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#5.Analyze seasonal patterns in the unemployment data
# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Decompose the time series
result = seasonal_decompose(df['Estimated Unemployment Rate (%)'], model='additive', period=12)
result.plot()
plt.show()

#6.Identify the impact of COVID-19 on unemployment rates
# Filter data for the year 2020
df_2020 = df[df.index.year == 2020]

# Plot the unemployment rate in 2020
plt.figure(figsize=(10,6))
sns.lineplot(x=df_2020.index, y='Estimated Unemployment Rate (%)', data=df_2020, marker='o')
plt.title('Unemployment rate in 2020')
plt.xlabel('Date')
plt.ylabel('Unemployment rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
