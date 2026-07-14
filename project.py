import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r"Daily FCI Stock Position of the Commodity.csv"

df = pd.read_csv(file_path)

# Clean column names
df.columns = [col.strip().replace(" ", "_") for col in df.columns]

# Set seaborn theme
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Objective 1: Summary
print("Data Shape:", df.shape)
print("Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Objective 2: Histogram of Total Stock
plt.figure()
sns.histplot(df['Total_stock'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Total Stock")
plt.xlabel("Total Stock")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Objective 3: Bar chart – Top 10 Commodities by Total Stock
top_commodities = df.groupby('Commodity_name')['Total_stock'].sum().sort_values(ascending=False).head(10)
plt.figure()
sns.barplot(x=top_commodities.values, y=top_commodities.index, palette='viridis')
plt.title("Top 10 Commodities by Total Stock")
plt.xlabel("Total Stock")
plt.ylabel("Commodity")
plt.show()

# Objective 4: Pie Chart – Top 5 Districts by Total Stock
top_districts = df.groupby('District_name')['Total_stock'].sum().sort_values(ascending=False).head(5)
plt.figure()
top_districts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'), shadow=True)
plt.title(" Top 5 Districts by Total Stock")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Objective 5: Line chart – Stock trend over time
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
daily_stock = df.groupby('Date')['Total_stock'].sum()
plt.figure()
daily_stock.plot(color='mediumseagreen', marker='o')
plt.title("Daily Total Stock Over Time")
plt.xlabel("Date")
plt.ylabel("Total Stock")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 6: Correlation Heatmap (numeric columns)
num_df = df.select_dtypes(include=[np.number])
plt.figure()
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title(" Correlation Heatmap")
plt.show()

# Bonus: Scatter plot – Commodity Stock vs Total Stock
plt.figure()
sns.scatterplot(data=df, x='Commodity_Stock', y='Total_stock', hue='Commodity_name', palette='tab10', legend=False)
plt.title(" Commodity Stock vs Total Stock")
plt.xlabel("Commodity Stock")
plt.ylabel("Total Stock")
plt.grid(True)
plt.show()



