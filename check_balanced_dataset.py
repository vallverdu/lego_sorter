import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./dataset.csv')

# Count the occurrences of each brick type
brick_type_counts = data['brick_type'].value_counts()

print(brick_type_counts)

# Plotting the distribution
plt.figure(figsize=(12, 8))
brick_type_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Brick Types')
plt.xlabel('Brick Type')
plt.ylabel('Number of Pieces')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()