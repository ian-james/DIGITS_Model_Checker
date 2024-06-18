# TODO: THIS ISN"T TESTED Yet
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/mnt/data/image.png'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Set the 'time' column as the index if it's not already
if 'time' in df.columns:
    df.set_index('time', inplace=True)

# Plotting each coordinate over time
plt.figure(figsize=(14, 8))

# List of columns to plot
columns_to_plot = df.columns

for column in columns_to_plot:
    plt.plot(df.index, df[column], label=column)

plt.title('Wrist and Tip Coordinates Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Save and show the plot
plt.savefig('wrist_tip_coordinates_over_time.png')
plt.show()