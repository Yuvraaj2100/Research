import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("test_folder_filtered_particle_sizes.csv")

# Extract image filenames and particle sizes
filenames = df["Image"]
particle_columns = [f"Particle {i} Size" for i in range(1, 11)]
particle_sizes = df[particle_columns]

# Plot the particle sizes for each image
for index, row in df.iterrows():
    image_filename = row["Image"]
    sizes = row[particle_columns].dropna()  # Drop NaN values (empty cells)
    
    # Check if there are valid sizes to plot
    if not sizes.empty:
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(sizes)), sizes)
        plt.xlabel("Particle Number")
        plt.ylabel("Particle Size")
        plt.title(f"Particle Sizes for {image_filename}")
        plt.xticks(range(len(sizes)), [f"Particle {i}" for i in range(1, len(sizes) + 1)])
        plt.savefig(f"particle_sizes_plot_{image_filename}.png")
        plt.close()

print("Plots saved as PNG files.")
