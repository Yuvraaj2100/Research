import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Your data
frames = [
    ["730"],
    ["138.0", "273.0", "270.0", "963.0", "358.5", "531.0"],
    ["1156.0", "957.5", "151.0"],
    ["267.0", "809.5"],
    ["406.0", "1458.5", "269.0", "281.0"],
    ["321.5", "439.0", "217.5"],
    ["571.0", "129.0", "152.0", "434.0"],
    ["714.0", "500.5", "134.0"],
    ["267.0", "690.5", "275.5", "244.5"],
    ["374.5", "343.5"],
    ["241.0"],
    ["123.5"],
    ["103.5", "112.5"],
    ["722.5"],
    ["425.0"],
    ["244.5", "198.5", "387.5", "364.0", "582.0", "802.5"],
    ["1063.0", "1222.0"],
    ["471.5", "908.0", "665.0", "693.0", "667.5"],
    ["425.0", "445.5", "337.5", "315.0"],
    ["282.5", "179.0", "769.0", "425.0", "562.5", "586.0"],
    ["407.0"]
]


# Corresponding particle types for each frame
particle_types = [
    "TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "TypeF",
    "TypeG", "TypeH", "TypeI", "TypeJ", "TypeK", "TypeL",
    "TypeM", "TypeN", "TypeO", "TypeP", "TypeQ", "TypeR",
    "TypeS", "TypeT", "TypeU"  # Add labels for all n frames here
]


X = []
for frame in frames:
    particle_sizes = [float(value) for value in frame]
    mean_particle_size = np.mean(particle_sizes)
    X.append([mean_particle_size])

X = np.array(X)

# Map class labels to unique colors
color_map = {
    'TypeA': 'red',
    'TypeB': 'green',
    'TypeC': 'blue',
    'TypeD': 'orange',
    'TypeE': 'purple',
    'TypeF': 'cyan',
    'TypeG': 'magenta',
    'TypeH': 'lime',
    'TypeI': 'pink',
    'TypeJ': 'yellow',
    'TypeK': 'brown',
    'TypeL': 'teal',
    'TypeM': 'lavender',
    'TypeN': 'maroon',
    'TypeO': 'gold',
    'TypeP': 'indigo',
    'TypeQ': 'violet',
    'TypeR': 'turquoise',
    'TypeS': 'gray',
    'TypeT': 'olive',
    'TypeU': 'navy',
    # Add color mappings for all 21 types here, depending on your work 
}

colors = [color_map[label] for label in particle_types]

# Create an SVM classifier
classifier = SVC(kernel='linear', C=10)  # You can adjust kernel and C as needed

# Train the classifier
classifier.fit(X, particle_types)

# Plot the data points
plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=colors)

# Plot the decision boundaries
ax = plt.gca()
xlim = ax.get_xlim()
xx = np.linspace(xlim[0], xlim[1], 100).reshape(-1, 1)
yy = np.zeros_like(xx)
Z = classifier.decision_function(xx)
plt.plot(xx, Z, color='k', linestyle='-', linewidth=0.5)

# Set labels and title
plt.xlabel('Mean Particle Size')
plt.title('SVM Decision Boundaries')

# Show the plot
plt.show()
