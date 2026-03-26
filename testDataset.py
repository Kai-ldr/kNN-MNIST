import random
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
data = np.genfromtxt(fname='mnist_full.csv', delimiter=',')

# Remove header row
data = np.delete(data, 0, 0)

# Separate labels and images
allLabels = data[:, 0]
data = np.delete(data, 0, 1)

# How many images to show
num_images = 16  # change this to whatever you want

# Grid size (auto)
rows = int(np.sqrt(num_images))
cols = int(np.ceil(num_images / rows))

plt.figure(figsize=(8, 8))

for i in range(num_images):
    idx = random.randint(0, len(data) - 1)

    image = data[idx, :].reshape(28, 28)
    label = int(allLabels[idx])

    plt.subplot(rows, cols, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()