from sklearn.datasets import fetch_openml
import pandas as pd

# Download MNIST from OpenML
mnist = fetch_openml("mnist_784", version=1, as_frame=True)

# Convert labels to integers
X = mnist.data
y = mnist.target.astype(int)

pixel_cols = []
for row in range(28):
    for col in range(28):
        pixel_cols.append(f"{row}x{col}")

# Assign new column names
X.columns = pixel_cols

# Combine label + pixels
df = pd.concat([y.rename("label"), X], axis=1)

# Save as CSV
df.to_csv("mnist_full.csv", index=False, header=True)

print("Saved mnist_full.csv")