from spectral import envi
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
x = 500  # Pixel X
y = 200  # Pixel Y
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'
SIMILARITY_THRESHOLD = 0.99

# Load the image
img = envi.open(hdr_path, image=spe_path)
spectrum = np.array(img[x, y])

# Load reference matrix and labels from .npz file
data = np.load(r"C:\Users\miosa\Documents\github repos\reference_matrix.npz")
reference_matrix = data["reference_matrix"]
labels = data["labels"]

# Classify using cosine similarity
similarities = cosine_similarity([spectrum], reference_matrix)
best_match = np.argmax(similarities)
best_score = similarities[0][best_match]

if best_score >= SIMILARITY_THRESHOLD:
    predicted_label = labels[best_match]
else:
    predicted_label = "Unknown"

# Output the classification result
print(f"Pixel at ({x}, {y}) classified as: {predicted_label} with a similarity score of {best_score:.5f}")

# Load wavelengths from metadata
wavelengths = img.metadata.get("wavelength")
wavelengths = np.array([float(w) for w in wavelengths]) if wavelengths else np.arange(len(spectrum))

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, spectrum, label='Unknown')
plt.plot(wavelengths, reference_matrix[best_match], label=f'Reference: {predicted_label}')
plt.title(f"Spectrum at ({x}, {y}) - Classified as {predicted_label}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
