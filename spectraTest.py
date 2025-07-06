from spectral import envi
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
x = 350  # Pixel X
y = 270  # Pixel Y
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'
SIMILARITY_THRESHOLD = 0.99
#labels = ['Plastic 1', 'Plastic 2', 'Plastic 3', 'Plastic 4', 'Plastic 5']

# Load the image
img = envi.open(hdr_path, image=spe_path)
spectrum = np.array(img[x, y])

# Load reference matrix and labels from .npz file
data = np.load(r"C:\Users\miosa\Documents\github repos\reference_matrix.npz", allow_pickle=True)
reference_matrix = data["ref_matrix"]
labels = data["labels"]
print("Loaded reference matrix with shape:", )

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

'''
# === Export spectrum to CSV with timestamp ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"spectrum_{x}_{y}_{timestamp}_{predicted_label}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Wavelength (nm)', 'Reflectance'])
    for wl, ref in zip(wavelengths, spectrum):
        writer.writerow([wl, ref])

print(f"Spectrum saved as: {filename}")
'''

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
