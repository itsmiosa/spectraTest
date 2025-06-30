from spectral import envi
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
x = 300  # Pixel X
y = 150  # Pixel Y
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'

# === Load the image ===
img = envi.open(hdr_path, image=spe_path)
spectrum = np.array(img[x, y])

# === Load reference spectra from your 5 plastic CSVs ===
def load_and_normalize(csv_path):
    ref = pd.read_csv(csv_path)['Reflectance'].values
    return ref / np.max(ref)

references = [
    load_and_normalize(r"C:\Users\miosa\Documents\github repos\plastic1.csv"),
    load_and_normalize(r"C:\Users\miosa\Documents\github repos\plastic2.csv"),
    load_and_normalize(r"C:\Users\miosa\Documents\github repos\plastic3.csv"),
    load_and_normalize(r"C:\Users\miosa\Documents\github repos\plastic4.csv"),
    load_and_normalize(r"C:\Users\miosa\Documents\github repos\plastic5.csv")
]
labels = ['Plastic 1', 'Plastic 2', 'Plastic 3', 'Plastic 4', 'Plastic 5']

# === Normalize the unknown spectrum ===
spectrum_norm = spectrum / np.max(spectrum)

# === Classify using cosine similarity ===
reference_matrix = np.vstack(references)
similarities = cosine_similarity([spectrum_norm], reference_matrix)
best_match = np.argmax(similarities)
predicted_label = labels[best_match]

print(f"Pixel at ({x}, {y}) classified as: {predicted_label}")

# === Load wavelengths from metadata ===
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

# === Plot the spectrum ===
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, spectrum, label='Unknown')
plt.plot(wavelengths, references[best_match], label=f'Reference: {predicted_label}')
plt.title(f"Spectrum at ({x}, {y}) - Classified as {predicted_label}")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
