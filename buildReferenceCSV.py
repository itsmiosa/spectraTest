import numpy as np
from spectral import envi
import csv

# Configuration
desired_label = "PET"
x = 600  # Pixel X
y = 200  # Pixel Y
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'

# Load the image
img = envi.open(hdr_path, image=spe_path)
spectrum = np.array(img[x, y])

# Load wavelengths from metadata
wavelengths = img.metadata.get("wavelength")
wavelengths = np.array([float(w) for w in wavelengths]) if wavelengths else np.arange(len(spectrum))

# Export CSV
filename = f"{desired_label}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Wavelength (nm)', 'Reflectance'])
    for wl, ref in zip(wavelengths, spectrum):
        writer.writerow([wl, ref])

print(f"Spectrum saved as: {filename}")