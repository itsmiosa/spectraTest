from spectral import envi
import matplotlib.pyplot as plt
import numpy as np
import csv

# Choose pixel to visualize
x = 300
y = 150

# Load metadata and image
img = envi.open(r'C:\Users\miosa\Documents\spectralData\spectraData.hdr', image=r'C:\Users\miosa\Documents\spectralData\spectraData.spe')

# Get full spectrum at pixel
spectrum = np.array(img[x, y])

# Get full wavelength list
wavelengths = img.metadata.get("wavelength")
wavelengths = np.array([float(w) for w in wavelengths]) if wavelengths else np.arange(len(spectrum))

# Plot full spectrum
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, spectrum)
plt.title(f"Full Spectrum at ({x}, {y})")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Export full spectrum to CSV
output_filename = f"spectrum_{x}_{y}_full.csv"
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Wavelength (nm)', 'Reflectance'])
    for wl, ref in zip(wavelengths, spectrum):
        writer.writerow([wl, ref])

print(f"Saved full spectrum to: {output_filename}")


#test