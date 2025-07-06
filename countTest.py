import numpy as np
from spectral import envi
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# CONFIGURATION 
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'
reference_csv_paths = [
    r"C:\Users\miosa\Documents\github repos\plastic1.csv",
    r"C:\Users\miosa\Documents\github repos\plastic2.csv",
    r"C:\Users\miosa\Documents\github repos\plastic3.csv",
    r"C:\Users\miosa\Documents\github repos\plastic4.csv",
    r"C:\Users\miosa\Documents\github repos\plastic5.csv"
]
labels = ['Plastic 1', 'Plastic 2', 'Plastic 3', 'Plastic 4', 'Plastic 5']
SIMILARITY_THRESHOLD = 0.996
MIN_PIXELS = 100

# LOAD REFERENCES
reference_matrix = np.load(r"C:\Users\miosa\Documents\github repos\reference_matrix.npy")

# LOAD IMAGE
img = envi.open(hdr_path, image=spe_path)
height, width = img.shape[:2]

# INITIALIZE LABEL MAP 
label_map = np.full((height, width), "Unknown", dtype=object)

# CLASSIFY EACH PIXEL
for x in range(height):
    if x % 50 == 0:
        print(f"Classifying row {x}/{height}")
    for y in range(width):
        spectrum = img[x, y]
        similarities = cosine_similarity([spectrum], reference_matrix)
        best_match = np.argmax(similarities)
        best_score = similarities[0][best_match]

        if best_score >= SIMILARITY_THRESHOLD:
            predicted_label = labels[best_match]
        else:
            predicted_label = "Unknown"

        label_map[x, y] = predicted_label

# MAP LABELS TO INTEGERS
label_to_id = {'Unknown': 0, 'Plastic 1': 1, 'Plastic 2': 2, 'Plastic 3': 3, 'Plastic 4': 4, 'Plastic 5': 5}
id_map = np.vectorize(label_to_id.get)(label_map)

# COUNT OBJECTS PER PLASTIC TYPE 
object_counts = {}
for class_id in range(1, 6):
    binary_mask = (id_map == class_id)
    labeled_array, num_features = label(binary_mask)

    # Count pixels per object
    sizes = np.bincount(labeled_array.ravel())[1:]

    # Count how many are big enough
    valid_objects = np.sum(sizes >= MIN_PIXELS)
    object_counts[labels[class_id - 1]] = valid_objects


# DISPLAY COUNTS
print("\nPlastic Object Counts:")
for label, count in object_counts.items():
    print(f"{label}: {count}")


color_list = [
    "#000000",  # Unknown → Black
    "#1f77b4",  # Plastic 1 → Blue
    "#2ca02c",  # Plastic 2 → Green
    "#d62728",  # Plastic 3 → Red
    "#ff7f0e",  # Plastic 4 → Orange
    "#9467bd",  # Plastic 5 → Purple
]
cmap = ListedColormap(color_list)

plt.imshow(id_map, cmap=cmap, vmin=0, vmax=5)
plt.title("Classified Plastic Types")

cbar = plt.colorbar(ticks=range(6))
cbar.set_ticklabels(['Unknown', 'Plastic 1', 'Plastic 2', 'Plastic 3', 'Plastic 4', 'Plastic 5'])
plt.tight_layout()
plt.show()
