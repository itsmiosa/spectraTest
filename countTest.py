import numpy as np
from spectral import envi
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import label as nd_label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
import csv

# CONFIGURATION 
hdr_path = r'C:\Users\miosa\Documents\spectralData\spectraData.hdr'
spe_path = r'C:\Users\miosa\Documents\spectralData\spectraData.spe'
SIMILARITY_THRESHOLD = 0.996
MIN_PIXELS = 50

# LOAD REFERENCE MATRIX
data = np.load(r"C:\Users\miosa\Documents\github repos\reference_matrix.npz")
reference_matrix = data["reference_matrix"]
labels = data["labels"]

# LOAD IMAGE
img = envi.open(hdr_path, image=spe_path)
height, width = img.shape[:2]

# INITIALIZE LABEL MAP 
label_map = np.full((height, width), "Unknown", dtype=object)

# CLASSIFY EACH PIXEL
similarity_map = np.zeros((height, width), dtype=np.float32)
for x in range(height):
    if x % 50 == 0:
        print(f"Classifying row {x}/{height}")
    for y in range(width):
        spectrum = img[x, y]
        similarities = cosine_similarity([spectrum], reference_matrix)
        best_match = np.argmax(similarities)
        best_score = similarities[0][best_match]
        similarity_map[x, y] = best_score

        if best_score >= SIMILARITY_THRESHOLD:
            predicted_label = labels[best_match]
        else:
            predicted_label = "Unknown"
            
        label_map[x, y] = predicted_label

# MAP LABELS TO INTEGERS
label_to_id = {'Unknown': 0}
for i, label in enumerate(labels, start=1):
    label_to_id[label] = i
id_map = np.vectorize(label_to_id.get)(label_map)

# COUNT OBJECTS PER PLASTIC TYPE
summary_counts = defaultdict(int)
total_objects = 0
object_num = 1
csv_rows = [("Plastic Number", "Plastic Type", "Area (pixels)", "Avg Similarity")]

print("\nDetected Plastic Objects:")

for class_id in range(1, len(labels) + 1):
    class_label = labels[class_id - 1]
    binary_mask = (id_map == class_id)
    labeled_array, num_features = nd_label(binary_mask)
    sizes = np.bincount(labeled_array.ravel())[1:]

    for obj_label in range(1, num_features + 1):
        mask = (labeled_array == obj_label)
        size = np.sum(mask)
        if size >= MIN_PIXELS:
            avg_similarity = np.mean(similarity_map[mask])
            print(f"Plastic #{object_num} ({class_label}) - size: {size} pixels - avg similarity: {avg_similarity:.4f}")
            csv_rows.append((object_num, class_label, size, round(avg_similarity, 4)))
            summary_counts[class_label] += 1
            total_objects += 1
            object_num += 1

# WRITE CSV
csv_path = r"C:\Users\miosa\Documents\spectralData\plasticCountResults.csv"
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("\nSummary per Plastic Type:")
for label in labels:
    print(f"{label}: {summary_counts[label]} objects")

print(f"\nTotal: {total_objects} plastic objects detected")

# VISUALIZE THE CLASSIFICATION
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