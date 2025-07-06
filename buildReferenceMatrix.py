import numpy as np
import pandas as pd

csv_paths = [
    r"C:\Users\miosa\Documents\github repos\plastic1.csv",
    r"C:\Users\miosa\Documents\github repos\plastic2.csv",
    r"C:\Users\miosa\Documents\github repos\plastic3.csv",
    r"C:\Users\miosa\Documents\github repos\plastic4.csv",
    r"C:\Users\miosa\Documents\github repos\plastic5.csv"
]

labels = [
    "Plastic 1",
    "Plastic 2",
    "Plastic 3",
    "Plastic 4",
    "Plastic 5"
]

# Load and stack
refs = [pd.read_csv(p)['Reflectance'].values for p in csv_paths]
ref_matrix = np.vstack(refs)

# Save both matrix and labels in a single file
np.savez("reference_matrix.npz", reference_matrix=ref_matrix, labels=np.array(labels))