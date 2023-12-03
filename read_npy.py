import numpy as np

# Replace 'path/to/your/file.npy' with the actual file path
#file_path = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug_report_embeddings.npy'
file_path = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/cosine_similarity_matrix.npy'

# Load the array from the .npy file
array = np.load(file_path)

# Now you can use 'array' as a regular NumPy array
print(array)
