import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the embeddings from the .npy file
# Replace '/path/to/your/bug_report_embeddings.npy' with the actual file path
embeddings_file = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug_report_embeddings.npy'
embeddings = np.load(embeddings_file)

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings)

# Now, cosine_sim_matrix[i][j] represents the cosine similarity between 
# the embeddings of the i-th and j-th bug reports

# Example: print cosine similarity between the first and second bug reports
print("Cosine Similarity between the first and second bug reports:", cosine_sim_matrix[0][1])

# If you want to save the cosine similarity matrix to a file:
np.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/cosine_similarity_matrix.npy', cosine_sim_matrix)
