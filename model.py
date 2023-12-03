import json
import numpy as np
from sentence_transformers import SentenceTransformer


from sklearn.metrics.pairwise import cosine_similarity

# Initialize the SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read multiple JSON objects from a file
def read_json_objects(file_path):
    bug_reports = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                report = json.loads(line)
                bug_reports.append(report)
            except json.JSONDecodeError:
                continue  # Skip malformed JSON
    return bug_reports

# Load bug reports
bug_reports = read_json_objects('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug-reports.json')

# Extract 'title' and 'description' from bug reports and concatenate them
bug_report_texts = []
for report in bug_reports:
    title = report.get('title', '')  # Get title if exists, else empty string
    description = report.get('description', '')  # Get description if exists, else empty string
    combined_text = title + ' ' + description  # Concatenate title and description
    bug_report_texts.append(combined_text)

# Generate embeddings for each combined bug report text
embeddings = model.encode(bug_report_texts)

# Save embeddings to a file
#np.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug_report_embeddings.npy', embeddings)

#import numpy as np

# Load the embeddings from the .npy file
# Replace '/path/to/your/bug_report_embeddings.npy' with the actual file path

#embeddings_file = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug_report_embeddings.npy'
#embeddings = np.load(embeddings_file)

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings)

# Now, cosine_sim_matrix[i][j] represents the cosine similarity between 
# the embeddings of the i-th and j-th bug reports

# Example: print cosine similarity between the first and second bug reports

#print("Cosine Similarity between the first and second bug reports:", cosine_sim_matrix[0][1])

# If you want to save the cosine similarity matrix to a file:

#np.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/cosine_similarity_matrix.npy', cosine_sim_matrix)

#import numpy as np

threshold = 0.8  # Example threshold, needs tuning
duplicates = []

#sim_matrix = np.load('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/cosine_similarity_matrix.npy')

for i in range(len(cosine_sim_matrix)):
    for j in range(i+1, len(cosine_sim_matrix)):
        if cosine_sim_matrix[i][j] > threshold:
            duplicates.append((i, j))

# Printing the pairs of indices that are considered duplicates
for dup_pair in duplicates:
    print(f"Bug Report {dup_pair[0]} is a duplicate of Bug Report {dup_pair[1]}")
