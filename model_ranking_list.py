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
bug_reports = read_json_objects('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/test.json')

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
sim_matrix = cosine_similarity(embeddings)

# Now, cosine_sim_matrix[i][j] represents the cosine similarity between 
# the embeddings of the i-th and j-th bug reports

# Example: print cosine similarity between the first and second bug reports
print("Cosine Similarity between the first and second bug reports:", cosine_sim_matrix[0][1])

# If you want to save the cosine similarity matrix to a file:

#np.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/cosine_similarity_matrix.npy', cosine_sim_matrix)

#import numpy as np

'''import json

#this is list the ID
bug_reports_file = '/home/aalmuhana/Desktop/replication_package/outputs/primary_80_reports.json'


bug_report_ids = []

# Reading and parsing bug reports from the file
with open(bug_reports_file, 'r') as file:
    for line in file:
        try:
            # Attempt to parse each line as a separate JSON object
            report = json.loads(line)
            # Extract the 'key' field and add it to the list
            bug_report_ids.append(report["key"])
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from line: {e}")

print(bug_report_ids)
'''
import json
import numpy as np

# File path for your bug reports file
bug_reports_file = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/test.json'


bug_report_ids = []

# Reading and parsing bug reports from the file
with open(bug_reports_file, 'r') as file:
    for line in file:
        try:
            # Attempt to parse each line as a separate JSON object
            report = json.loads(line)
            # Extract the 'key' field and add it to the list
            bug_report_ids.append(report["key"])
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from line: {e}")

# Assuming 'sim_matrix' is your cosine similarity matrix
# Replace the following line with your actual cosine similarity matrix
sim_matrix = np.array([...])  # Replace with your actual matrix

# Create a dictionary for each bug report
bug_report_rankings = {}

for i, bug_id in enumerate(bug_report_ids):
    similarities = sim_matrix[i]
    rankings = {bug_report_ids[j]: similarities[j] for j in range(len(bug_report_ids)) if i != j}
    # Sorting the rankings based on similarity score, highest similarity first
    sorted_rankings = dict(sorted(rankings.items(), key=lambda item: item[1], reverse=True))
    bug_report_rankings[bug_id] = sorted_rankings

# 'bug_report_rankings' now contains a dictionary for each bug report,
# where each key is another bug report's ID and the value is their similarity score

# Example: Print the rankings for the first bug report
print(bug_report_rankings[bug_report_ids[0]])
