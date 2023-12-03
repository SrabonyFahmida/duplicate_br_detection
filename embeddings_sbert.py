import json
import numpy as np
from sentence_transformers import SentenceTransformer

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
np.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/continuum/bug_report_embeddings.npy', embeddings)
