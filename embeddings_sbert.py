#   !pip install sentence-transformers


import json
import numpy as np

# Load the JSON file
with open('/mnt/data/bug-reports.json', 'r') as file:
    bug_reports = json.load(file)
    
# Extract the descriptions (or the relevant field) from the bug reports
bug_report_texts = [report['description'] for report in bug_reports]

from sentence_transformers import SentenceTransformer

# Initialize the SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each bug report
embeddings = model.encode(bug_report_texts)



# Save embeddings to a file
np.save('/mnt/data/bug_report_embeddings.npy', embeddings)
