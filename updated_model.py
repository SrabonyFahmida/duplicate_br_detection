import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize the SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read multiple JSON objects from a file
def read_json_objects(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file if line.strip()]

# Function to prepare data (concatenate title and description)
def prepare_data(bug_reports):
    prepared_texts = []
    for report in bug_reports:
        title = report.get('title', '')  # Get title if exists, else use empty string
        description = report.get('description', '')  # Get description if exists, else use empty string
        combined_text = title + ' ' + description  # Concatenate title and description
        prepared_texts.append(combined_text)
    return prepared_texts

# Function to check if two reports are duplicates based on ground truth
def load_ground_truth(file_path):
    duplicate_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            duplicate_dict[item['key']] = set(item['duplicateBugs'])
    return duplicate_dict


def are_duplicates(report1, report2, duplicate_dict):
    return report2['key'] in duplicate_dict.get(report1['key'], set())

# Load the split datasets and ground truth
# Load the split datasets
train_reports = read_json_objects('/home/aalmuhana/Desktop/replication_package/primary_80_reports.json')
test_reports = read_json_objects('/home/aalmuhana/Desktop/replication_package/secondary_20_reports.json')

duplicate_dict = load_ground_truth('/home/aalmuhana/Desktop/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/dup-bug-report-queries.json')

# Prepare and encode data
test_texts = prepare_data(test_reports)
test_embeddings = model.encode(test_texts)

# Calculate cosine similarity for test set
cosine_sim_matrix_test = cosine_similarity(test_embeddings)

# Define a threshold for considering duplicates
threshold = 0.8

# Generate true labels and predictions
true_labels = []
predictions = []

for i in range(len(test_reports)):
    for j in range(i + 1, len(test_reports)):
        if are_duplicates(test_reports[i], test_reports[j], duplicate_dict):
            true_labels.append(1)
        else:
            true_labels.append(0)
        if cosine_sim_matrix_test[i][j] > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

# Calculate evaluation metrics
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print the evaluation metrics
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
