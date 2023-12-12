test_fine_tune_model9.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# read JSON objects from a file
def read_json_objects(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def calculate_precision_recall_f1(predicted, actual):
    true_positives = len(set(predicted) & set(actual))
    false_positives = len(set(predicted) - set(actual))
    false_negatives = len(set(actual) - set(predicted))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# Load the fine-tuned model
model = SentenceTransformer('/home/aalmuhana/Desktop/replication_package/outputs/fine_tuned_model')

# Load test data
test_bug_reports = read_json_objects('/home/aalmuhana/Desktop/replication_package/outputs/preprocessed_80_percent_data.json')

# Generate embeddings for test data
test_texts = [report['text'] + ' ' for report in test_bug_reports]
test_embeddings = model.encode(test_texts)
print("Calculating similarity matrix for all pairs in test data...")
# Calculate similarity matrix for all pairs in test data
similarity_matrix = cosine_similarity(test_embeddings)

# Open a file to write similarity scores
print(f"Writing similarity scores to file...")
with open('/home/aalmuhana/Desktop/replication_package/outputs/similarity_between_bugs.txt', 'w') as sim_file:
    for i in range(len(test_bug_reports)):
        for j in range(i + 1, len(test_bug_reports)):
            report_id_i = test_bug_reports[i]['key']
            report_id_j = test_bug_reports[j]['key']
            sim_file.write(f"Similarity between report {report_id_i} and report {report_id_j}: {similarity_matrix[i][j]}\n")

print("Loading ground truth data for evaluation...")
# Load the ground truth data for evaluation
ground_truth_list = read_json_objects('/home/aalmuhana/Desktop/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/dup-bug-report-queries.json')
ground_truth_data = {item['key']: item['duplicateBugs'] for item in ground_truth_list}

# Open a file to write evaluation metrics
with open('/home/aalmuhana/Desktop/replication_package/outputs/evaluation_fine_tuned.txt', 'w') as eval_file:
    for report in test_bug_reports:
        report_id = report['key']
        report_embedding = model.encode(report['text'] + ' ')
        predicted_duplicates = []

     # Calculate cosine similarities and predict duplicates
        for other_report in test_bug_reports:
            other_report_id = other_report['key']
            if report_id != other_report_id:
                other_report_embedding = model.encode(other_report['text'] + ' ')
                similarity = cosine_similarity([report_embedding], [other_report_embedding])[0][0]
                if similarity >= 0.5:  # similarity threshold of 0.5 for duplicates
                    predicted_duplicates.append(other_report_id)

        actual_duplicates = ground_truth_data.get(report_id, [])
        precision, recall, f1 = calculate_precision_recall_f1(predicted_duplicates, actual_duplicates)
        eval_file.write(f"Report ID: {report_id}, Precision: {precision}, Recall: {recall}, F1: {f1}\n")
print("Script execution completed.")
