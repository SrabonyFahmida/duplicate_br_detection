import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# read JSON objects from a file
def read_json_objects(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

"""def calculate_precision_recall_f1(predicted, actual):
    true_positives = len(set(predicted) & set(actual))
    false_positives = len(set(predicted) - set(actual))
    false_negatives = len(set(actual) - set(predicted))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1 """

# Load the fine-tuned model
model = SentenceTransformer('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/fine_tuned_model')

# Load test data
bug_reports = read_json_objects('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/preprocessed_train.json')

# Generate embeddings for test data
test_texts = [report['text'] + ' ' for report in bug_reports]

bug_report_ids = []
bug_report_texts = []

# Parse each JSON string and extract 'key' and 'text'
for json_str in bug_reports:
    try:
        report = json.loads(json_str)
        bug_report_ids.append(report["key"])
        bug_report_texts.append(report["text"])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e} in string: {json_str}")
test_embeddings = model.encode(test_texts)
print("Calculating similarity matrix for all pairs in test data...")
# Calculate similarity matrix for all pairs in test data
similarity_matrix = cosine_similarity(test_embeddings)

# Create a dictionary for each bug report
bug_report_rankings = {
    bug_id: dict(sorted({bug_report_ids[j]: similarities[j] for j in range(len(bug_report_ids)) if i != j}.items(), key=lambda item: item[1], reverse=True))
    for i, bug_id in enumerate(bug_report_ids)
    for similarities in [sim_matrix[i]]
}

# Open a file to write similarity scores
print(f"Writing similarity scores to file...")
with open('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/similarity_between_bugs.txt', 'w') as sim_file:
    for i in range(len(bug_reports)):
        for j in range(i + 1, len(bug_reports)):
            report_id_i = bug_reports[i]['key']
            report_id_j = bug_reports[j]['key']
            sim_file.write(f"Similarity between report {report_id_i} and report {report_id_j}: {similarity_matrix[i][j]}\n")

print("Loading ground truth data for evaluation...")
# Load the ground truth data for evaluation
ground_truth_list = read_json_objects('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/GT_for_placeholders.json')
ground_truth_data = {item['key']: item['duplicateBugs'] for item in ground_truth_list}

# Open a file to write evaluation metrics
"""with open('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/evaluation_fine_tuned.txt', 'w') as eval_file:
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
print("Script execution completed.")"""

# Define evaluation metric functions

def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    return np.mean(r)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

'''def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max'''
    
def ndcg_at_k(r, k, method=0):
    ndcg_scores = []
    for rel_list in r:
        dcg_max = dcg_at_k(sorted(rel_list, reverse=True), k, method)
        if not dcg_max:
            ndcg_scores.append(0.)
        else:
            dcg_score = dcg_at_k(rel_list, k, method)
            ndcg_scores.append(dcg_score / dcg_max)
    return np.mean(ndcg_scores)


# Prepare the relevance lists for each bug report
k = 5  # Adjust k as needed
relevance_lists = []
for bug_id in bug_report_ids:
    rankings = bug_report_rankings.get(bug_id, {})
    top_k_ranked_ids = list(rankings.keys())[:k]
    relevance_list = [1 if id in ground_truth_data.get(bug_id, []) else 0 for id in top_k_ranked_ids]
    relevance_lists.append(relevance_list)

    # Debugging: Check the length of each sublist
    if len(relevance_list) != k:
        print(f"Relevance list for {bug_id} has a length of {len(relevance_list)}, expected {k}")

# Debugging: Print the relevance lists
print("Relevance Lists:", relevance_lists)
#for r_list in relevance_lists:
  #  print("List shape:", np.shape(r_list))

# Calculate evaluation metrics
try:
    mrr = mean_reciprocal_rank(relevance_lists)
    print(f"Mean Reciprocal Rank: {mrr}")
    map_score = mean_average_precision(relevance_lists)
    ndcg = ndcg_at_k(relevance_lists, k)
    
    #print(f"Mean Reciprocal Rank: {mrr}")
    print(f"Mean Average Precision: {map_score}")
    print(f"NDCG@K: {ndcg}")
except Exception as e:
    print(f"Error in calculating metrics: {e}")

# Specify your desired output file path
output_file_path = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/fine_tuned_Ranking_output.txt'  # Update file path


# Save the similarity rankings to the output file
with open(output_file_path, 'w') as file:
    for bug_id, rankings in bug_report_rankings.items():
        file.write(f"Rankings for Bug Report {bug_id}:\n")
        for ranked_bug_id, similarity_score in rankings.items():
            file.write(f"{ranked_bug_id}: {similarity_score}\n")
        file.write("\n")  # Add a new line for readability

print(f"Rankings saved to {output_file_path}")
