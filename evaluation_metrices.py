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
output_file_path = '/home/aalmuhana/Desktop/replication_package/outputs/RAanking_output.txt'  # Update file path


# Save the similarity rankings to the output file
with open(output_file_path, 'w') as file:
    for bug_id, rankings in bug_report_rankings.items():
        file.write(f"Rankings for Bug Report {bug_id}:\n")
        for ranked_bug_id, similarity_score in rankings.items():
            file.write(f"{ranked_bug_id}: {similarity_score}\n")
        file.write("\n")  # Add a new line for readability

print(f"Rankings saved to {output_file_path}")
