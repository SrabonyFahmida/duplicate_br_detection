import numpy as np

# Assuming 'sim_matrix' is your cosine similarity matrix
# and 'bug_report_ids' is a list of your bug report IDs corresponding to the rows and columns of the matrix

sim_matrix = np.array([...])  # your cosine similarity matrix
bug_report_ids = [...]  # your list of bug report IDs

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
