from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarities
# This will create a matrix where each element [i, j] represents the cosine similarity 
# between the embeddings of bug report i and bug report j
cosine_similarities = cosine_similarity(embeddings)


# Example: Get the most similar bug reports to the first report
first_report_similarities = cosine_similarities[0]

# Sort the reports based on similarity (excluding the first report itself)
most_similar_indices = first_report_similarities.argsort()[::-1][1:]

# Print the indices of the most similar reports
print("Most similar reports to the first report:", most_similar_indices)
