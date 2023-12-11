import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

# Function to read JSON objects from a file
def read_json_objects(file_path):
    reports = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                report = json.loads(line)
                reports.append(report)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON from line: {e}")
    return reports

# Function to create training examples from the bug reports
def create_training_examples(reports, num_negatives=1):
    examples = []
    for report in reports:
        
        anchor_text = report['key']

        # Creating positive 
        for duplicate_key in report.get('duplicates', []):
            duplicate_report = next((r for r in reports if r['key'] == duplicate_key), None)
            if duplicate_report:
                examples.append(InputExample(texts=[anchor_text, duplicate_report['text']], label=1))

        # Creating negative 
        for _ in range(num_negatives):
            non_duplicate_report = random.choice(reports)
            while non_duplicate_report['key'] in report.get('duplicates', []):
                non_duplicate_report = random.choice(reports)
            examples.append(InputExample(texts=[anchor_text, non_duplicate_report['key']], label=0))

    return examples

# Load the bug reports from the file
file_path = '/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/train.json'


bug_reports = read_json_objects(file_path)

# Create training examples
training_examples = create_training_examples(bug_reports, num_negatives=3)

# Initialize the SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a DataLoader for our training examples
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)

# Define a loss function. Adjust the loss function as needed.
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=4, warmup_steps=100)

model.save('/home/fhossain/replication_package/0_data/0_bug report collection/corpus and queries/accumulo/fine_tuned_model')  
