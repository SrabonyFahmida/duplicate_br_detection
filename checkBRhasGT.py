import json
import pandas as pd

def load_json_file_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def check_ground_truth(bug_reports, ground_truths):
    results = []
    for report in bug_reports:
        key = report['key']
        match = any(gt['key'] == key for gt in ground_truths)
        results.append({'Bug Report': key, 'Match Found': match})
    return results


base_path = 'replication_package/0_data/0_bug report collection/corpus and queries'
output_file = 'replication_package/outputs/Check_GT_of_BR_file.xlsx' 


projects = ['accumulo', 'ambari', 'amq', 'cassandra', 'cb', 'continuum', 'drill', 'eclipse', 'groovy', 'hadoop', 'hbase', 'hive', 'mng', 'mozilla', 'myfaces', 'openoffice', 'pdfbox', 'spark', 'wicket', 'ww']


writer = pd.ExcelWriter(output_file, engine='openpyxl')


for project in projects:
    bug_reports_file = f'bug-reports.json'
    ground_truths_file = f'dup-bug-report-queries.json'
    
    bug_reports = load_json_file_lines(bug_reports_file)
    ground_truths = load_json_file_lines(ground_truths_file)
    
    results = check_ground_truth(bug_reports, ground_truths)

    df = pd.DataFrame(results)
    
    df.to_excel(writer, sheet_name=project, index=False)

writer.close()
