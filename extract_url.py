import json
import re
import requests

# Path to the JSON file uploaded by the user
#file_path = 'path_to_your_json_file.json'  # Replace with your actual file path
path_to_json = '/Users/spl/Documents/Archive/final_json/zhanghai/'
#file_path = '/Users/spl/Documents/Archive/final_json/zhanghai/MaterialFiles_issue_575.json'  # Replace with your actual file path

# Function to extract image and video URLs from a JSON file
def extract_media_urls(json_file_path):
    # Regular expressions for finding image and video URLs
    image_url_pattern = re.compile(r'http[s]?://\S+\.(?:jpg|jpeg|png|gif)', re.IGNORECASE)
    video_url_pattern = re.compile(r'http[s]?://\S+\.(?:mp4|avi|mov|wmv|flv)', re.IGNORECASE)

    # Lists to store extracted URLs
    image_urls = []
    video_urls = []

    try:
        # Open and load the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            
            # Convert the JSON object to a string
            json_string = json.dumps(data)
            
            # Find all matches of the URL patterns in the string
            image_urls = image_url_pattern.findall(json_string)
            video_urls = video_url_pattern.findall(json_string)
            
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

    return image_urls, video_urls

# Function to check if a list of URLs is valid
def check_urls_validity(urls):
    results = {}
    
    for url in urls:
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            # If the response status code is 200-299, it is successful
            results[url] = response.ok
        except requests.RequestException as e:
            # If there is any requests-related issue, mark the URL as invalid
            results[url] = False
    return results

for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  with open(path_to_json + file_name) as json_file:
    data = json.load(json_file)
    # Call the function and assign the results
    image_urls, video_urls = extract_media_urls(json_file)
    print(file_name)
    validity_results = check_urls_validity(image_urls)
    print('Valid Image URLs:', *validity_results, sep='\n')
    #print(validity_results)

    validity_results = check_urls_validity(video_urls)
    print('Valid Video URLs:', *validity_results, sep='\n')

# Output the results
#print('Image URLs:', *image_urls, sep='\n')
#print(*lst, sep='\n')
#print('Video URLs:', *video_urls, sep='\n')
