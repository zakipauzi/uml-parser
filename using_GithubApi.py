import pandas as pd
import requests
import os

# Load the CSV file
file_path = '/Users/elifnazduman/Downloads/Project_FileTypes_V2.0.csv'
df = pd.read_csv(file_path)

# Store your GitHub token
github_token = ''


# Function to get the last updated time of a GitHub repository
def get_repo_last_updated(repo_url):
    try:
        # Extract owner and repo name from the URL
        parts = repo_url.split('/')
        owner, repo = parts[-2], parts[-1]

        # GitHub API URL
        api_url = f'https://api.github.com/repos/{owner}/{repo}'

        # Set up the headers with the token
        headers = {
            'Authorization': f'token {github_token}'
        }

        # Make a request to the GitHub API
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the JSON response
        repo_info = response.json()
        return repo_info['updated_at']
    except Exception as e:
        print(f"Error fetching data for {repo_url}: {e}")
        return None


# Apply the function to each GitHub link in the DataFrame
df['Last_Updated'] = df['GitHub URL'].apply(get_repo_last_updated)

# Save the updated DataFrame back to a CSV file
updated_file_path = '/mnt/data/Updated_Project_FileTypes_V2.0.csv'
df.to_csv(updated_file_path, index=False)

print(f"Updated CSV file saved to {updated_file_path}")
