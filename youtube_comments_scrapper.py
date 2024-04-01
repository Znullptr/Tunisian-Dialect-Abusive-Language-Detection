from googleapiclient.discovery import build
import pandas as pd

# Define your API key
api_key = "Your_Api_Key"

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)


# Function to scrape comments of a video
def scrape_comments(video_id):
    comments = []
    nextPageToken = None

    while True:
        # Make request to fetch comments
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        # Extract comments and nextPageToken
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        nextPageToken = response.get('nextPageToken')

        if not nextPageToken:
            break

    return comments


# Example usage
video_id = "lsOt9bD1_IE"
comments = scrape_comments(video_id)

# Convert comments to DataFrame
df = pd.DataFrame(comments, columns=["Comment"])
df['label'] = ""
df.to_excel('data6.xlsx', index=False)
