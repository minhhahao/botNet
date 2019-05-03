'''
    Processing data from Reddit Comment (http://files.pushshift.io/reddit/comments/)
    Database of choice: Pandas/numpy
'''
import pandas as pd
import json
import os.path

# Init
timeframe = '2010-01'
db_folder = os.path.join(os.getcwd(), 'database', 'RC')
data_file = os.path.join(db_folder, "RC_{}.json".format(timeframe))

# String of line to filter()
searchfor = ['\n', '\r', '[deleted]', '[removed]']

def format_data(d):
    for i in range(d.size):
        string = str(d.iloc[i]).replace('\n',' ').replace('\r',' ').replace('"',"'")
    return string

if __name__ == "__main__":
    # Processing the data
    # Open JSON data
    with open(data_file) as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))
    df = pd.DataFrame(data)
    # Remove unecessary data
    cols_to_keep = ["parent_id", "created_utc",
                    "subreddit_id", "score", "body", "name"]
    db = df.drop(df.columns.difference(cols_to_keep), axis=1)
    # Filter trash data
    db = db[~db['body'].isin(searchfor)]

    # Listing some variables
    body = db['body']
    created_utc = db['created_utc']
    id = db['name']
    parent_id = db['parent_id']
    score = db['score']
    subreddit = db['subreddit_id']

    body
