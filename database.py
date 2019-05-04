'''
    Processing data from Reddit Comment (http://files.pushshift.io/reddit/comments/)
    Database of choice: Pandas
'''
import pandas as pd
import json
import os.path

# Init
# '2010-01', '2010-10', '2015-01'
global timeframe
timeframe = '2015-01'
data_dir = os.path.join(os.getcwd(), 'database', 'RC',
                        "RC_{}.json".format(timeframe))
output_dir = os.path.join(os.getcwd(), 'database', 'output',
                          'RC_{}.txt'.format(timeframe))
output_file = open(output_dir, 'w+', encoding="utf-8")


def create_table():
    global db
    # Processing the data
    with open(data_dir) as json_file:
        data = json_file.readlines()
        data = list(map(json.loads, data))
    df = pd.DataFrame(data)

    # Filter out necessary data array
    cols_to_keep = ["parent_id", "created_utc",
                    "subreddit_id", "score", "body", "name"]
    db = df.drop(df.columns.difference(cols_to_keep), axis=1)
    # Filter array for comment
    db = db[~db['body'].isin(['[deleted]', '[removed]'])]
    # Drop all the comment that have a score < 0
    db.drop(db[db['score'] < 3].index, inplace=True)
    # Remove special character from comments
    db = db.replace('\n', '', regex=True)


def output():
    database = db['body']
    database.to_csv(output_file, sep=' ', index=False, header=False)


if __name__ == "__main__":
    create_table()
    output()
