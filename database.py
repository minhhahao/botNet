'''
    Processing data from http://files.pushshift.io/reddit/comments/
    Database of choice: Pandas
'''
from datetime import datetime
import pandas as pd
import os


class dataHandler():

    global db
    # timeframe for RC = [201001, 201010, 201501], using 201001
    timeframe = 0
    # Directory path
    RC_dir = ' '
    output_dir = ' '
    tmp_dir = os.path.join(os.getcwd(), 'database',
                           'output', 'temp', 'chunk.json')
    # important columns
    cols_to_keep = []

    def __init__(self, RC, output, important_column, tf):
        if os.path.exists(RC) and os.path.exists(output):
            self.RC_dir = RC
            self.output_dir = output
            self.cols_to_keep = important_column
            self.timeframe = tf
        else:
            print("Invalid directory")

    def checkSize(file):
        size = os.path.getsize(file)
        return size

    def process_data(file=None, column=[]):
        print("Begin Processing @ " + str(datetime.now()))
        df = pd.read_json(file, orient='columns')
        db = df.drop(df.columns.difference(column), axis=1)
        # Filter array for comment
        db = db[~db['body'].isin(['[deleted]', '[removed]'])]
        # Drop all the comment that have a score < 0
        db.drop(db[db['score'] < 3].index, inplace=True)
        # Remove special character from comments
        db = db.replace('\n', '', regex=True)
        print("Process finished @ " + str(datetime.now()))
        return db

    def output_data(file=None, comment=' '):
        with open(file, 'w+', encoding='utf-8') as output:
            print("Start writing outputs @ " + str(datetime.now()))
            database = db[comment]
            out = database.to_csv(output, sep='', index=False, header=False)
        print("Finished @ " + str(datetime.now()))
        return out

    if __name__ == '__main__':
        tf = 201501
        RC = str(os.path.join(os.getcwd(), 'database', 'RC',
                              "RC_{}.json".format(tf)))
        output = str(os.path.join(os.getcwd(), 'database', 'output',
                                  'dataset.txt'))
        important_col = ["parent_id", "created_utc",
                         "subreddit_id", "score", "body", "name"]
        process_data(file=RC, column=important_col)
        output_data(file=output, comment='body')
