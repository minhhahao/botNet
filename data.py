'''
    Processing data from http://files.pushshift.io/reddit/comments/
    Database of choice: Pandas
'''
from datetime import datetime
import pandas as pd
import os
import glob


class dataHandler():

    global db
    # timeframe for RC = [201001, 201010, 201501, 201601], using 201001
    timeframe = 0
    # Directory path
    RC_dir = ' '
    output_dir = ' '
    tmp_dir = ' '
    # important columns
    cols_to_keep = []

    def __init__(self, RC, output, tmp, important_column, tf):
        if os.path.exists(RC) and os.path.exists(output):
            self.RC_dir = RC
            self.output_dir = output
            self.tmp_dir = tmp
            self.cols_to_keep = important_column
            self.timeframe = tf
        else:
            print("Invalid directory")

    def process_data(file=None, column=[], tmp_dir=None):
        print("Begin Processing @ " + str(datetime.now()))
        if os.path.getsize(file) > 16e9:
            os.system('sh database/process_big_data.sh')
            for tmp_file in glob(tmp_dir):
                df = pd.read_json(tmp_file, orient='column')
                os.remove(tmp_file)
        else:
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
        tf = 201601
        RC = str(os.path.join(os.getcwd(), 'database', 'RC',
                              "RC_{}.json".format(tf)))
        output = str(os.path.join(os.getcwd(), 'database', 'output',
                                  'dataset.txt'))
        important_col = ["parent_id", "created_utc",
                         "subreddit_id", "score", "body", "name"]
        tmp = str(os.path.join(os.getcwd(), 'database', 'temp'))

        process_data(file=RC, column=important_col, tmp_dir=tmp)
        output_data(file=output, comment='body')
