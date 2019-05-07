'''
    Processing data from http://files.pushshift.io/reddit/comments/
    Database of choice: Pandas
'''
from datetime import datetime
import pandas as pd
import os
import fnmatch


class Processor():
    '''
        Usage: Do all the dirty jobs
        Since json cannot load a big JSON file into RAM, split the file into smaller chunk using big_data.sh
    '''
    # Directory Settings
    reddit_comment_dir = ' '
    temporary_json_dir = ' '
    output_dir = ' '

    # Array of String indicates location of tmp file
    tmp_arr = []

    # timeframe for different RC : 201010,201001, 201501,201601. [default]: 201601 (processing big data)
    timeframe = 201601

    def __init__(self, database, tf=None):
        if os.path.exists(database):
            # Set string paths
            self.reddit_comment_dir = database + "/RC/"
            self.temporary_json_dir = database + "/temp/"
            self.output_dir = database + "/output/"
            self.timeframe = tf

            # Walking tmp_array for big data processing
            self.tmp_arr = list(pos_json for pos_json in os.listdir(
                self.temporary_json_dir) if pos_json.endswith('.json'))
        else:
            print("Invalid directory! Terminated @ " + str(datetime.now()))

    def checkSize(self, tf=None):
        # Checking the size of the current working file that match with timeframe, whether it exceeds RAM or not
        current_working_file = str(file for file in os.listdir(
            self.reddit_comment_dir) if fnmatch.fnmatch(file, 'RC_{}.json'.format(tf)))
        if os.path.getsize(current_working_file) > 5e9:
            os.system('sh database/big_data.sh')
        else:
            return True

    def process(self, ):


class dataHandler():

    # timeframe for RC = [201001, 201010, 201501, 201601], using 201001
    timeframe = 0
    # Directory path
    RC_dir = ' '
    output_dir = ' '
    tmp = ' '
    # important columns
    cols_to_keep = []
    # Database using pandas
    db = pd.DataFrame()

    def __init__(self, tf, RC, output, tmp, important_column, db):
        if os.path.exists(RC) and os.path.exists(output):
            self.timeframe = tf
            self.RC_dir = RC
            self.output_dir = output
            self.tmp = tmp
            self.cols_to_keep = important_column
            self.db = db
        else:
            print("Invalid directory")

    def process_data(db=None, file=None, tmp_path=None, tmp_dict=[]):
        # Input the data
        print("Begin Processing @ " + str(datetime.now()))
        db = pd.DataFrame()
        if os.path.getsize(file) > 5e9:
            os.system('sh database/process_big_data.sh')
            for index, js in enumerate(tmp_dict):
                with open(str(os.path.join(tmp_path, js)), encoding='utf-8') as json_file:
                    js_text = json.load(json_file, encoding='utf-8')
                    parent_id = js_text['parent_id']
                    created_utc = js_text['created_utc']
                    subreddit_id = js_text['subreddit_id']
                    score = js_text['score']
                    body = js_text['body']
                    name = js_text['name']
                    db.loc[index] = [parent_id, created_utc,
                                     subreddit_id, score, body, name]
        else:
            with open(file, 'r') as json_file:
                for r in json_file:
                    row = json.loads(r)
                    parent_id = row['parent_id']
                    created_utc = row['created_utc']
                    subreddit_id = row['subreddit_id']
                    score = row['score']
                    body = row['body']
                    name = row['name']
                db = pd.DataFrame(
                    columns=[parent_id, created_utc, subreddit_id, score, body, name])
        # A way to remove unecessary data from a directory
        # db = db.drop(db.columns.difference(important_col), axis=1)
        # Filter array for comment
        db = db[~db[body].isin(['[deleted]', '[removed]'])]
        # Drop all the comment that have a score < 0
        db.drop(db[db[score] < 3].index, inplace=True)
        # Remove special character from comments
        db = db.replace('\n', '', regex=True)
        print("Process finished @ " + str(datetime.now()))
        return db

    def output_data(db=None, file=None, comment=' '):
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
        tmp = str(os.path.join(os.getcwd(), 'database',
                               'temp'))
        tmp_json = list(pos_json for pos_json in os.listdir(
            tmp) if pos_json.endswith('.json'))

        process_data(file=RC, tmp_path=tmp, tmp_dict=tmp_json)
        output_data(file=output, comment='body')
