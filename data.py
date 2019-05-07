'''
    Processing data from http://files.pushshift.io/reddit/comments/
    Database of choice: Pandas
'''
from datetime import datetime
import pandas as pd
import os
import fnmatch
import json


class Processor():
    '''
        Usage: Do all the dirty jobs
        Since json cannot load a big JSON file into RAM, split the file into
        smaller chunk using big_data.sh
    '''
    # General database
    database = []

    # Directory Settings
    reddit_comment_dir = ' '
    temporary_json_dir = ' '
    output_dir = ' '

    # Array of String indicates location of tmp file
    tmp_arr = []

    # timeframe : 201010,201001, 201501,201601. [default] : 201601
    timeframe = 0

    # important columns
    important_col = ["parent_id", "created_utc",
                     "subreddit_id", "score", "body", "name"]

    def __init__(self, database_dir=None, tf=None):
        if os.path.exists(database_dir):
            # Set string paths
            self.database = pd.DataFrame()
            self.reddit_comment_dir = database_dir + "/RC/"
            self.temporary_json_dir = database_dir + "/temp/"
            self.output_dir = database_dir + "/output/"
            self.timeframe = tf

            # Walking tmp_array for big data processing
            self.tmp_arr = list(pos_json for pos_json in os.listdir(
                self.temporary_json_dir) if pos_json.endswith('.json'))
        else:
            print("Invalid directory! Terminated @ " + str(datetime.now()))

    def checkSize(self, tf=None):
        '''
        Usage: Checking the size of the current working file
        that match with timeframe, whether it exceeds RAM or not
        '''
        current_working_file = str(file for file in os.listdir(
            self.reddit_comment_dir) if fnmatch.fnmatch(file, 'RC_{}.json'.format(tf)))
        if os.path.getsize(current_working_file) > 5e9:
            os.system('sh database/big_data.sh')
        else:
            return True

    def createDataFrame(self, file=None):
        print("Begin Processing @ " + str(datetime.now()))
        with open(file, 'r', encoding='utf-8', buffering=20000000) as js:
            json_file = json.load(js)
            parent_id = json_file['parent_id']
            created_utc = json_file['created_utc']
            subreddit_id = json_file['subreddit_id']
            score = json_file['score']
            body = json_file['body']
            name = json_file['name']
        self.db = pd.DataFrame(
            columns=[parent_id, created_utc, subreddit_id, score, body, name])
        return self.db

    def process(self):
        # database
        self.db = self.db[~self.db['body'].isin(['[deleted]', '[removed]'])]
        self.db.drop(self.db[self.db['score'] < 3].index, inplace=True)
        self.db = self.db.replace('\n', '', regex=True)
        print("Process finished @ " + str(datetime.now()))
        return self.db

    def output(self, output_file=None):
        with open(output_file, 'w+', encoding='utf-8') as output:
            print("Start writing outputs @ " + str(datetime.now()))
            database = self.db['body']
            out = database.to_csv(output, sep='', index=False, header=False)
        print("Finished @ " + str(datetime.now()))
        return out
