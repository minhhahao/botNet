'''
    Processing data from http://files.pushshift.io/reddit/comments/
    Database of choice: Pandas / MongoDB
'''
import pandas as pd
import os
from datetime import datetime
from pymongo import MongoClient


class handler():
    # path
    output_dir = ''

    # important column
    cols_to_keep = ["parent_id", "created_utc",
                    "subreddit_id", "score", "body", "name"]

    # timeframe for RC
    # timeframe = ['2010-01', '2010-10', '2015-01']
    timeframe = '2010-01'

    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(), 'database', 'output')

    def read_mongo(database, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
        '''
        Usage: Read query from MongoDB then put into Pandas DataFrame
        '''
        # Connect to MongoDB
        conn = MongoClient(host=host, port=port, username=username,
                           password=password, authSource="Admin")
        db = conn['database']

        # Make query to MongoDB
        cur = db.collection.find(query)

        # Expand the cursor and construct pd DataFrame
        df = pd.DataFrame(list(cur))

        # Delete _id that exists for each document in MongoDB
        if no_id:
            del df['id']

        return df

    def process_data(self):
        db = self.read_mongo()
        column = self.cols_to_keep
        # Filter out necessary data array
        df = db.drop(db.columns.difference(column), axis=1)
        # Filter array for comment
        df = df[~df['body'].isin(['[deleted]', '[removed]'])]
        # Drop all the comment that have a score < 0
        df.drop(df[df['score'] < 3].index, inplace=True)
        # Remove special character from comments
        df = df.replace('\n', '', regex=True)
        print("Finished creating DB @ " + str(datetime.now()))
        return df

    def training_data(self, output_file, comments=['body']):
        database = self.read_mongo()
        bd = database[comments]
        if not os.path.exists(output_file):
            print("Writing output file @ " + str(datetime.now()))
            with open(output_file, 'a', encoding="utf-8") as output:
                bd.to_csv(output, sep=' ', index=False, header=False)

    if __name__ == '__main__':
        read_mongo('RC', 2010 - 10, {}, host='localhost', port= 27017,
                   username='root', password='toor', no_id=True)
        process_data()
        training_data(os.path.join(
            os.getcwd(), 'database', 'output', 'db.txt'))
