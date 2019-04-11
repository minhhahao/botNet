import sqlite3
import json
import datetime

tf = '2005-12'
trans = []

connection = sqlite3.connect('{}.db'.format(tf))

c = connection.cursor()


def create_table():
    c.execute('''CREATE TABLE IF NOT EXIST parent_reply(
        parent_id TEXT PRIMARY KEY,
        comment_id TEXT UNIQUE,
        parent TEXT,
        comment TEXT,
        subreddit TEXT,
        unix INT,
        score INT)'''
              )


if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0

with open('/home/aazasdass/Documents/Coding/tob/RC/RC_{}'.format(tf.split('-')[0], tf), buffering=1000) as f:
    for r in f:
        row_counter += 1
        row = json.loads(row)
        parent_id = row['parent_id']
