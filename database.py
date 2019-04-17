import sqlite3
import json
from datetime import datetime

tf = '2010-10'
sql_transaction = []
connection = sqlite3.connect('database/{}.db'.format(tf))
c = connection.cursor()


def create_table():
    c.execute(''' CREATE TABLE IF NOT EXISTS parent_reply(
    parent_id TEXT PRIMARY KEY,
    comment_id TEXT UNIQUE,
    parent TEXT,
    comment TEXT,
    subreddit TEXT,
    unix INT,
    score INT)
    ''')


def formatted(data):
    data = data.replace('\n', ' ').replace(
        '\r', ' ').replace('"', "'")
    return data


def fparent(pid):
    sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' \
        LIMIT 1".format(pid)
    c.execute(sql)
    result = c.fetchone()
    if result is not None:
        return result[0]
    else:
        return False


def fscore(pid):
    sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' \
        LIMIT 1".format(pid)
    c.execute(sql)
    result = c.fetchone()
    if result is not None:
        return result[0]
    else:
        return False


def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True


def insrcomment(cid, pid, parent, comment, subreddit, time, score):
    sql = '''
        UPDATE parent_reply SET
        parent_id = ?,
        comment_id = ?,
        parent = ?,
        comment = ?,
        subreddit = ?,
        unix = ?,
        score = ?
        WHERE parent_id =?;'''.format(
        pid, cid, parent, comment, subreddit, int(time), score, pid)
    builder(sql)


def inspar(cid, pid, parent, comment, subreddit, time, score):
    sql = '''
        INSERT INTO parent_reply
        (parent_id, comment_id, parent, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}","{}",{},{});'''.format(
        pid, cid, parent, comment, subreddit, int(time), score)
    builder(sql)


def insnopar(cid, pid, comment, subreddit, time, score):
    sql = '''INSERT INTO parent_reply
        (parent_id, comment_id, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}",{},{});'''.format(
        pid, cid, comment, subreddit, int(time), score)
    builder(sql)


def builder(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0

# Windows path : C:\\Users\\Aaron Pham\\Documents\\Coding\\CS\\repo\\tob\\RC\\RC_2010-10
# Needed to find a fix for this incompablility

with open("/home/aazasdass/Documents/Coding/tob/RC/RC_{}".format(
        tf), buffering=1000) as f:
    for r in f:
        row_counter += 1
        row = json.loads(r)
        parent_id = row['parent_id']
        body = formatted(row['body'])
        created_utc = row['created_utc']
        score = row['score']
        subreddit = row['subreddit']
        comment_id = row['name']

        parent_data = fparent(parent_id)
        if score >= 2:
            if acceptable(body):
                existing_comment_score = fscore(parent_id)
                if existing_comment_score:
                    if score > existing_comment_score:
                        insrcomment(comment_id, parent_id, parent_data,
                                    body, subreddit, created_utc, score)
                else:
                    if parent_data:
                        inspar(comment_id, parent_id, parent_data,
                               body, subreddit, created_utc, score)
                        paired_rows += 1
                    else:
                        insnopar(comment_id, parent_id, body,
                                 subreddit, created_utc, score)

        # Logging purposes
        if row_counter % 100000 == 0:
            print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(
                row_counter, paired_rows, str(datetime.now())))
