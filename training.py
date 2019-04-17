import sqlite3
import pandas as pd

tf = '2010-10'

connection = sqlite3.connect('database/{}.db'.format(tf))
c = connection.cursor()
limit = 1000  # limit to pull data into pandas
last_unix = 0  # help buffer through data
cur_length = limit  # keep track of when the test is finish
counter = 0
test_done = False

while cur_length == limit:

    df = pd.read_sql(
        '''SELECT * FROM parent_reply WHERE unix > {} AND parent
             NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}'''.format(
            last_unix, limit), connection)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)

    if not test_done:
        with open("training_data/tst.from", "a", encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content + '\n')

        with open("training_data/tst.to", "a", encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(content + '\n')

        test_done = True

    else:
        with open("training_data/tr.from", "a", encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content + '\n')
        with open("training_data/tr.to", "a", encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(content + '\n')

    counter += 1
    if counter & 20 == 0:
        print(counter * limit, 'rows completed so far')
