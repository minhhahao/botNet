import sqlite3
import pandas as pd
from pathlib import Path

tf = ['2016-01', '2010-10']
dbpath = Path("/home/aar0npham/Documents/Coding/tob/structure/database/")
tpath = Path("/home/aar0npham/Documents/Coding/tob/structure/training_data/")

for timeframe in tf:
    db = dbpath / "{}.db".format(timeframe)
    connection = sqlite3.connect(str(db))
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
        cur_length = len(df)

        if not test_done:
            with open(tpath / "tst.from", "a", encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content + '\n')

            with open(tpath / "tst.to", "a", encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(str(content) + '\n')

            test_done = True

        else:
            with open(tpath / "tr.from", "a", encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content + '\n')
            with open(tpath / "tr.to", "a", encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(str(content) + '\n')

        counter += 1
        if counter & 20 == 0:
            print(counter * limit, 'rows completed so far')
