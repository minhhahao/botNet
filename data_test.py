import dataset
import os
import pandas as pd

tf = 201010
db_dir = '/home/aar0npham/Documents/Coding/tob/database'
RC = str(os.path.join(os.getcwd(), 'database', 'RC', "RC_{}.json".format(tf)))
RC
output = str(os.path.join(os.getcwd(), 'database', 'output',
                          'dataset.txt'))
tmp = str(os.path.join(os.getcwd(), 'database',
                       'temp'))
# Another way to find files in 1 line
# import fnmatch
# dir = list(file for file in os.listdir('database/RC') if fnmatch.fnmatch(file, 'RC_{}.json'.format(tf)))

db = dataset.Processor(database_dir=db_dir, tf=tf)
db.workingFile()
db.checkSize()
if os.path.getsize(RC) > 5e9:
    for f in db.tmp_arr:
        f_dir = os.path.join(db.temporary_json_dir, f)
        df = db.createDataFrame(file=f_dir)
        df.output()
else:
    dfr = db.createDataFrame(file=RC)
    db.output(output_file=output)
