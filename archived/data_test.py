import data
import os

tf = 201601
db_dir = '/home/aar0npham/Documents/Coding/tob/database'
RC = str(os.path.join(os.getcwd(), 'database', 'RC',
                      "RC_{}.json".format(tf)))
output = str(os.path.join(os.getcwd(), 'database', 'output',
                          'dataset.txt'))
tmp = str(os.path.join(os.getcwd(), 'database',
                       'temp'))


db = data.Processor(database_dir=db_dir, tf=tf)
