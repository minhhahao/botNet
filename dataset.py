import os
import pandas as pd
from datetime import datetime

class dataHandler:
    # General database
    database = []

    # Directory Settings
    JSON_dir = ''
    temporary_json_dir = ''
    output_dir = ''

    # Array of String indicates location of tmp file
    tmp_arr = []

    # row counter for debugging
    row_counter = 0

    def __init__(self, database_dir=None):
        if os.path.exists(database_dir):
            # Set string paths
            self.database = pd.DataFrame()
            self.JSON_dir = database_dir + "/JSON"
            self.temporary_json_dir = database_dir + "/temp"
            self.output_dir = database_dir + "/output"

            # Walking tmp_array for big data
            self.tmp_arr = list(pos_json for pos_json in os.listdir(
                self.temporary_json_dir) if pos_json.endswith('.json'))
        else:
            print("Invalid directory! Terminated @ " + str(datetime.now()))

    def getData(self, file=None):
