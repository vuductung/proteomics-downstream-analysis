import pandas as pd
import os

class Loader():
    
    """
    This class encapsulates loading methods
    """
    def __init__(self, filepath=None):

       self.filepath = filepath
       self.fc = "fc.csv"
       self.pv = "pv.csv"
       self.qv = "qv.csv"
    
    def stat_reader(self):
        """
        Read the statistics file
        """

        path = os.join(self.filepath, self.fc)
        fc_data = pd.read_csv(path)
        path = os.join(self.filepath, self.pv)
        pv_data = pd.read_csv(path)
        path = os.join(self.filepath, self.qv)
        qv_data = pd.read_csv(path)

        return fc_data, pv_data, qv_data