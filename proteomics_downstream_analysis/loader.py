import pandas as pd
import os
from functools import reduce

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

        path = os.path.join(self.filepath, self.fc)
        fc_data = pd.read_csv(path)
        path = os.path.join(self.filepath, self.pv)
        pv_data = pd.read_csv(path)
        path = os.path.join(self.filepath, self.qv)
        qv_data = pd.read_csv(path)

        return fc_data, pv_data, qv_data
    
    def merge(self, datasets, suffixes, on=["Genes", "First.Protein.Description"], how="inner"):
        """
        Merge the datasets and add suffixes
        """
        datasets = [d.set_index(on).select_dtypes("float") for d in datasets]

        # add suffix
        zipped = zip(datasets, suffixes)
        datasets = [d.add_suffix(s) for d, s in zipped]

        # merge the datasets
        merged = reduce(lambda left, right: pd.merge(left, right, on=on, how=how), datasets)
        return merged