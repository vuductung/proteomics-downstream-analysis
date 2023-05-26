import pandas as pd
from preprocessing import Preprocessor
from multileveldata import MultiLevelData
from dimensionalityreduction import DimensionalityReduction
from contamination import ContaminationAnalysis
from statistics import Statistics
from visualizer import Visualizer
from dataqualityinformation import DataQualityInformation

class DiannData(MultiLevelData, DimensionalityReduction, ContaminationAnalysis, Statistics, Visualizer, DataQualityInformation):
    
    """
    param data: diann output data
    """
    
    def __init__(self, filepath):
                
        self._Preprocessor = Preprocessor()
        self.data = pd.read_csv(filepath, delimiter='\t')
        self.datasets = []

    def update_col_names(self, col_names):
            
        """
        change the colums names of DIANN output tables
        replicates will have same name
        param col_names: list of column names in the order of measurement
        """
        
        self.data.columns = ['Protein.Group',
                             'Protein.Ids',
                             'Protein.Names',
                             'Genes', 
                             'First.Protein.Description'] + col_names
    
    def drop_col(self, col_name):
        self.data = self.data.drop(col_name, axis=1)
        return self.data
    
    def preprocessing(self):
        self.data = self._Preprocessor._process(self.data)

    def change_dtypes(self):
        self.data = self._Preprocessor._change_dtypes(self.data)

    def add_data(self, data):
        self.datasets.append(data)

    def del_data(self, index):
        del self.datasets[index]