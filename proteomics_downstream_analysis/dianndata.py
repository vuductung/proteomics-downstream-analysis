import pandas as pd
import numpy as np
from proteomics_downstream_analysis.preprocessor import Preprocessor
from proteomics_downstream_analysis.multileveldata import MultiLevelData
from proteomics_downstream_analysis.dimensionalityreduction import DimensionalityReduction
from proteomics_downstream_analysis.contamination import ContaminationAnalysis
from proteomics_downstream_analysis.statistics import Statistics
from proteomics_downstream_analysis.visualizer import Visualizer
from proteomics_downstream_analysis.dataqualityinformation import DataQualityInformation

class DiannData(MultiLevelData, DimensionalityReduction, ContaminationAnalysis, Statistics, Visualizer, DataQualityInformation):
    
    """ This class encapsulates methods for DIANN output tables """
    
    def __init__(self, filepath=None):

        self.title =[]
        self.datasets = []

        if filepath is None:
            self._Preprocessor = Preprocessor()
            self.datasets = []
            
        else:
            self._Preprocessor = Preprocessor()
            self.data = pd.read_csv(filepath, delimiter='\t')
            self.datasets = []

    def update_col_names(self, col_names):
            
        """
        Change the colums names of DIANN output tables
        Replicates will have same name

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        self.data : pandas.DataFrame
            Updated DIANN output table with new column names
        """
        
        self.data.columns = ['Protein.Group',
                             'Protein.Ids',
                             'Protein.Names',
                             'Genes', 
                             'First.Protein.Description'] + col_names
    
    def drop_col(self, col_name):
        """
        Drop columns from DIANN output tables
        Parameters
        ----------
        col_name : str
            Column name to be dropped
        
        Returns
        -------
        self.data : pandas.DataFrame
            Updated DIANN output table with dropped column
        """
        self.data = self.data.drop(col_name, axis=1)
        return self.data
    
    def preprocessing(self):
        
        """
        Prepocess data
        """
        self.data = self._Preprocessor._process(self.data)

    def change_dtypes(self):

        """ 
        Change data types of columns
        """
        self.data = self._Preprocessor._change_dtypes(self.data)

    def add_data(self, data, title):

        """
        Add data to the datasets list

        Parameters
        ----------
        data :  pandas.DataFrame
            Data to be added
            
        title : str
            Title of the data

        Returns
        -------
        self.datasets : list
            List of datasets
        self.title : list
            List of data titles
        """

        self.datasets.append(data)
        self.title.append(title)
        print('Data with the title "{}" is added'.format(title))
        print('Total number of datasets: "{}"'.format(len(self.datasets)))

    def del_data(self, index):

        """
        Delete data from the datasets list

        Parameters
        ----------
        index : int
            Index of the data to be deleted

        Returns
        -------
        self.datasets : list
            List of datasets without the deleted data
        """
        del self.datasets[index]

    def from_spectronaut_to_diann(self, filepath):
        
        data = pd.read_csv(filepath)
        data = data.filter(regex='Groups|UniProtIds|ProteinNames|Genes|ProteinDescriptions|raw.PG.Quantity')

        # rename columns
        col_rename = {'PG.ProteinGroups':'Protein.Group',
                    'PG.Genes':'Genes',
                    'PG.ProteinDescriptions': 'First.Protein.Description',
                    'PG.UniProtIds': 'Protein.Ids',
                    'PG.ProteinNames': 'Protein.Names'}
        data = data.rename(columns = col_rename)

        # reorder columns
        reorder = ['Protein.Group',
                'Protein.Ids',
                'Protein.Names',
                'Genes',
                'First.Protein.Description'] + data.columns[5:].tolist()
        data = data[reorder]

        # introduce np.nan
        data = data.replace('Filtered', np.nan)
        self.data = data

    def reorder_columns(self):

        # reorder the columns (string first, float last)
        float_cols = self.data.select_dtypes(float).columns.to_list()
        string_cols = self.data.select_dtypes('string').columns.to_list()

        self.data = self.data[string_cols + float_cols]

    def synchronize_data_and_annotation(self, annotation, sample_id):

        # synchronize data and annotation by a specific sample ID

        sample_id_data = self.data.select_dtypes(float).columns
        sample_id_annot = annotation[sample_id].tolist()

        intersec = list(set(sample_id_annot).intersection(set(sample_id_data)))

        annotation = annotation[annotation[sample_id].isin(intersec)]

        return annotation