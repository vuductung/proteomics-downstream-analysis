import pandas as pd
import numpy as np
from proteomics_downstream_analysis.preprocessor import Preprocessor
from proteomics_downstream_analysis.multileveldata import MultiLevelData
from proteomics_downstream_analysis.dimensionalityreduction import DimensionalityReduction
from proteomics_downstream_analysis.contamination import ContaminationAnalysis
from proteomics_downstream_analysis.stats import Statistics
from proteomics_downstream_analysis.visualizer import Visualizer
from proteomics_downstream_analysis.dataqualityinformation import DataQualityInformation


class Analysis(MultiLevelData, DimensionalityReduction,
                ContaminationAnalysis, Statistics, Visualizer,
                DataQualityInformation):
    
    """
    This class encapsulates methods for DIANN output tables
    """
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

    def sort_by_column_names(self, data):
        '''
        sort data by column names
        '''
        string_data = data.select_dtypes('string')
        float_data = data.select_dtypes(float)

        float_data = float_data.sort_index(axis=1)
        data = string_data.join(float_data)

        return data

    def update_col_names(self, data, index, column_name):

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
        
        template = self.data.select_dtypes('float').columns.tolist()
        data[index] = data[index].astype('str')
        updated_col_data = data.set_index(index).reindex(template)

        col_names = updated_col_data[column_name].tolist()

        self.data.columns = self.data.select_dtypes('string').columns.tolist() + col_names
    
    def drop_col(self, data, col_name):

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
        self.data = data.drop(col_name, axis=1)

        return self.data
    
    def preprocessing(self, data, method, **kwargs):
        
        """
        Prepocess data
        """
        
        if method == 'simple':
            return self._Preprocessor._process(data)

        elif method == 'hybrid':
            return self._Preprocessor._hybrid_process(data, **kwargs)

        elif method == 'no imputation':
            return self._Preprocessor._simple_process(data, **kwargs)

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def change_dtypes(self, data):

        """ 
        Change data types of columns
        """
        self.data = self._Preprocessor._change_dtypes(data)

        return self.data

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
        self.data = data.copy()
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

    def reorder_columns(self, data):

        # reorder the columns (string first, float last)
        float_cols = data.select_dtypes(float).columns.to_list()
        string_cols = data.select_dtypes('string').columns.to_list()

        self.data = data[string_cols + float_cols]

        return self.data

    def synchronize_data_and_annotation(self, data, annotation, sample_id):

        # synchronize data and annotation by a specific sample ID

        sample_id_data = data.select_dtypes(float).columns
        sample_id_annot = annotation[sample_id].tolist()

        intersec = list(set(sample_id_annot).intersection(set(sample_id_data)))

        annotation = annotation[annotation[sample_id].isin(intersec)]

        return annotation
    
    def filter(self, data, col_name, value, axis):

        """
        Filter data by column names

        Parameters
        ----------
        col_names : list
            List of column names

        Returns
        -------
        self.data : pandas.DataFrame
            Filtered DIANN output table
        """
        if axis == 0:
            data = data[data[col_name].isin(value)]

        elif axis == 1:
            data = data[col_name]
            
        return data