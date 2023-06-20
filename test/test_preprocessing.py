import numpy as np
import pandas as pd
from proteomics_downstream_analysis.preprocessing import Preprocessing
import unittest

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        
        self.helper = Preprocessing() 
        self.data = pd.DataFrame({
            'Protein.Group': [1, 2, 3, 4, 5],
            'Protein.Ids': ['P123', 'P234', 'P345', 'P456', 'P567'],
            'Protein.Names': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'Genes': ['g1', 'g2', 'g3', 'g4', 'g5'],
            'First.Protein.Description': ['d1', 'd2', 'd3', 'd4', 'd5'],
            'WT1': [1.0, 1.0, 1, 1.0, 1.0],
            'WT2': [1.0, 1.0, 1.0, 1.0, 1.0],
            'WT3': [1.0, 1.0, 1.0, 1.0, 1.0],
            'KO1': [1.0, 1.0, 1.0, 1.0, 1.0],
            'KO2': [1.0, 1.0, 1.0, 1.0, 1.0],
            'KO3': [1.0, 1.0, 1.0, 1.0, 1.0],   
        })
        
        self.data.columns = ['Protein.Group',
                            'Protein.Ids',
                            'Protein.Names',
                            'Genes',
                            'First.Protein.Description',
                            'WT', 
                            'WT',
                            'WT',
                            'KO',
                            'KO',
                            'KO']
        
    def test_change_datatypes(self):
        
        # change dataype of test_data
        self.data = self.data.astype('object')

        converted_data = self.helper._change_datatypes(self.data)
        expected_columns_dt_in_string = ['Protein.Group',
                                         'Protein.Ids',
                                         'Protein.Names',
                                         'Genes',
                                         'First.Protein.Description']
        expected_columns_dt_in_float = ['WT', 'KO']

        for col in expected_columns_dt_in_string:
            self.assertTrue(converted_data[col].dtype == 'string')

        for col in expected_columns_dt_in_float:
            self.assertTrue(all(converted_data[col].dtypes == 'float'))

    def test_log2_transform(self):  
        
        converted_data = self.helper._log2_transform(self.data)

        expected_columns = ['WT', 'KO']

        for col in expected_columns:
            self.assertTrue(all(converted_data[col] == 0))

    def test_filter_for_valid_vals_in_one_exp_group(self):
            
            # insert nan values
            self.data.loc[0, 'WT'] = np.nan
            self.data.loc[0, 'KO'] = np.nan
            converted_data = self.helper._filter_for_valid_vals_in_one_exp_group(self.data)
            
            self.assertTrue(converted_data.shape[0] == 4)

    def test_impute_genes_with_protein_ids(self):
            
            # insert nan values
            self.data.loc[0, 'Genes'] = np.nan
            converted_data = self.helper._impute_genes_with_protein_ids(self.data)
            
            self.assertTrue(converted_data.loc[0, 'Genes'] == 'P123')

    def test_filter_for_valid_vals_perc(self):
         # insert na values
         self.data.loc[0:3, 'WT'] = np.nan
         self.data.loc[1,:] = np.nan
         converted_data = self.helper._filter_for_valid_vals_perc(self.data, 0.5)
         self.assertTrue(converted_data.shape[0] == 4)

    def test_filter_for_valid_vals_in_one_exp_group_perc(self):
         
        #insert na values
        self.data.loc[0, 'WT'] = np.nan
        self.data.loc[0, 'KO'] = np.nan

        converted_data = self.helper._filter_for_valid_vals_in_one_exp_group_perc(self.data, 0.7)
        self.assertTrue(converted_data.shape[0] == 4)

if __name__ == '__main__':
    unittest.main()