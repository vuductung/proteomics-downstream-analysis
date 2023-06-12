import numpy as np
import pandas as pd
from proteomics_downstream_analysis import Preprocessing
import unittest

class TestPreprocessing:

    def __init__(self):
        pass

    def setUp(self):
        self.helper = Preprocessing() 
        self.data = pd.DataFrame({
            'Protein.Group': [1, 2],
            'Protein.Ids': [112, 115],
            'Protein.Names': ['p1', 'p2'],
            'Genes': ['g1', 'g2'],
            'First.Protein.Description': ['d1', 'd2'],
            'some_integer_column': [100, 200],
            'some_float_column': [0.5, 1.5]
        })

    def test_change_datatypes(self):
        
        converted_data = self.helper._change_datatypes(self.data)
        expected_columns_dt_in_string = ['Protein.Group', 'Protein.Ids', 'Protein.Names', 'Genes', 'First.Protein.Description']
        expected_columns_dt_in_float = ['some_integer_column', 'some_float_column']

        for col in expected_columns_dt_in_string:
            self.assertTrue(np.issubdtype(converted_data[col].dtype, np.dtype("O")))

        for col in expected_columns_dt_in_float:
            self.assertTrue(np.issubdtype(converted_data[col].dtype, np.dtype("float64")))

if __name__ == '__main__':
    unittest.main()