import pandas as pd
import numpy as np
from proteomics_downstream_analysis.parallelprocessing import ParallelProcessing
import unittest

class TestParallelProcessing(unittest.TestCase):

    def setUp(self):
        
        self.helper = ParallelProcessing() 
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

        self.data1 = pd.DataFrame({
            'Protein.Ids': ['P123', 'P234', 'P345', 'P456', 'P567',
                            'P232', 'P124', 'P214', 'P352', 'P109'],
            'WT1': [1.0, 1.0, 1, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],
            'WT2': [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],
            'WT3': [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],
            'KO1': [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],
            'KO2': [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],
            'KO3': [1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1, 1.0, 1.0],   
        })
    

    def test_split_data_for_parallel_processing(self):

        datasets = self.helper.split_data_for_parallel_processing(self.data,
                                                       numbers_of_datasets=3,
                                                       subset_column='Protein.Ids',)
        
        self.assertTrue(len(datasets) == 3)

    def test_parallel_processing(self):

        def dummy_function(data):
            return data.select_dtypes(float)

        data = self.helper.paralell_processing(self.data1, dummy_function)
        data = pd.concat(data, axis=0)

        self.assertTrue(data.shape == (10, 6))

if __name__ == '__main__':
    unittest.main()