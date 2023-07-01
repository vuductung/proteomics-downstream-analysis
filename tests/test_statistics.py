import numpy as np
import pandas as pd
from proteomics_downstream_analysis.statistics import Statistics
import unittest

class TestStatistics(unittest.TestCase):

    def setUp(self):
        
        self.helper = Statistics() 
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
        
    def test_fold_change(self):

        comparison = [('WT', 'KO')]
        self.helper.data = self.data
        self.helper.student_ttest(comparisons=comparison)
        self.assertTrue(all(self.helper.fc_data.select_dtypes('float') == 1))

    def test_get_unique_comb(self):
        a = ['WT']
        b = ['KO1', 'KO2', 'KO3']
        expected = [('WT', 'KO1'), 
                    ('WT', 'KO2'),
                    ('WT', 'KO3'),]
        comparisons = self.helper.get_unique_comb(a, b)
        self.assertTrue(comparisons == expected) 

if __name__ == '__main__':
    unittest.main()