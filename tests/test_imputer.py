import numpy as np
import pandas as pd
from proteomics_downstream_analysis.imputer import Imputer
import unittest

class TestImputer(unittest.TestCase):

    def setUp(self):
        
        self.helper = Imputer() 
        self.data = pd.DataFrame(
                [['Genes_1', 2.0, 2.0, 2.0, np.nan, 2.0, 2.0],
                  ['Genes_2',6.0, np.nan, 6.0, 6.0, np.nan, 6.0],
                  ['Genes_3',7.0, 5.0, 5.0, 5.0, np.nan, 6.0],
                  ['Genes_4',6.0, np.nan, np.nan, 6.0, np.nan, np.nan],],
                  columns=['Genes', 'A', 'B', 'A', 'A', 'B', 'B']
                  )
        
    def test_simple_imputer_mean(self):

        imputed_data = self.helper._impute(self.data,
                                            kind='simple',
                                            strategy='mean',
                                            percentage=0.5,
                                            constant=None
                                            )
        
        self.assertTrue(imputed_data.isnull().sum().sum() == 5)

    def test_simple_imputer_mean(self):

        imputed_data = self.helper._impute(self.data,
                                            kind='simple',
                                            strategy='mean',
                                            percentage=0.5,
                                            constant=None
                                            )
        
        self.assertTrue(imputed_data.iloc[2,4] == 5.5)

    def test_simple_imputer_median(self):

        imputed_data = self.helper._impute(self.data,
                                            kind='simple',
                                            strategy='median',
                                            percentage=0.5,
                                            constant=None
                                            )
        
        self.assertTrue(imputed_data.iloc[2,4] == 5.5)

    def test_simple_imputer_constant(self):

        imputed_data = self.helper._impute(self.data,
                                            kind='simple',
                                            strategy='constant',
                                            percentage=0.5,
                                            constant=10
                                            )
        
        self.assertTrue(imputed_data.iloc[2,4] == 10.0)


    def test_knn_imputer(self):

        imputed_data = self.helper._impute(self.data,
                                            kind='knn',
                                            strategy='mean',
                                            percentage=0.5,
                                            constant=None
                                            )
        
        self.assertTrue(imputed_data.isnull().sum().sum() == 5)

    def test_impute_based_on_gaussian(self):
        
        imputed_data = self.helper._impute_based_on_gaussian(self.data)
        self.assertTrue(imputed_data.isnull().sum().sum() == 0)

if __name__ == '__main__':
    unittest.main()