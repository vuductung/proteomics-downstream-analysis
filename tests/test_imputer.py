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

        self.data1 = pd.DataFrame({
            'Protein.Names': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'Genes': ['g1', 'g2', 'g3', 'g4', 'g5'],
            'First.Protein.Description': ['d1', 'd2', 'd3', 'd4', 'd5'],
            'WT1': [5.0, np.nan, 12.0, 18.0, 12.0],
            'WT2': [11.0, 13.0, np.nan, 13.0, 15.0],
            'WT3': [13.0, np.nan, np.nan, 19.0, 10.0],
            'KO1': [12.0, 14.0, 9.0, 7.0, 11.0],
            'KO2': [11.0, 15.0, np.nan, np.nan, 18.0],
            'KO3': [12.0, 17.0, 14.0, 12.0, np.nan],
        })
        self.data1.columns = ['Protein.Names', 'Genes',
                        'First.Protein.Description',
                        'WT', 'WT', 'WT',
                        'KO', 'KO', 'KO']

        self.data1[['Protein.Names', 'Genes', 'First.Protein.Description']] = self.data1[['Protein.Names', 'Genes', 'First.Protein.Description']].astype('string')

        
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

    def test_normal_imputation(self):
        
        imputed_data = self.helper._normal_imputation(self.data1, axis=0, shift=1.8, width=0.3, seed=42)
        self.assertTrue(imputed_data.isnull().sum().sum() == 0)

    def test_normal_imputation_row_wise(self):
        
        imputed_data = self.helper._normal_imputation(self.data1, axis=0, shift=1.8, width=0.3, seed=42)
        imputed_values = np.array([4.70095814, 5.10594318, 9.12330024,
                                    3.37999379, 7.27819062,
                                    8.41641979, 9.2899923])
        tolerance = 1e-8
        na_mask = self.data1.select_dtypes(float).isnull()
        dif = (imputed_data.select_dtypes(float).values[na_mask] - imputed_values).sum()

        self.assertTrue(tolerance >  dif)

    def test_normal_imputation(self):

        imputed_data = self.helper._normal_imputation(self.data1, axis=0, shift=1.8, width=0.3, seed=42)
        imputed_values = np.array([11.6050754 , 11.43729928, 7.71616652,
                                    7.31944351,  7.18774868,
                                    2.95630842,  7.02579964])
        tolerance = 1e-8
        na_mask = self.data1.select_dtypes(float).isnull()
        dif = (imputed_data.select_dtypes(float).values[na_mask] - imputed_values).sum()

        self.assertTrue(tolerance >  dif)


if __name__ == '__main__':
    unittest.main()