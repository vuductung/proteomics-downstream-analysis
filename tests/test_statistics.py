import pandas as pd
import numpy as np
from proteomics_downstream_analysis.stats import Statistics
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
        
        self.ancova_data = pd.DataFrame({
                    'age': [10, 12, 13, 16, 12, 18, 12, 13, 16],
                    'sex': [0, 1, 1, 1, 0, 1, 0, 1, 1],
                    'sample': ['Control', 'Control', 'Control',
                            'Disease1', 'Disease1', 'Disease1',
                            'Disease2', 'Disease2', 'Disease2'],
                    'ProteinId_1': [123., 124., 128., 129., 194.,
                                    283., 290., 290., 290.,],
                    'ProteinId_2': [123., 134., 188., 199., 154.,
                                    273., 210., 210., 210.,],
                    'ProteinId_3': [173., 134., 148., 179., 124.,
                                    283., 220., 260., 220.,],
                    'ProteinId_4': [163., 144., 128., 119., 124.,
                                    283., 210., 200., 260.,],
                    'ProteinId_5': [183., 154., 178., 129., 154.,
                                    283., 200., 220., 270.,],
                    'ProteinId_6': [123., 124., 128., 129., 194.,
                                    283., 290., 290., 290.,],
                    'ProteinId_7': [123., 134., 188., 199., 154.,
                                    273., 210., 210., 210.,],
                    'ProteinId_8': [173., 134., 148., 179., 124.,
                                    283., 220., 260., 220.,],
                    'ProteinId_9': [163., 144., 128., 119., 124.,
                                    283., 210., 200., 260.,],
                    'ProteinId_10': [183., 154., 178., 129., 154.,
                                     283., 200., 220., 270.,],
                })
        
        # some preprocessing for ancova data
        self.cov_data = self.ancova_data.iloc[:, :2]
        self.ancova_data = self.ancova_data.iloc[:, 2:].set_index('sample').T.reset_index(names='Protein.Ids')
        self.ancova_data['Genes'] = self.ancova_data['Protein.Ids']
        self.cov = ['age', 'sex']
        self.groups =[['Control', 'Disease1'], ['Control', 'Disease2']]

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

    # def test_ancova(self):

    #     res = self.helper.ancova(self.ancova_data, self.cov_data, self.cov)
    #     pvalues = np.array(res['-log10 pvalue'])
    #     pvaluestest = np.array([1.2793825470554916, 0.33969768256486804, 0.4643819003925277,
    #                             0.819345265524742, 0.6797582459528124, 1.2793825470554916,
    #                             0.33969768256486804, 0.4643819003925277, 0.819345265524742,
    #                             0.6797582459528124]
    #                             )
    #     tolerance = 1e-9 

    #     self.assertTrue(all(abs(pv - pvt) <= tolerance for pv, pvt in zip(pvalues, pvaluestest)))

    def test_two_tailed_ancova_pvalues(self):
        pvals, _, _ = self.helper.two_tailed_ancova(self.ancova_data,
                                                       self.cov_data,
                                                       self.groups,
                                                       'sample',
                                                       self.cov)
        
        pvals = np.array(pvals)
        pvals=np.concatenate([pvals[:, 0], pvals[:,1]])
        
        pvaluestest = np.array([0.18193002, 0.5267712 , 1.20771329, 0.68867216, 0.81890753,
                                0.18193002, 0.5267712 , 1.20771329, 0.68867216, 0.81890753,
                                3.55702539, 0.52897325, 1.11859599, 0.5515604 , 0.27872586,
                                3.55702539, 0.52897325, 1.11859599, 0.5515604 , 0.27872586]
                                )

        tolerance = 1e-8
        self.assertTrue(all(abs(pvals - pvaluestest) <= tolerance))

    def test_two_tailed_ancova_qvalues(self):
        _, qvals, _ = self.helper.two_tailed_ancova(self.ancova_data,
                                                       self.cov_data,
                                                       self.groups,
                                                       'sample',
                                                       self.cov)
        
        qvals = np.array(qvals)
        qvals=np.concatenate([qvals[:, 0], qvals[:,1]])
        
        qvaluestest = np.array([0.65776381, 0.371654  , 0.30992508, 0.34133167, 0.34133167,
                                0.65776381, 0.371654  , 0.30992508, 0.34133167, 0.34133167,
                                0.00138658, 0.36977433, 0.19025848, 0.36977433, 0.52634941,
                                0.00138658, 0.36977433, 0.19025848, 0.36977433, 0.52634941]
                                )
        tolerance = 1e-8
        self.assertTrue(all(abs(qvals - qvaluestest) <= tolerance))


    def test_two_tailed_ancova_results(self):
        _, _, results = self.helper.two_tailed_ancova(self.ancova_data,
                                                       self.cov_data,
                                                       self.groups,
                                                       'sample',
                                                       self.cov)
        pvals_res = np.array(pd.concat([results[0]['-log10 pvalue'], 
                                        results[1]['-log10 pvalue']],
                                        axis=0))
        pvals = np.array(pvals_res)
        
        pvaluestest = np.array([0.18193002, 0.5267712 , 1.20771329, 0.68867216, 0.81890753,
                                0.18193002, 0.5267712 , 1.20771329, 0.68867216, 0.81890753,
                                3.55702539, 0.52897325, 1.11859599, 0.5515604 , 0.27872586,
                                3.55702539, 0.52897325, 1.11859599, 0.5515604 , 0.27872586]
                                )

        tolerance = 1e-8
        self.assertTrue(all(abs(pvals_res - pvaluestest) <= tolerance))

if __name__ == '__main__':
    unittest.main()