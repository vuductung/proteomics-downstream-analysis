import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

class Statistics:
    """ """

    def student_ttest(self, comparisons, return_output =False):
    
        """Unpaired two tailed student's t-test with BH pvalue adjustment and fold change calculation

        Parameters
        ----------
        comparisons : list
            List of tuples with comparisons to be made.
        return_output : boolean
             If True, fc_data, pv_data and qv_data will be returned. (Default value = False)  

        Returns
        -------
        fc_data : pandas.DataFrame
            Fold change data.
        pv_data : pandas.DataFrame
            -log10 pvalue data.
        qv_data : pandas.DataFrame
            q-value data. Benjamini-Hochberg adjusted pvalues.
        """
        
        # create fc_data, pv_data, qv_data
        fc_data = pd.DataFrame()
        pv_data = pd.DataFrame()
        qv_data = pd.DataFrame()
        
        for a, b in comparisons:
            

            a_data = self.data[a].to_numpy()
            b_data = self.data[b].to_numpy()

            # T test for each row pair.
            pvalues = stats.ttest_ind(a=a_data, b=b_data, axis=1)[1]

            pv_data[f'{a}/{b}'] = - np.log10(pvalues)

            # Mean difference for each row.
            fc_data[f'{a}/{b}'] = np.mean(a_data, axis=1) - np.mean(b_data, axis=1)

            # bh adjusted pvalue 
            qv_data[f'{a}/{b}'] = fdrcorrection(pvalues)[1]
        
        self.fc_data = self.data.select_dtypes('string').merge(fc_data, left_index = True, right_index = True)
        self.pv_data = self.data.select_dtypes('string').merge(pv_data, left_index = True, right_index = True)
        self.qv_data = self.data.select_dtypes('string').merge(qv_data, left_index = True, right_index = True)
        
        if return_output == True:
            return self.fc_data, self.pv_data, self.qv_data
    
    def get_unique_comb(self, a, b):

        """
        Get unique combinations between array a and b

        Parameters
        ----------
        a : list
            a list of strings     
        b : list
            a list of strings

        Returns
        -------
        comparisons : list
            a list of tuples with unique combinations between a and b.
        """

        comparisons = []
        for comp1 in a:
            for comp2 in b:
                comparisons.append((comp1, comp2))

        return comparisons

    def get_summary_data(self, summary='mean'):

        """
        Create a summary dataframe with mean or median values for each sampletype

        Parameters
        ----------
        summary : str
             Calculate the 'mean' or 'median' (Default value = 'mean').

        Returns
        -------
        summary_data : pandas.DataFrame
            A summary dataframe with mean or median values for each sampletype.
        """

        string_data = self.data.select_dtypes('string')

        if summary == 'mean':
            float_data = self.data.select_dtypes('float').T.reset_index(names='sample').groupby('sample').mean().T
        
        elif summary == 'median':
            float_data = self.data.select_dtypes('float').T.reset_index(names='sample').groupby('sample').median().T 
        
        summary_data = pd.concat([string_data, float_data], axis=1)

        self.summary_data = summary_data.copy()