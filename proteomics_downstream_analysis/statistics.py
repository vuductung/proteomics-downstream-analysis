import pandas as pd
import numpy as np

from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import pingouin as pg

from tqdm import tqdm

from proteomics_downstream_analysis.parallelprocessing import ParallelProcessing

class Statistics(ParallelProcessing):

    """
    Methods for statistical analysis of proteomics data.
    
    """

    def prepare_for_ancova(self, data, cov_data):

        data = data.set_index('Protein.Ids').select_dtypes(float).T.reset_index(names='sample')

        data = data = pd.concat([cov_data, data], axis=1)

        return data

    def split_data_by_groups(self, data, groups, sample_col):
        
        return [data[data[sample_col].isin(group)] for group in groups]
    
    def perform_ancova(self, data, cov_data, cov):
        
        proteinids = data['Protein.Ids'].tolist()
        genes = data['Genes'].tolist()
        data = self.prepare_for_ancova(data, cov_data)

        results = [pg.ancova(data=data, dv=id, between='sample', covar=cov).iloc[0] for id in tqdm(proteinids)]

        results = pd.DataFrame(results)
        results['Protein.Ids'] = proteinids
        results['Genes'] = genes
        results['qvalue'] = fdrcorrection(results['p-unc'])[1]
        results['-log10 pvalue'] = -np.log10(results['p-unc'])
        results = results[['Protein.Ids', 'Genes', 'Source', 'SS',
                           'DF', 'F','np2', '-log10 pvalue', 'qvalue']]
        
        self.ancova_results = results.copy()

        return results
    
    def ancova(self, datasets, cov_data, covariates):

        return self.paralell_processing(datasets, self.perform_ancova, cov_data, covariates)

    def two_tailed_ancova(self, dataset, covar_data, meta_col, cov, groups):

        """
        A two tailed ancova test for each protein id in the dataset. 
        The test filters the covariance data and protein group data
        for each group in groups and thus is performed for each group
        in groups, preventing the need for post hoc testing. 

        Parameters
        ----------
        dataset : pd.DataFrame
            Protein group data (wide format).
        
        covar_data : pd.DataFrame
            Covariance data (long format). It's important that
            the covariance data should be in the same order as
            the pg data.

        meta_col : str
            Column name in covar_data that contains the group
            information to be filtered on for each group in groups.
        
        cov : list
            List of covariates to be used in the ancova test.

        groups : list
            List of lists with groups to be compared. 
            e.g. [('group1', 'group2'), ('group1', 'group3')]

        Returns
        -------
        results : pandas.DataFrame
            Ancova result data for each protein id.
        pv_data : pandas.DataFrame
            -log10 pvalue data.
        qv_data : pandas.DataFrame
            q-value data. Benjamini-Hochberg adjusted pvalues.
        """

        results = []
        pv_data = pd.DataFrame()
        qv_data = pd.DataFrame()
        pv_data[['Protein.Ids', 'Genes']] = dataset.reset_index()[['Protein.Ids', 'Genes']].copy()
        qv_data[['Protein.Ids', 'Genes']] = dataset.reset_index()[['Protein.Ids', 'Genes']].copy()

        for group in groups:
            data = dataset[group].reset_index()
            cov_data = covar_data[covar_data[meta_col].isin(group)].reset_index()

            result= self.ancova(data, cov_data, cov)
            results.append(result)

            pv_data[f'{group[0]}/{group[1]}'] = result['-log10 pvalue'].copy()
            qv_data[f'{group[0]}/{group[1]}'] = result['qvalue'].copy()
        
        return results, pv_data, qv_data
        
    def anova(self):
        
        f_stat_data = pd.DataFrame()
        anova_pv_data = pd.DataFrame()

        columns = self.data.select_dtypes('string').columns

        samples = [self.data[col] for col in columns]

        f_stat, pvalues = stats.f_oneway(*samples, axis=1)
        f_stat_data['f_stat'] = f_stat
        anova_pv_data['pvalue'] = pvalues
        anova_pv_data['qvalue'] = fdrcorrection(pvalues)[1]

        self.f_stat_data = f_stat_data
        self.anova_pv_data = anova_pv_data

    def student_ttest(self, comparisons, return_output =False):
    
        """
        Unpaired two tailed student's t-test with BH pvalue
        adjustment and fold change calculation

        Parameters
        ----------
        comparisons : list
            List of tuples with comparisons to be made.
        return_output : boolean
             If True, fc_data, pv_data and qv_data will be returned. 
             (Default value = False)  

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
        t_stat_data = pd.DataFrame()
        ci_data = pd.DataFrame()
        cohensd_data = pd.DataFrame()

        for a, b in comparisons:
            
            # T test for each row pair.
            t_stats, pvalues = stats.ttest_ind(a=self.data[a], b=self.data[b], axis=1, nan_policy='omit')

            t_stat_data[f'{a}/{b}'] = t_stats

            pv_data[f'{a}/{b}'] = - np.log10(pvalues)

            # Mean difference for each row.
            fc_data[f'{a}/{b}'] = self._fold_change(self.data[a], self.data[b])
            
            # 95% confidence interval for each row.
            ci_data[f'{a}/{b}'] = self._ttest_conf_int(self.data[a], self.data[b])

            # Cohens d for each row.
            cohensd_data[f'{a}/{b}'] = self._cohensd(self.data[a], self.data[b])

            # bh adjusted pvalue 
            qv_data[f'{a}/{b}'] = fdrcorrection(pvalues)[1]

        self.fc_data = self.data.select_dtypes('string').merge(fc_data, left_index = True, right_index = True)
        self.pv_data = self.data.select_dtypes('string').merge(pv_data, left_index = True, right_index = True)
        self.qv_data = self.data.select_dtypes('string').merge(qv_data, left_index = True, right_index = True)
        self.t_stat_data = self.data.select_dtypes('string').merge(t_stat_data, left_index = True, right_index = True)
        self.ci_data = self.data.select_dtypes('string').merge(ci_data, left_index = True, right_index = True)
        self.cohensd_data = self.data.select_dtypes('string').merge(cohensd_data, left_index = True, right_index = True)

        if return_output == True:
            return self.fc_data, self.pv_data, self.qv_data, self.t_stat_data, self.ci_data, self.cohensd_data
    
    def _ttest_conf_int(self, a, b, conf_int=0.95):

        sd_a = np.std(a, ddof=1, axis=1)
        sd_b = np.std(b, ddof=1, axis=1)

        n_a = a.shape[1]
        n_b = b.shape[1]
        df = n_a + n_b - 2
        s_pooled = np.sqrt(((n_a-1)*sd_a**2 + (n_b-1)*sd_b**2) / df)

        percentile = (1 + conf_int)/2
        t_crit = stats.t.ppf(percentile, df)
        lower = self._fold_change(a, b) - t_crit * s_pooled
        upper = self._fold_change(a, b) + t_crit * s_pooled

        # Combine lower with upper
        conf_intervals = list(zip(lower, upper))
        return conf_intervals

    def _fold_change(self, a, b):

        fold_change = np.mean(a, axis=1) - np.mean(b, axis=1)
        return fold_change
    
    def _cohensd(self, a, b):

        n1, n2 = a.shape[1], b.shape[1]
        s1, s2 = np.std(a, ddof=1, axis=1), np.std(b, ddof=1, axis=1)
        s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1 + n2 - 2))

        return (self._fold_change(a, b)) / s_pooled
    
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