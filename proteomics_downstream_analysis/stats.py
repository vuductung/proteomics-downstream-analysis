import pandas as pd
import numpy as np

from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import pingouin as pg

from tqdm import tqdm
from numba import njit

@njit
def nan_correlation_matrix(data):
    n = len(data)
    correlation_matrix = np.empty((n, n))

    for i in range(n):
        for j in range(n):  # Compute all elements
            mask = np.isfinite(data[i]) & np.isfinite(data[j])

            if np.sum(mask) > 1:  # Ensure there are at least two data points
                xi = data[i][mask]
                xj = data[j][mask]
                std_dev_i = np.std(xi)
                std_dev_j = np.std(xj)

                if (std_dev_i > 0) and (std_dev_j > 0):
                    mean_i = np.mean(xi)
                    mean_j = np.mean(xj)
                    sparsity = np.mean(mask)
                    covariance = np.mean((xi - mean_i) * (xj - mean_j))
                    corr = covariance / (std_dev_i * std_dev_j)
                    correlation_matrix[i, j] = corr * sparsity

                else:
                    correlation_matrix[i, j] = np.nan  # Set to NaN if no variation

            else:
                correlation_matrix[i, j] = (
                    np.nan
                )  # Set to NaN if not enough data points

    np.fill_diagonal(correlation_matrix, np.nan)
    
    return correlation_matrix


class Statistics():

    """
    Methods for statistical analysis of proteomics data.
    
    """

    def prepare_for_ancova(self, data, cov_data):

        data = data.set_index('Protein.Ids').select_dtypes(float).T.reset_index(names='sample')

        data = data = pd.concat([cov_data, data], axis=1)

        return data

    def split_data_by_groups(self, data, groups, sample_col):
        
        return [data[data[sample_col].isin(group)] for group in groups]
    
    def proteinid_and_genes_extraction(self, data):
        
        return data['Protein.Ids'].tolist(), data['Genes'].tolist()
    
    def ancova_dataframe_creation_and_column_modification(self, results, proteinids, genes):

        results = pd.DataFrame(results)
        results['Protein.Ids'] = proteinids
        results['Genes'] = genes
        results['qvalue'] = fdrcorrection(results['p-unc'])[1]
        results['-log10 pvalue'] = -np.log10(results['p-unc'])
        results = results[['Protein.Ids', 'Genes', 'Source', 'SS',
                           'DF', 'F','np2', '-log10 pvalue', 'qvalue']]
        
        return results

    def create_qv_and_pv_dataframe(self, result, group):
        
        pv_data = result[['Protein.Ids', 'Genes']].copy()
        qv_data = result[['Protein.Ids', 'Genes']].copy()

        pv_data[f'{group[0]}/{group[1]}'] = result['-log10 pvalue'].copy()
        qv_data[f'{group[0]}/{group[1]}'] = result['qvalue'].copy()

        return pv_data, qv_data

    def perform_ancova(self, data, cov_data, cov):
        
        proteinids, genes = self.proteinid_and_genes_extraction(data)

        data = self.prepare_for_ancova(data, cov_data)

        results = [pg.ancova(data=data, dv=id, between='sample', covar=cov).iloc[0] for id in tqdm(proteinids)]

        results = self.ancova_dataframe_creation_and_column_modification(results, proteinids, genes)
        
        self.ancova_results = results.copy()

        return results
    
    def perform_two_tailed_ancova(self, data, cov_data, groups, sample_col, cov):
       
        results = []

        proteinids, genes = self.proteinid_and_genes_extraction(data)
        data = self.prepare_for_ancova(data, cov_data)
        datasets = self.split_data_by_groups(data, groups, sample_col)

        for data in datasets:
            result = [pg.ancova(data=data, dv=id, between='sample', covar=cov).iloc[0] for id in tqdm(proteinids)]
            result = self.ancova_dataframe_creation_and_column_modification(result, proteinids, genes)
            results.append(result)
        
        pv_qv_datasets = [self.create_qv_and_pv_dataframe(result, group) for group, result in zip(groups, results)]

        pv_datasets = [dataset[0].set_index(['Protein.Ids', 'Genes']) for dataset in pv_qv_datasets]
        qv_datasets = [dataset[1].set_index(['Protein.Ids', 'Genes']) for dataset in pv_qv_datasets]

        pv_data = pd.concat(pv_datasets, axis=1)
        qv_data = pd.concat(qv_datasets, axis=1)

        return pv_data, qv_data, results

    def two_tailed_ancova(self, datasets, cov_data, groups, sample_col, cov):

        return self.perform_two_tailed_ancova(datasets, cov_data, groups, sample_col, cov)

    # def two_tailed_ancova(self, dataset, cov_data, groups, sample_col, cov):
        
    #     results = self.paralell_processing(dataset, self.perform_two_tailed_ancova, cov_data, groups, sample_col, cov)

    #     # data preparation
    #     pv_data = pd.concat([result[0] for result in results], axis=0).reset_index()
    #     qv_data = pd.concat([result[1] for result in results], axis=0).reset_index()
    #     res = []
        
    #     for idx in range(len(groups)):
    #             res.append(pd.concat([result[2][idx] for result in results], axis=0).reset_index(drop=True))

    #     return pv_data, qv_data, res
        
    # def ancova(self, datasets, cov_data, covariates):
        
    #     results = self.paralell_processing(datasets, self.perform_ancova, cov_data, covariates)
        
    #     results = pd.concat(results, axis=0).reset_index(drop=True)

    #     return results

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

    def student_ttest(self, data, comparisons, return_output =False):
    
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
            t_stats, pvalues = stats.ttest_ind(a=data[a], b=data[b], axis=1, nan_policy='omit')

            t_stat_data[f'{a}/{b}'] = t_stats

            pv_data[f'{a}/{b}'] = - np.log10(pvalues)

            # Mean difference for each row.
            fc_data[f'{a}/{b}'] = self._fold_change(data[a], data[b])
            
            # 95% confidence interval for each row.
            ci_data[f'{a}/{b}'] = self._ttest_conf_int(data[a], data[b])

            # Cohens d for each row.
            cohensd_data[f'{a}/{b}'] = self._cohensd(data[a], data[b])

            # bh adjusted pvalue 
            qv_data[f'{a}/{b}'] = fdrcorrection(pvalues)[1]

        self.fc_data = data.select_dtypes('string').merge(fc_data, left_index = True, right_index = True)
        self.pv_data = data.select_dtypes('string').merge(pv_data, left_index = True, right_index = True)
        self.qv_data = data.select_dtypes('string').merge(qv_data, left_index = True, right_index = True)
        self.t_stat_data = data.select_dtypes('string').merge(t_stat_data, left_index = True, right_index = True)
        self.ci_data = data.select_dtypes('string').merge(ci_data, left_index = True, right_index = True)
        self.cohensd_data = data.select_dtypes('string').merge(cohensd_data, left_index = True, right_index = True)

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
    
    