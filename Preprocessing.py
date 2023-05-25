import pandas as pd
import numpy as np

class Preprocessing:
    
    """
    This class encapsulates preprocessing steps
    """
    
    def __init__(self):
        pass 
        
    def _change_datatypes(self, data):
        
        """
        change datatypes
        """
        
        self.data = data.copy()
        for i in self.data.columns.unique():
            if i in ['Protein.Group', 'Protein.Ids', 'Protein.Names', 'Genes','First.Protein.Description']:
                self.data[i] = self.data[i].astype('string')
            else:
                self.data[i] = self.data[i].astype('float') 
        return self.data
    
    def _log2_transform(self, data):
        
        """
        log2 transfrom float data
        """

        for i in self.data.select_dtypes('float').columns.unique():
            self.data[i] = np.log2(self.data[i])
        return self.data
        
    def _filter_for_valid_vals_in_one_exp_group(self, data):
        
        """
        Filter for valid values in at least one experimental group
        param data: diann output table
        """
            
        columns_name = self.data.select_dtypes('float').columns.unique()
            
        indices = []
            
        for col in columns_name:
            non_na_indices = self.data[self.data[col].isna().sum(axis=1)==0].index.to_list()
            indices += non_na_indices
        indices = list(set(indices))
        self.data = self.data[self.data.index.isin(indices)].reset_index(drop = True)
        
        return self.data
    
    def _impute_genes_with_protein_ids(self, data):
        
        """
        Imputing missing values in Genes column with the protein IDs
        """
        imputation_values = [i.split(';')[0] for i in self.data[self.data['Genes'].isna()]['Protein.Ids'].values]
        
        for idx, value in zip(self.data[self.data['Genes'].isna()]['Protein.Ids'].index, imputation_values):
            self.data.loc[idx, ['Genes']] = value

        return self.data
    
    def _impute_based_on_gaussian(self, data):
         
        """
        This function imputes missing values in a dataset using a Gaussian distribution.
        The missing values are imputed by random sampling values from a Gaussian distribution with a mean
        of 3 standard deviations below the computed mean and a width of 0.3 times the computed standard deviation.
        The function returns a copy of the imputed dataset with the missing values replaced.
        """
        # select only float data
        float_data = self.data.select_dtypes('float')
        na_data = float_data[float_data.isna().any(axis=1)]

        # loop through na_data rows and generate imputed data
        imp_array = []

        for i in na_data.index: 
            na_count = na_data.loc[i].isna().sum()

            mean = na_data.loc[i].mean(skipna=True)
            std = na_data.loc[i].std(skipna=True)

            np.random.seed(i)
            imp_array.append(list(np.random.normal(loc=mean-3*std, scale=0.3*std, size=na_count)))

        imp_values_list = np.array([imp_value for innerlist in imp_array for imp_value in innerlist])

        copied_data = float_data.copy()
        columns = float_data.columns
        copied_data.columns = np.arange(copied_data.shape[1])
        
        # impute the data
        stacked_na_data = copied_data.stack(dropna=False)
        na_index = stacked_na_data[stacked_na_data.isna()].index
        stacked_na_data.loc[na_index] = imp_values_list
        imp_data = stacked_na_data.unstack()
        imp_data.columns = columns
        imp_data = self.data.select_dtypes('string').merge(imp_data, left_index=True, right_index=True)
        self.data = imp_data.copy()
        
        return self.data

    def _filter_for_valid_vals_perc(self, data, percentage):
    
        '''Filter for valid values using at least x % data completeness'''
    
        col_len = len(self.data.select_dtypes('float').columns)
        min_sum = np.ceil(col_len * percentage)
        bool_array = self.data.select_dtypes('float').notna().sum(axis=1) >= min_sum

        return data[bool_array]

    def _filter_for_valid_vals_in_one_exp_group_perc(self, data, percentage):


        '''Filter for valid values using at least x % data completenes in each experimental group'''

        columns_name = self.data.select_dtypes('float').columns.unique()
        
        indices = []
        
        for col in columns_name:
            non_na_indices = self.data[self.data[col].isna().sum(axis=1)/len(self.data[col].columns) >= percentage].index
            indices.append(non_na_indices)
        indices = list(set(indices))
        self.data = self.data[self.data.index.isin(indices)].reset_index(drop = True)
        
        return self.data


class Preprocessor:
    
    """
    This class applies preprocessing steps to DIANN data
    """
    
    def __init__(self):
        self.data = None
        self._preprocessing = Preprocessing()
    
    def _process(self, data):
        
        self.data = data.copy()
        
        # change the datatypes
        self.data = self._preprocessing._change_datatypes(self.data)
        
        # log2 transform data
        self.data = self._preprocessing._log2_transform(self.data)
        
        # filter for valid values in at least one experimental group
        self.data = self._preprocessing._filter_for_valid_vals_in_one_exp_group(self.data)
        
        # impute genes data
        self.data = self._preprocessing._impute_genes_with_protein_ids(self.data)
        
        # impute data based on normal distribution
        self.data = self._preprocessing._impute_based_on_gaussian(self.data)
        
        return self.data
    
    def _change_dtypes(self, data):
        
        self.data = data.copy()
        
        # change the datatypes
        self.data = self._preprocessing._change_datatypes(self.data)
        
        return self.data