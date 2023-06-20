import numpy as np

class Preprocessing:
    
    """This class encapsulates preprocessing steps"""
    
    def __init__(self):
        pass 
        
    def _change_datatypes(self, data):
        
        """
        change datatypes

        Parameters
        ----------
        data : pd.DataFrame
            data with original datatypes
        Returns
        -------
        data : pd.DataFrame
            data with changed datatypes
        """
        
        for i in data.columns.unique():
            if i in ['Protein.Group', 'Protein.Ids', 'Protein.Names', 'Genes','First.Protein.Description']:
                data[i] = data[i].astype('string')
            else:
                data[i] = data[i].astype('float') 
        return data
    
    def _log2_transform(self, data):
        
        """
        log2 transfrom float data

        Parameters
        ----------
        data : pd.DataFrame
            data with float data

        Returns
        -------
        data : pd.DataFrame
            data with log2 transformed float data
        """

        for i in data.select_dtypes('float').columns.unique():
            data[i] = np.log2(data[i])
        return data
        
    def _filter_for_valid_vals_in_one_exp_group(self, data):
        
        """
        Filter for valid values in at least one experimental group

        Parameters
        ----------
        data : pd.DataFrame
            data with float data
        
        Returns
        -------
        data : pd.DataFrame
            data with valid values in at least one experimental group
        """
            
        columns_name = data.select_dtypes('float').columns.unique()
            
        indices = []
            
        for col in columns_name:
            non_na_indices = data[data[col].isna().sum(axis=1)==0].index.to_list()
            indices += non_na_indices
        indices = list(set(indices))
        data = data[data.index.isin(indices)].reset_index(drop = True)
        
        return data
    
    def _impute_genes_with_protein_ids(self, data):
        
        """
        Imputing missing values in Genes column with the protein IDs

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed

        Returns
        -------
        data : pd.DataFrame
            data with imputed gene values
        """
        imputation_values = [i.split(';')[0] for i in data[data['Genes'].isna()]['Protein.Ids'].values]
        
        for idx, value in zip(data[data['Genes'].isna()]['Protein.Ids'].index, imputation_values):
            data.loc[idx, ['Genes']] = value

        return data
    
    def _impute_based_on_gaussian(self, data):
         
        """
        This function imputes missing values in a dataset using a Gaussian distribution.
        The missing values are imputed by random sampling values from a Gaussian distribution with a mean
        of 3 standard deviations below the computed mean and a width of 0.3 times the computed standard deviation.
        The function returns a copy of the imputed dataset with the missing values replaced.

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed

        Returns
        -------
        data : pd.DataFrame
            data with imputed values
        """
        # select only float data
        float_data = data.select_dtypes('float')
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
        imp_data = data.select_dtypes('string').merge(imp_data, left_index=True, right_index=True)
        data = imp_data.copy()
        
        return data

    def _filter_for_valid_vals_perc(self, data, percentage):
    
        """Filter for valid values using at least x % data completeness

        Parameters
        ----------
        data : pd.DataFrame
            data to be filtered
            
        percentage : float
            percentage of desired data completeness

        Returns
        -------
        data : pd.DataFrame
            data with valid values using at least x % data completeness
        """
    
        col_len = len(data.select_dtypes('float').columns)
        min_sum = np.ceil(col_len * percentage)
        bool_array = data.select_dtypes('float').notna().sum(axis=1) >= min_sum

        return data[bool_array]

    def _filter_for_valid_vals_in_one_exp_group_perc(self, data, percentage):


        """
        Filter for valid values using at least x % data completenes in each experimental group

        Parameters
        ----------
        data : pd.DataFrame
            data to be filtered
            
        percentage : float
            percentage of desired data completeness

        Returns
        -------
        data : pd.DataFrame
            data with valid values using at least x % data completeness in each experimental group
        """

        columns_name = data.select_dtypes('float').columns.unique()
        
        indices = []
        
        for col in columns_name:
            numerator = data[col].notna().sum(axis=1)
            denominator = data[col].shape[1]
            non_na_indices = data[numerator/denominator >= percentage].index.tolist()
            indices += non_na_indices

        indices = list(set(indices))
        data = data[data.index.isin(indices)].reset_index(drop = True)
        
        return data