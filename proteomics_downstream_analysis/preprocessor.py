from proteomics_downstream_analysis.preprocessing import Preprocessing

class Preprocessor:
    
    """
    This class applies preprocessing steps to DIANN data
    """
    
    def __init__(self):
        self.data = None
        self._preprocessing = Preprocessing()
    
    def _process(self, data):
                
        # change the datatypes
        data = self._preprocessing._change_datatypes(data)
        
        # log2 transform data
        data = self._preprocessing._log2_transform(data)
        
        # filter for valid values in at least one experimental group
        data = self._preprocessing._filter_for_valid_vals_in_one_exp_group(data)
        
        # impute genes data
        data = self._preprocessing._impute_genes_with_protein_ids(data)
        
        # impute data based on normal distribution
        data = self._preprocessing._impute_based_on_gaussian(data)
                
        return data
    
    def _hybrid_process(self, data, completeness, percentage, strategy, kind, constant=None):
        
        # change the datatypes
        data = self._preprocessing._change_datatypes(data)
        
        # log2 transform data
        data = self._preprocessing._log2_transform(data)
        
        # filter for valid values in each exp group based on min data completeness
        data = self._preprocessing._filter_for_valid_vals_in_one_exp_group_perc(data, completeness)

        # impute genes data
        data = self._preprocessing._impute_genes_with_protein_ids(data)
        
        # reorder data
        string_cols = data.select_dtypes('string').columns.unique().tolist()
        float_cols = data.select_dtypes('float').columns.unique().tolist()

        data = data[string_cols + float_cols]

        # impute data based on random missingness
        data = self._preprocessing._impute(data, kind, strategy, percentage, constant)

        # impute data based on systematic missingness (gaussian distribution)
        data = self._preprocessing._impute_based_on_gaussian(data)

        return data

    def _simple_process(self, data, completeness):

         # change the datatypes
        data = self._preprocessing._change_datatypes(data)
        
        # log2 transform data
        data = self._preprocessing._log2_transform(data)
        
        # filter for valid values in each exp group based on min data completeness
        data = self._preprocessing._filter_for_valid_vals_in_one_exp_group_perc(data, completeness)

        # impute genes data
        data = self._preprocessing._impute_genes_with_protein_ids(data)

        return data

    def _change_dtypes(self, data):
                
        # change the datatypes
        data = self._preprocessing._change_datatypes(data)
        
        return data