from preprocessing import Preprocessing

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