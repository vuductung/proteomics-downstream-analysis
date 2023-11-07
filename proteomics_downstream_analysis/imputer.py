from numpy.random import rand
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
import pandas as pd
from proteomics_downstream_analysis.utils import float_string_split

class Imputer:

    def __init__(self):
        pass

    def fill_dummy_values(self, scaling_factor):

        dummy_data = self.data.copy()

        for col in dummy_data:

            # Get column, column missing values and range
            col = dummy_data[col]
            col_null = col.isnull()
            num_nulls = col_null.sum()
            col_range = col.max() - col.min()

            # Shift and scale dummy values
            dummy_values = (rand(num_nulls) - 2)
            dummy_values = dummy_values * scaling_factor * col_range + col.min()

            # Return dummy values
            col[col_null] = dummy_values

        return dummy_data

    def _impute(self, data, kind='simple', strategy='mean', percentage=0.9, constant=None):
        
        imputed_data = data.select_dtypes(float).copy()
        imputed_data = imputed_data.sort_index(axis=1)

        for col in imputed_data.columns.unique():
            
            # create imputer
            if kind == 'simple':
                imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=constant)

            elif kind == 'knn':
                n_neighbours = int(np.sqrt(imputed_data[col].shape[1]))
                imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbours)

            # Get column, column missing values and range
            denominator = imputed_data[col].notnull().sum(axis=1)
            numerator = imputed_data[col].shape[1]
            null_rows = (denominator/numerator) >= percentage

            # Impute
            to_be_imputed_rows = imputed_data.loc[null_rows, col]

            imputed_data.loc[null_rows, col] = imputer.fit_transform(to_be_imputed_rows.T).T
        
        new_data = pd.concat([data.select_dtypes('string'), imputed_data], axis=1)

        return new_data

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
    
    def _normal_imputation(self, data, axis=0, shift=1.8, width=0.3, seed=42):

        """
        This function imputes missing values in a dataset using a Gaussian distribution.
        The missing values are imputed by random sampling values from a Gaussian distribution with a mean
        of 1.8 standard deviations below the computed mean and a width of 0.3 times the computed standard deviation.
        The function returns a copy of the imputed dataset with the missing values replaced.

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed
        
        axis : int
            axis along which to impute data
            0 : impute along rows
            1 : impute along columns

        shift : float
            number of standard deviations below
            the mean to shift the distribution
        
        width : float
            width of the distribution as a
            fraction of the standard deviation

        seed : int
            seed for reproducibility

        Returns
        -------
        data : pd.DataFrame
            data with imputed values
        """

        # split data
        float_data, string_data = float_string_split(data)

        # impute data
        nan_max = float_data.isnull().sum(axis=axis).max()
        means = float_data.mean(axis=axis).values
        stds = float_data.std(axis=axis).values

        # set a seed or reproducibility
        np.random.seed(seed)

        samples = np.random.normal(loc=means-shift*stds,
                                scale=width*stds,
                                size=(float_data.shape[axis],
                                    float_data.shape[abs(axis-1)]))

        vals = float_data.values
        mask = np.isnan(vals)
        if axis == 0:
            vals[mask] = samples[mask]

        elif axis == 1:
            vals[mask] = samples.T[mask]

        # create new imputed data
        new_data = pd.DataFrame(vals)
        new_data.columns = float_data.columns
        new_data = pd.concat([string_data, new_data], axis=1)
        
        return new_data