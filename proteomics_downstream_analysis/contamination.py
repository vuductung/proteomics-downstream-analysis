import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

class ContaminationAnalysis():

    """
    This class contains methods for detecting and removing contamination in a 2D dataframe.
    """
    
    def __init__(self):
        pass
    
    def sort_by_column_names(self, data):

        # sort data by column names
        string_data = data.select_dtypes('string')
        float_data = data.select_dtypes(float)

        float_data = float_data.sort_index(axis=1)
        data = string_data.join(float_data)

        return data

    def robust_zscore(self, data):

        # get the robust zscore 
        data = data.select_dtypes('float')
        median = data.median(axis=1).values.reshape(-1,1)
        mad = stats.median_abs_deviation(data, axis=1, nan_policy='omit').reshape(-1,1)
        robust_zscores = 0.6745*(data-median)/mad
        return robust_zscores

    def number_of_protein_outliers(self, array):

        # compute the number of outliers for each protein
        above = (array > 3.5).sum(axis=0)
        below = (array < -3.5).sum(axis=0)

        protein_outliers = above + below

        return protein_outliers.values

    def upper_limit(self, protein_outliers):

        # compute the upper limit for outliers
        q1 = np.percentile(protein_outliers, 25)
        q3 = np.percentile(protein_outliers, 75)
        iqr = q3-q1
        upper_lim = q3 + 1.5*iqr

        return upper_lim

    def outliers(self, upper_lim, protein_outliers):

        # get the mask to filter outliers
        mask = np.greater(protein_outliers, upper_lim)

        return mask

    def outlier_plot(self, data, plot='bar', kind='zscore', panel=None, contam_type='RBC'):

        groups = data.select_dtypes(float).columns.unique()
        len_groups = len(groups)
        fig, ax = plt.subplots(1, len_groups, figsize=(20, 5))
        
        if kind == 'zscore':
        
            output = [self.compute_zscore_outliers(data[i])[1:3] for i in groups]

        elif kind == 'contamination':
            
            output = [self.compute_contamination_outlier(data[['Genes',i]], panel, contam_type)[:2] for i in groups]

        elif kind == 'missing values':

            output = [self.compute_missing_values_outlier(data[i])[:2] for i in groups]

        # plot the data
        for values, axes, group in zip(output, ax.flat, groups):
            outliers = values[0]
            upper_lim = values[1]
            
            if plot == 'bar':
                sns.barplot(x=[*range(len(outliers))],
                            y=outliers,
                            color='lightgrey', 
                            ax= axes)
            
                axes.set_title(f'{group}')
                axes.set_ylabel('outlier frequency')
                axes.axhline(y = upper_lim, color = 'red', linestyle = '--')
            
            if plot == 'hist':
                sns.histplot(x=outliers,
                            color='lightgrey', 
                            ax= axes)
            
                axes.set_title(f'{group}')
                axes.set_ylabel('outlier frequency')
                axes.axvline(x = upper_lim, color = 'red', linestyle = '--')
                
        fig.tight_layout()

    def compute_zscore_outliers(self, data):

        # sort data
        data = self.sort_by_column_names(data)

        # calculate robus zscores
        robust_zscores = self.robust_zscore(data)

        # get the protein outliers
        protein_outliers = self.number_of_protein_outliers(robust_zscores)

        # get the upper limit (boxplot)
        upper_lim = self.upper_limit(protein_outliers)

        # get the mask to filter outliers
        mask = self.outliers(upper_lim, protein_outliers)
        
        return robust_zscores, protein_outliers, upper_lim, mask

    def zscore_outlier(self, data, experimental=True, remove=False):
        
        # sort data
        data = self.sort_by_column_names(data)

        # compute the robust zscore to find outliers
        # within experimental group or all data
        if experimental == True:

            master_mask = np.array([], dtype=bool)  

            for i in data.select_dtypes(float).columns.unique():
                _, _, _, mask = self.compute_zscore_outliers(data[i])
                master_mask = np.concatenate((master_mask, mask))

        else:

            _, _, _, master_mask = self.compute_zscore_outliers(data)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes('string').columns)

        if remove == True:

            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]

            return data
        
        else:
            return np.where(master_mask)[0] + number_of_string_cols
        
    def compute_rbc_total_ratio(self, data, panel, contam_type='RBC'):

        # calculat the RBC to total protein ratio
        total = data.sum(axis=0, numeric_only=True)
        contam = panel[panel['Type'] == contam_type]['Gene names'].tolist()
        rbc_sum = data[data['Genes'].isin(contam)].sum(axis=0,
                                                numeric_only=True)
        rbc_total_ratio = rbc_sum/total

        return rbc_total_ratio.values
    
    def compute_contamination_outlier(self, data, panel, contam_type='RBC'):
            
            # sort data
            data = self.sort_by_column_names(data)

            # calculate rbc to total protein ratio
            rbc_total_ratio = self.compute_rbc_total_ratio(data, panel, contam_type)

            # get the upper limit (boxplot)
            upper_lim = self.upper_limit(rbc_total_ratio)

            # get the mask to filter outliers
            mask = self.outliers(upper_lim, rbc_total_ratio)

            return rbc_total_ratio, upper_lim, mask 
    
    def contamination_outlier(self, data, panel, contam_type='RBC', remove=False, experimental=True):

        # compute contamination outliers
        if experimental == True:
            master_mask = np.array([], dtype=bool)  
            for i in data.select_dtypes(float).columns.unique():
                _, _, mask = self.compute_contamination_outlier(data[['Genes', i]], panel, contam_type)
                master_mask = np.concatenate((master_mask, mask))

        else:
            _, _, master_mask = self.compute_contamination_outlier(data, panel, contam_type)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes('string').columns)

        if remove == True:
            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]
            return data
        
        else:
            return np.where(master_mask)[0] + number_of_string_cols
        
    def contamination_outlier_plot(self, data, panel, experimental=True, type='RBC'):

        # compute the robust zscore to find outliers
        if experimental == True:

            groups = data.select_dtypes(float).columns.unique()
            len_groups = len(groups)
            fig, ax = plt.subplots(1, len_groups, figsize=(20, 5))

            for group, axes in zip(groups, ax.flat):
                rbc_total_ratio, upper_lim, _  = self.compute_contamination_outlier(data[group], panel, type)

                # plot the data
                sns.barplot(x=[*range(len(rbc_total_ratio))],
                            y=rbc_total_ratio,
                            color='lightgrey', 
                            ax= axes)
                
                axes.set_ylabel('outlier frequency')
                axes.axhline(y = upper_lim, color = 'red', linestyle = '--')
            
            fig.tight_layout()

    def count_missing_values(self, data):

        nan_values = data.select_dtypes(float).isna().sum(axis=0).values

        return nan_values
    
    def compute_missing_values_outlier(self, data):

        # sort data
        data = self.sort_by_column_names(data)

        # count missing values
        nan_values = self.count_missing_values(data)

        # get the upper limit (boxplot)
        upper_lim = self.upper_limit(nan_values)

        # get the mask to filter outliers
        mask = self.outliers(upper_lim, nan_values)

        return nan_values, upper_lim, mask
    
    def missing_values_outlier(self, data, experimental=True, remove=False):

        if experimental == True:
            master_mask = np.array([], dtype=bool)  
            for i in data.select_dtypes(float).columns.unique():
                _, _, mask = self.compute_missing_values_outlier(data[i])
                master_mask = np.concatenate((master_mask, mask))
        
        else:
             _, _, master_mask = self.compute_missing_values_outlier(data)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes('string').columns)

        if remove == True:
            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]
            return data
        
        else:
            return np.where(master_mask)[0] + number_of_string_cols

    def missing_values_outlier_plot(self, data, experimental):

        # compute the robust zscore to find outliers
        if experimental == True:

            groups = data.select_dtypes(float).columns.unique()
            len_groups = len(groups)
            fig, ax = plt.subplots(1, len_groups, figsize=(20, 5))

            for group, axes in zip(groups, ax.flat):
                nan_values, upper_lim, _  = self.compute_missing_values_outlier(data[group])

                # plot the data
                sns.barplot(x=[*range(len(nan_values))],
                            y=nan_values,
                            color='lightgrey', 
                            ax= axes)
                
                axes.set_ylabel('outlier frequency')
                axes.axhline(y = upper_lim, color = 'red', linestyle = '--')
            
            fig.tight_layout()

    def outlier(self, data, kind='zscore', remove=False, panel=None, contam_type='RBC'):

        '''
        docstring
        An outlier algorithm using the z-score and the IQR to detect outliers

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed

        kind : string
            what kind of outlier detection algorithm to use. 'zscore' for 
            zscore algorithm, 'contamination' to detect e.g. RBC contamination
            and 'missing values' for detecting outliers based on the numbers
            of missing values

        remove : boolean
            if True removes outliers from dataset

        panel: pd.DataFrame
            a panel that includes the kind of contamination and the Protein.Ids
            of each contamination

        type = string
            What kind of contamination should be used e.g. 'RBC' for red blood
            cell

        Returns
        -------
        
        array1: np.array
            outliers with their index (excluding the string columns)
        array2 : np.array
            outliers with boolean values (True is an outlier and False
            is inlier)

        or

        data : pd.DataFrame
            data without outliers

        '''
        
        # sort data
        data = self.sort_by_column_names(data)

        master_mask = np.array([], dtype=bool)  

        for i in data.select_dtypes(float).columns.unique():

            if kind == 'zscore':
                _, _, _, mask = self.compute_zscore_outliers(data[i])
                master_mask = np.concatenate((master_mask, mask))

            elif kind == 'contamination':
                _, _, mask = self.compute_contamination_outlier(data[['Genes', i]], panel, contam_type)
                master_mask = np.concatenate((master_mask, mask))

            elif kind == 'missing values':
                _, _, mask = self.compute_missing_values_outlier(data[i])
                master_mask = np.concatenate((master_mask, mask))

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes('string').columns)

        if remove == True:

            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]

            return data
        
        else:
            return np.where(master_mask)[0], master_mask