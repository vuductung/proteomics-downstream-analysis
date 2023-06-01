import pandas as pd

class MultiLevelData():
    """ This class encapsulates data wrangling methods for multi level data """

    def __init__(self, filepath):
        pass
    
    def multi_level_data(self, annotation, multilevel_cols):
        
        """
        Get multi level data

        Parameters
        ----------
        annotation : pd.DataFrame
            Annotation data that contains the multilevel columns

        multilevel_cols : list
            List of column names to be used for multilevel data

        Returns
        -------
        multilevel_data : pd.DataFrame
            Data with multilevel columns
        """

        data_copy = self.data.copy()
        string_cols = data_copy.select_dtypes('string').columns.to_list()
        cols = annotation[multilevel_cols]
        cols_data = pd.DataFrame()

        for cols_name in cols.columns.unique():
            cols_data[cols_name] = string_cols + cols[cols_name].to_list()

        multi_cols = pd.MultiIndex.from_frame(cols_data)
        data_copy.columns = multi_cols
        self.multilevel_data = data_copy.copy()

        return self.multilevel_data

    def single_level_data(self, level, data=None):

        """
        Get a single level column data from multilevel data

        Parameters
        ----------
        level : int
            Level of the column to be extracted

        data : pd.DataFrame
             (Default value = None)

        Returns
        -------
        get_level_data : pd.DataFrame
            Data with single level columns
        """
        if isinstance(data, pd.DataFrame):
            get_level_data = data.copy()
            get_level_data.columns = get_level_data.columns.get_level_values(level)

            return get_level_data

        else:
            get_level_data = self.multilevel_data.copy()
            get_level_data.columns = get_level_data.columns.get_level_values(level)
            self.data = get_level_data.copy()

            return get_level_data

    def multilevel_summary_stat(self, level, output_level, summary_statistic):

        """
        Get the mean of a single level data

        Parameters
        ----------
        level : int
            Level of the column to calculate the statistic from

        output_level : int
            Level of the column to be used as the output
            
        summary_statistic : str
            Summary statistic to be calculated. Either 'mean' or 'median'

        Returns
        -------
        summary_data : pd.DataFrame
            Data with the summary statistic and single level columns
        """

        data_copy = self.multilevel_data.select_dtypes('float').copy()

        if summary_statistic == 'mean':
            summary_data = data_copy.groupby(level=[level, output_level], axis=1).mean()

        elif summary_statistic == 'median':
            summary_data = data_copy.groupby(level=[level, output_level], axis=1).median()

        summary_data = self.single_level_data(output_level, summary_data)
        string_data = self.single_level_data(output_level, self.multilevel_data.select_dtypes('string'))
        summary_data = string_data.merge(summary_data, left_index=True, right_index=True)
        self.summary_data = summary_data.copy()

        return summary_data