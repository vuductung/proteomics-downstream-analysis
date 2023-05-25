import pandas as pd

class MultiLevelData():

    def __init__(self, filepath):
        pass
    
    def multi_level_data(self, annotation, multilevel_cols):

        data_copy = self.data.copy()
        string_cols = data_copy.select_dtypes('string').columns.to_list()
        cols = annotation[multilevel_cols]
        cols_data = pd.DataFrame()

        for cols_name in cols.columns.unique():
            cols_data[cols_name] = string_cols + cols[cols_name].to_list()

        multi_cols = pd.MultiIndex.from_frame(cols_data)
        data_copy.columns = multi_cols
        self.multilevel_data = data_copy.copy()

        return data_copy

    def single_level_data(self, level, data=None):

        '''get one level data'''
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

        '''get the mean of a one level data'''

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