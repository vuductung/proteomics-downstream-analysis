import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from proteomics_downstream_analysis.dataqualityinformation import DataQualityInformation

class MultiLevelData(DataQualityInformation):
    """ This class encapsulates data wrangling methods for multi level data """

    def __init__(self):

        pass
    
    def get_multi_level_data(self, annotation, multilevel_cols):
        
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

        elif summary_statistic == 'sd':
            summary_data = data_copy.groupby(level=[level, output_level], axis=1).std()

        summary_data = self.single_level_data(output_level, summary_data)
        string_data = self.single_level_data(output_level, self.multilevel_data.select_dtypes('string'))
        summary_data = string_data.merge(summary_data, left_index=True, right_index=True)
        self.summary_data = summary_data.copy()

        return summary_data
    
    def cv_plot(self, data, figsize=(8, 4)):
        
        # calculate cv
        cv_data = self._calculate_coef_var(data)
        cv_data = cv_data.melt()
        cv_data.columns = ['groups', 'CV in %']

        # plot data
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.kdeplot(cv_data,
                    x='CV in %',
                    hue='groups',
                    ax=axes[0])

        sns.ecdfplot(cv_data,
                    x='CV in %',
                        hue='groups',
                        ax=axes[1])
        fig.tight_layout()

    def pearson_corr_clustermap(self, levels, labels):

        sgl_lvl_iterator = self.multilevel_iterator(levels)
        corr_data = [df.corr(numeric_only=True) for df in sgl_lvl_iterator]

        luts = []
        sa_type_no = 0
        col_palette = sns.color_palette("Set3")
        for df in corr_data:
                luts.append(dict(zip(df.index.unique(), col_palette[sa_type_no:])))
                sa_type_no += len(df.index.unique())

        row_cols = [pd.Series(df.index).map(lut) for df, lut in zip(corr_data, luts)]
        row_cols = pd.concat(row_cols, axis=1)
        row_cols.columns = labels

        cluster_data = corr_data[0].reset_index(drop=True)
        cluster_data.columns = cluster_data.index

        g = sns.clustermap(data=cluster_data,
                        cmap=sns.diverging_palette(220, 20, as_cmap=True),
                        row_colors=row_cols,
                        col_colors=row_cols,
                        yticklabels=False,
                        xticklabels=False)

        handles = [Patch(facecolor=lut[name]) for lut in luts for name in lut]
        labels = [key for lut in luts for key in lut.keys()]
        plt.legend(handles,
                labels, 
                bbox_to_anchor=(1.20, 0.8),
                bbox_transform=plt.gcf().transFigure,
                loc='upper right',
                ncol=2)
        plt.show()

    def multilevel_iterator(self, levels):

        sgl_lvl_iterator = [self.single_level_data(level) for level in levels]

        return sgl_lvl_iterator