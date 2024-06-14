import pandas as pd
import numpy as np
import textwrap

import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from adjustText import adjust_text

class DataQualityInformation:

    """Visualize data quality before data preprocessing"""
    
    def __init__(self):
        pass

    def _missing_vals_heatmap(self, n_rows=1, n_cols=1, titles=[''], figsize=(10, 5), savefig=False):

        """
        Plot the number of missing values per column as a heatmap

        Parameters
        ----------
        n_rows : int
            Number of rows in the subplot
             (Default value = 1)

        n_cols : int
            Number of columns in the subplot
             (Default value = 1)

        titles : list
            List of titles for each subplot
             (Default value = [''])

        figsize : tuple
            Figure size
             (Default value = (10, 5) 
            
        savefig : bool
            Save figure
             (Default value = False)

        Returns
        -------
        seaborn.heatmap
        """
        
        # Create a single dataset list and a single title list when no subplots
        if (n_rows ==1) and (n_cols == 1):
            datasets = [self.data.select_dtypes(float)]

        else:
            datasets = [i.select_dtypes(float) for i in self.datasets]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for data, axes, title in zip(datasets, np.array(ax).flatten(), titles):
            sns.heatmap(data.select_dtypes('float').isnull(),
                        cbar=False,
                        ax=axes,
                        xticklabels=False,
                        yticklabels=False,
                        cmap='viridis',
                        rasterized=True)
            axes.set_title(title)
        fig.tight_layout()

        if savefig == True:
            fig.savefig('missing_values_heatmap.pdf', bbox_inches='tight', transparent=True)

    def _missing_vals_barplot(self, n_rows=1, n_cols=1, titles=[''], wrap=8, figsize=(10, 5), savefig=False):
        """
        Plot the number of missing values per column as a barplot

        Parameters
        ----------
        n_rows : int
            Number of rows in the subplot
             (Default value = 1)

        n_cols : int
            Number of columns in the subplot
             (Default value = 1)

        titles : list
            List of titles for each subplot
             (Default value = [''])

        wrap : int
            Number of characters to wrap the x-axis labels
             (Default value = 8)

        figsize :  tuple
            Figure size
             (Default value = (10, 5) 
            
        savefig : bool
            Save figure
             (Default value = False)

        Returns
        -------
        seaborn.barplot
        """
    
        # Create a single dataset list and a single title list when no subplots
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]

        else:
            datasets = self.datasets.copy()

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for data, axes, title in zip(datasets, np.array(ax).flatten(), titles):
            mean_na_data = {key: data[key].isna().sum().values for key in data.select_dtypes('float').columns}
            mean_na_data = pd.DataFrame.from_dict(mean_na_data, orient='index').T.melt().dropna()
            mean_na_data['variable'] = [textwrap.fill(i, wrap) for i in mean_na_data['variable']]
            
            # Use the provided palette if available; otherwise, use Seaborn default color
            sns.barplot(data=mean_na_data, x='variable', y='value', ax=axes, errorbar='sd', capsize=.3, err_kws={'linewidth': 1.5})
            axes.set(ylabel='# missing values', xlabel='')
            axes.set_title(title)
            sns.despine()

        fig.tight_layout()

        if savefig == True:
            fig.savefig('mean_missing_values_barplot.pdf', bbox_inches='tight', transparent=True)


    def _clustermap_pearson_corr(self, figsize=(5, 5), titles=[''], savefig=False):

        """
        Plot the Pearson correlation between the features as a clustermap

        Parameters
        ----------
        figsize : tuple
            Figure size
             (Default value = (5, 5) 

        titles : list
            List of titles for each subplot
             (Default value = [''])

        savefig : bool
            Save figure
             (Default value = False)

        Returns
        -------
        seaborn.clustermap
        """

        if self.datasets:
            datasets = self.datasets.copy()

        else:
            datasets = [self.data]

        for data, title in zip(datasets, titles):
            clutermap_data = data.corr(numeric_only=True)

            # plot the data
            sample_data = pd.Series(index=data.corr(numeric_only=True).index, data=data.corr(numeric_only=True).index)
            lut = dict(zip(sample_data.unique(), sns.color_palette("Paired", int((data.shape[1]-2)/3))))
            row_colors = sample_data.map(lut)
            col_colors = sample_data.map(lut)
            g = sns.clustermap(data=clutermap_data,
                               figsize=figsize,
                               yticklabels=True,
                               xticklabels=True,
                               cmap='viridis',
                               row_colors=row_colors,
                               col_colors=col_colors)

            handles = [Patch(facecolor=lut[name]) for name in lut]
            plt.legend(handles, lut, title='Samples',
                    bbox_to_anchor=(1.4, 0.8), bbox_transform=plt.gcf().transFigure, loc='upper right')
            g.fig.suptitle(title) 
            
            if savefig == True:
                plt.savefig(f'clustermap_pearson_corr_{title}.pdf', bbox_inches='tight', transparent=True)

    def _calculate_coef_var(self, data):

        """
        Calculate the coefficient of variation (CV) for each feature

        Parameters
        ----------
        data : pandas.DataFrame
            data to calculate the CV        

        Returns
        -------
        cv_data : pandas.DataFrame
            CV data
        """
    
        cv_data = pd.DataFrame()
        
        for i in data.select_dtypes('float').columns.unique():
            cv_data[i] = (data[i].std(axis=1)/data[i].mean(axis=1))*100
            
        return cv_data

    def _cv_kdeplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(10, 5), savefig=False):
        """

        Parameters
        ----------
        n_rows :
             (Default value = 1)
        n_cols :
             (Default value = 1)
        titles :
             (Default value = [''])
        figsize :
             (Default value = (10)
        5) :
            
        savefig :
             (Default value = False)

        Returns
        -------

        """
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()

        cv_data = [self._calculate_coef_var(i) for i in datasets]
        cv_data_melt = [i.melt() for i in cv_data]
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for cv_dataset, axes, title in zip(cv_data_melt, np.array(ax).flatten(), titles):
            sns.kdeplot(data=cv_dataset, hue='variable', x='value', ax=axes)
            axes.set(xlabel='CV in %')
            axes.set(ylabel='frequency')
            axes.set_title(title)
            plt.setp(axes, xticks=np.arange(0, cv_dataset['value'].max(), 20))
            sns.despine()
        
        fig.tight_layout()
        
        if savefig == True:
            
            fig.savefig('coef_var_kdeplot.pdf', bbox_inches='tight', transparent=True)
    
    def _cv_violinplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(10, 5), savefig=False):

        """
        Plot the coefficient of variation (CV) for each feature as a violinplot

        Parameters
        ----------
        n_rows : int
            Number of rows
             (Default value = 1)

        n_cols : int
            Number of columns
             (Default value = 1)

        titles : list
            List of titles for each subplot
             (Default value = [''])

        figsize : tuple
            Figure size
             (Default value = (10, 5) 
            
        savefig : bool
            Save figure
             (Default value = False)

        Returns
        -------
        seaborn.violinplot
        """
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]

        else:
            datasets = self.datasets.copy()

        cv_data = [self._calculate_coef_var(i) for i in datasets]
        cv_data_melt = [i.melt() for i in cv_data]
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        for cv_dataset, axes, title in zip(cv_data_melt, np.array(ax).flatten(), titles):
            sns.violinplot(data=cv_dataset, x='variable', y='value', ax=axes)
            axes.set(ylabel='CV in %')
            axes.set(xlabel='')
            axes.set_title(title)
            # plt.setp(axes, yticks=np.arange(0, cv_dataset['value'].max(), 20))
            sns.despine()
    
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('coef_var_violinplot.pdf', bbox_inches='tight', transparent=True)

    def abundance_plot(self, data, top=3):
        
        # calculate the mean log2 intenstiy for each feature
        mean_data = data.mean(axis=1, numeric_only=True)
        x = np.arange(len(mean_data))
        y = sorted(mean_data.values)[::-1]

        # plot data
        sns.scatterplot(x=x, y=y)
        plt.xlabel('Rank')
        plt.ylabel('Abundance')

        # annotate the top features
        idx_sorted = np.argsort(mean_data)[::-1]
        annotation = data['Genes'].squeeze()[idx_sorted].reset_index(drop=True)

        texts = [plt.text(x[idx], y[idx], annotation[idx], ha='center', va='center', fontsize=9) for idx in np.arange(top)]
        adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black'))

        plt.show()

    def number_ids_distplot(self, n_rows=1, n_cols=1, titles=[''], wrap=8, figsize=(10, 5), savefig=False):

        """
        Plot the number of identified proteins for each sample as a barplot

        Parameters
        ----------
        n_rows : int
            Number of rows
             (Default value = 1)

        n_cols : int
            Number of columns
             (Default value = 1)

        titles : list
            List of titles for each subplot
             (Default value = [''])

        wrap : int
            Number of characters to wrap the xticklabels
             (Default value = 8)

        figsize : tuple
            Figure size
             (Default value = (10, 5)
            
        savefig : bool
             (Default value = False)

        Returns
        -------
        seaborn.barplot
        """
        
        if (n_rows == 1) and (n_cols == 1):
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()

        # plot number of identified proteins
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    
        for dataset, axes, title in zip(datasets, np.array(ax).flatten(), titles):
            id_data = dataset.select_dtypes(float).notna().sum(axis=0)
            id_data.index = [textwrap.fill(i, wrap) for i in id_data.index]
            
            sns.swarmplot(y=id_data.values,
                          x=id_data.index,
                          color='black',
                          size=3,
                          ax=axes)

            sns.violinplot(y=id_data.values,
                           x=id_data.index,
                           ax=axes)

            axes.set(ylabel='# identifications')
            axes.set_title(title)
            sns.despine()
            
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('number_of_ids_distplot.pdf', bbox_inches='tight', transparent=True)

    def data_quality_info_plot(self, n_rows=1, n_cols=1, titles=[''], figsize=(10, 5), savefig=False):
        """
        Plot the data quality information

        Parameters
        ----------
        n_rows : int
            Number of rows
             (Default value = 1)
        n_cols : int
            Number of columns
             (Default value = 1)
        titles : list
            List of titles for each subplot
             (Default value = [''])
        figsize : tuple
            Figure size
             (Default value = (10, 5)
            
        savefig : bool
             Save figure
             (Default value = False)

        Returns
        -------

        """
        # show missing values
        self._missing_vals_heatmap(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)
        self._missing_vals_barplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)

        # show reproducibility between replicates
        self._clustermap_pearson_corr(titles=titles, figsize=figsize, savefig=savefig)
        self._cv_kdeplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)
        self._cv_violinplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)

        # show proteome depth
        self.number_ids_distplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)

    
    def calc_depth(self, data, normalize=None):
    
        '''
        depth = number of valid values in each sample
        '''

        # calculate depth
        float_data = data.select_dtypes(float)

        if normalize is None:
            depth = float_data.notnull().sum()
        
        elif normalize == 'percent':
            depth = float_data.notnull().mean() * 100
        
        # rename data
        depth_renames = {'index': 'groups', 
                        0: 'depth'}

        data_depth = pd.DataFrame(depth).reset_index().rename(columns=depth_renames)

        return data_depth

    def calc_completeness(self, data, normalize=None):
        
        '''
        completeness = number of valid values in each feature
        '''

        # calculate completeness
        float_data = data.select_dtypes(float)
        groups = float_data.columns.unique()
        data_completeness = pd.DataFrame()

        if normalize is None:
            for group in groups:    
                data_completeness[group] = float_data[group].notnull().sum(axis=1)
        
        if normalize == 'percent':
            for group in groups:
                data_completeness[group] = float_data[group].notnull().mean(axis=1) * 100

        data_completeness = data_completeness.melt()
        data_completeness.columns = ['groups', 'completeness']

        return  data_completeness

    def depth_dist(self, data, figsize=(10, 4), normalize=None):

        depth = self.calc_depth(data, normalize)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        sns.histplot(data=depth,
                x='depth',
                hue='groups',
                kde=True,
                alpha=0.5,
                ax =ax[0],
                stat='percent')

        sns.rugplot(data=depth,
                x='depth',
                hue='groups',
                alpha=0.5,
                ax =ax[0])

        sns.ecdfplot(data=depth,
                x='depth',
                hue='groups',
                ax =ax[1], 
                complementary=True)
        ax[1].axhline(y=0.5, color='lightgrey', linestyle='--')
        ax[1].axhline(y=0.8, color='lightgrey', linestyle='--')
        fig.tight_layout()

    def completeness_dist(self, data, figsize=(10, 4), normalize=None):

        completeness = self.calc_completeness(data, normalize)

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        sns.histplot(data=completeness,
                x='completeness',
                hue='groups',
                kde=True,
                alpha=0.5,
                ax =ax[0],
                stat='percent')

        sns.rugplot(data=completeness,
                x='completeness',
                hue='groups',
                alpha=0.5,
                ax =ax[0])

        sns.ecdfplot(data=completeness,
                x='completeness',
                hue='groups',
                ax =ax[1], 
                complementary=True,
                stat='count')
        ax[1].axhline(y=0.5, color='lightgrey', linestyle='--')
        ax[1].axhline(y=0.8, color='lightgrey', linestyle='--')
        fig.tight_layout()

    def depth_completeness_dist(self, data, figsize, normalize):
        
        # plot completeness and depth of data
        self.depth_dist(data, figsize, normalize)
        self.completeness_dist(data, figsize, normalize)


    def completeness_plot(self, data):

        completeness = data.select_dtypes(float).notnull().sum(axis=1).values
        completeness_normalized = np.array(completeness) / np.max(completeness)
        sns.ecdfplot(completeness_normalized, stat='count', complementary=True)
        plt.axhline(len(data.dropna()), color='lightgrey', linestyle='--', label='100% completeness')
        plt.axhline(0.5 * len(data), color='grey', linestyle='--', label='50% completeness')
        plt.legend(loc='best')

        plt.ylabel('Number of identified\nproteins')