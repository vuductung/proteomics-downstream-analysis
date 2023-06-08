import pandas as pd
import numpy as np
import textwrap

import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import streamlit as st
from .utils import is_jupyter_notebook

class DataQualityInformation:

    """Visualize data quality before data preprocessing"""
    
    def __init__(self):
        pass

    def _missing_vals_lineplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(10, 5), savefig=False):
        
        """
        Plot the number of missing values per column as a lineplot

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
        seaborn.lineplot
        """
    
        # Create a single dataset list and a single title list when no subplots required
        if n_rows == 1 and n_cols == 1:
            datasets_new_cols = [self.data.select_dtypes('float')]
            datasets_new_cols[0].columns = [str(i) for i in np.arange(datasets_new_cols[0].shape[1])]

        else:
            datasets_new_cols = []
            for data in self.datasets:
                new_dataset = data.select_dtypes('float')
                new_dataset.columns = [str(i) for i in np.arange(new_dataset.shape[1])]
                datasets_new_cols.append(new_dataset)

        # Plot the na values
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        for data, axes, name in zip(datasets_new_cols, np.array(ax).flatten(), titles):
            sns.lineplot(y=data.isna().sum().values, x=data.isna().sum().index, ax=axes, marker='o')
            axes.set(ylabel='missing values')
            axes.set_title(name)
            sns.despine()
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('missing_values_lineplot.pdf', bbox_inches='tight', transparent=True)

        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 

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
        if n_rows ==1 and n_cols == 1:
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

        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 

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
            sns.barplot(data=mean_na_data, x='variable', y='value', ax=axes, errorbar='sd', capsize=.3, errwidth=1.5)
            axes.set(ylabel='# missing values', xlabel='')
            axes.set_title(title)
            sns.despine()

        fig.tight_layout()

        if savefig == True:
            fig.savefig('mean_missing_values_barplot.pdf', bbox_inches='tight', transparent=True)

        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 

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
            
            # if is_jupyter_notebook():
            #     plt.show()
            # else:
            #     st.pyplot(g)

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

        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 
    
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
        
        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 
    
    def _number_ids_barplot(self, n_rows=1, n_cols=1, titles=[''], wrap=8, figsize=(10, 5), savefig=False):

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
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()

        # plot number of identified proteins
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    
        for dataset, axes, title in zip(datasets, np.array(ax).flatten(), titles):
            id_data = dataset.select_dtypes(float).notna().sum(axis=0)
            id_data.index = [textwrap.fill(i, wrap) for i in id_data.index]
            sns.barplot(x=id_data.index, y=id_data.values, errorbar='sd', capsize=.3, errwidth=1.5, ax=axes,color='skyblue')
            axes.set(ylabel='# identifications')
            axes.set_title(title)
            sns.despine()
            
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('number_of_ids_barplot.pdf', bbox_inches='tight', transparent=True)

        # if is_jupyter_notebook():
        #     fig.show()
        # else:
        #     st.pyplot(fig) 

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
        self._missing_vals_lineplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)
        self._missing_vals_heatmap(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)
        self._missing_vals_barplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)

        # show reproducibility between replicates
        self._clustermap_pearson_corr(titles=titles, figsize=figsize, savefig=savefig)
        self._cv_kdeplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)
        self._cv_violinplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)

        # show proteome depth
        self._number_ids_barplot(n_rows=n_rows, n_cols=n_cols, titles=titles, figsize=figsize, savefig=savefig)