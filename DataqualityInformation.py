import pandas as pd
import numpy as np
import textwrap

import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

class DataQualityInformation:

    '''Visualize data quality before data preprocessing'''
    
    def __init__(self):
        pass

    def _missing_vals_lineplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):
    
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

    def _missing_vals_heatmap(self, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):
        
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

    def _missing_vals_barplot(self, n_rows=1, n_cols=1, titles=[''], wrap=8, figsize=(10, 5), savefig=False):
    
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
            sns.barplot(data=mean_na_data, x='variable', y='value', ax=axes, ci='sd', capsize=.3, errwidth=1.5)
            axes.set(ylabel='# missing values', xlabel='')
            axes.set_title(title)
            sns.despine()

        fig.tight_layout()

        if savefig == True:
            fig.savefig('mean_missing_values_barplot.pdf', bbox_inches='tight', transparent=True)

    def _clustermap_pearson_corr(self, figsize=(5, 5), titles=[''], savefig=False):

        for data, title in zip(self.datasets, titles):
            clutermap_data = data.corr()

            # plot the data
            sample_data = pd.Series(index=data.corr().index, data=data.corr().index)
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
    
        cv_data = pd.DataFrame()
        
        for i in data.select_dtypes('float').columns.unique():
            data_trans = data.select_dtypes('float').apply(lambda x: 2**x)
            cv_data[i] = (data_trans[i].std(axis=1)/data_trans[i].mean(axis=1))*100
            
        return cv_data
    
    def _cv_kdeplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):
        
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
    
    def _cv_violinplot(self, n_rows=1, n_cols=1, titles=[''], figsize=(4, 10), savefig=False):
        
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
    
    def _number_ids_barplot(self, n_rows=1, n_cols=1, titles=[''], wrap=8, figsize=(10, 5), savefig=False):
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()

        # plot number of identified proteins
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    
        for dataset, axes, title in zip(datasets, np.array(ax).flatten(), titles):
            id_data = dataset.select_dtypes(float).notna().sum(axis=0)
            id_data.index = [textwrap.fill(i, wrap) for i in id_data.index]
            sns.barplot(x=id_data.index, y=id_data.values, ci='sd', capsize=.3, errwidth=1.5, ax=axes,color='skyblue')
            axes.set(ylabel='# identifications')
            axes.set_title(title)
            sns.despine()
            
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('number_of_ids_barplot.pdf', bbox_inches='tight', transparent=True)
 