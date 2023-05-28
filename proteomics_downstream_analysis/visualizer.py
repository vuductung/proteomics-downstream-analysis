import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

import textwrap
import numpy as np
import pandas as pd


class Visualizer:

    def __init__(self):
        pass

    def volcano_plot(self, n_rows, n_cols, gene_list=None, figsize = (8,8), savefig = False, upper_fc_cutoff = 1, lower_fc_cutoff = 1,
                 annot = None, gene_column = 'Genes', qvalue=True):
    
        ''' Plot a volcano plot '''
        fig, ax = plt.subplots(n_rows, n_cols, figsize = figsize)

        for i, axes in zip(self.fc_data.select_dtypes('float').columns, ax.flat):
            
            if qvalue is True:

                if (self.qv_data[i] < 0.05).sum() == 0:

                    sns.scatterplot(x = self.fc_data[i], y = self.pv_data[i], color = 'lightgrey', alpha = 0.5, ax=axes)
                    sns.scatterplot(x = self.fc_data[self.pv_data.select_dtypes('float') > 1.3][i],
                                    y = self.pv_data[self.pv_data.select_dtypes('float') > 1.3][i], color = sns.color_palette("deep", 20)[1], ax=axes)
                    sns.scatterplot(x = self.fc_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i],
                                    y = self.pv_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i],
                                    color = sns.color_palette("deep", 20)[2], ax=axes)
                    
                    axes.set(xlabel=f'log2 fold change ({i})',
                             ylabel=f'-log10 p-value ({i})')
                    
                    # set p-value threshold 
                    axes.axhline(1.3,ls = '--', color = 'lightgrey')

                    if annot == 'fc_cutoff':
                        indices = self.fc_data[((self.fc_data[i] < lower_fc_cutoff) | 
                                                (self.fc_data[i] > upper_fc_cutoff)) &
                                                (self.pv_data[i] >1.3)].index

                        axes.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                        axes.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')
                
                    else:
                        indices = self.fc_data[self.fc_data[gene_column].isin(gene_list)].index

                else:
                    sns.scatterplot(x = self.fc_data[i], y = self.pv_data[i], color = 'lightgrey', alpha = 0.5, ax=axes)
                    sns.scatterplot(x = self.fc_data[self.qv_data.select_dtypes('float') < 0.05][i],
                                    y = self.pv_data[self.qv_data.select_dtypes('float') < 0.05][i], color = 'lightcoral', ax=axes)
                    sns.scatterplot(x = self.fc_data[(self.qv_data.select_dtypes('float') < 0.05) & (self.fc_data.select_dtypes('float') < 0)][i],
                                    y = self.pv_data[(self.qv_data.select_dtypes('float') < 0.05) & (self.fc_data.select_dtypes('float') < 0)][i],
                                    color = 'cornflowerblue', ax=axes)
                    
                    axes.set(xlabel=f'log2 fold change ({i})',
                             ylabel=f'-log10 p-value ({i})')
                    
                    # set q-value threshold
                    threshold = self.pv_data[self.qv_data[i] < 0.05][i].sort_values().values[0]
                    axes.axhline(threshold, ls = '--', color = 'lightgrey')

                    if annot == 'fc_cutoff':
                        indices = self.fc_data[((self.fc_data[i] < lower_fc_cutoff) |
                                                (self.fc_data[i] > upper_fc_cutoff)) &
                                                (self.qv_data[i] <0.05)].index
                        axes.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                        axes.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')

                    else:
                        indices = self.fc_data[self.fc_data[gene_column].isin(gene_list)].index

            if qvalue is False:
                sns.scatterplot(x = self.fc_data[i], y = self.pv_data[i], color = 'lightgrey', alpha = 0.5, ax=axes)
                sns.scatterplot(x = self.fc_data[self.pv_data.select_dtypes('float') > 1.3][i],
                                y = self.pv_data[self.pv_data.select_dtypes('float') > 1.3][i], color = sns.color_palette("deep", 20)[1], ax=axes)
                sns.scatterplot(x = self.fc_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i],
                                y = self.pv_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i],
                                color = sns.color_palette("deep", 20)[2], ax=axes)
                
                axes.set(xlabel=f'log2 fold change ({i})',
                             ylabel=f'-log10 p-value ({i})')
                
                # set p-value threshold 
                axes.axhline(1.3,ls = '--', color = 'lightgrey')

                if annot == 'fc_cutoff':
                    indices = self.fc_data[((self.fc_data[i] < lower_fc_cutoff) |
                                            (self.fc_data[i] > upper_fc_cutoff) |
                                            (self.fc_data['Genes'].isin(gene_list))) &
                                            (self.pv_data[i] >1.3)].index
                    axes.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                    axes.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')
                
                else:
                    indices = self.fc_data[self.fc_data[gene_column].isin(gene_list)].index

            if gene_list is not None:
                genes_indices = self.fc_data[self.fc_data[gene_column].isin(gene_list)].index
                
                if indices.tolist():
                    indices = indices.tolist() + genes_indices.tolist()
                    

                else:
                    indices = genes_indices.tolist()

            sns.scatterplot(x = self.fc_data.loc[indices][i],
                            y = self.pv_data.loc[indices][i],
                            color = 'lightblue',
                            ax=axes)
            sns.despine()

            # annotation
            if indices:
                texts = [axes.text(self.fc_data[i][idx], self.pv_data[i][idx], self.fc_data[gene_column][idx], ha='center', va='center', fontsize=9) for idx in indices]
                adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black'), ax=axes)
            
            else:
                pass
            
            fig.tight_layout()

            if savefig == True:
                fig.savefig(f'{i.replace("/", "_vs_")}_volcano.pdf', bbox_inches = "tight")


    def sign_prots_barplot(self, n_rows=1, n_cols=1, qvalue=True, normalize=False, figsize=(5,5), wrap =5, savefig=False):
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()
        
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        for data, axes in zip(datasets, np.array(ax).flatten()):

            if qvalue == True:
                numb_sign_prots = [(data[col] <0.05).sum(axis=0) for col in data.select_dtypes('float').columns]
                sign_prots_norm = [(data[col]<0.05).sum(axis=0)/len(data[col].dropna())*100 for col in data.select_dtypes('float').columns]

            else:
                numb_sign_prots = [(data[col] >1.3).sum(axis=0) for col in data.select_dtypes('float').columns]
                sign_prots_norm = [(data[col]>1.3).sum(axis=0)/len(data[col].dropna())*100 for col in data.select_dtypes('float').columns]
                
                comparisons = data.select_dtypes('float').columns.to_list()
                comparisons = [textwrap.fill(i, wrap) for i in comparisons]
                
                if normalize is True:
                        
                    sns.barplot(x=comparisons, y=sign_prots_norm, color='lightgrey', ax=axes)
                    sns.despine()
                    axes.set(ylabel= '# of significant proteins')

                else:
                    sns.barplot(x=comparisons, y=numb_sign_prots, color='lightblue', ax=axes)
                    sns.despine()
                    axes.set(ylabel= 'significant proteins in %')

            fig.tight_layout()
            if savefig == True:
                fig.savefig('significant_prot.pdf', bbox_inches='tight', transparent=True)  