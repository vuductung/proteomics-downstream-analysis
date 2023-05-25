import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

import numpy as np
import pandas as pd


class Visualizer:

    def __init__(self):
        pass

    def volcano_plot(self, gene_list,figsize = (8,8), savefig = False, upper_fc_cutoff = 1, lower_fc_cutoff = 1,
                 annot = None, gene_column = 'Genes', hline = False, vline = False, qvalue=False):
    
        '''
        Plot a volcano plot 
        gene_list = annotate the datapoints with Gene names in gene list
        upper_fc_cutoff = 
        annot = use conditions to make annotations of Gene names to datapoints ("fc_cutoff", None)
        upper_fc_cutoff = use upper fc cutoff, draws a line at cutoff and annotates datapoints that cross fc cutoff
        lower_fc_cutoff = use lower fc cutoff, draws a line at cutoff and annotates datapoints that cross fc cutoff
        qvalue = True when qvalue is used for significance cutoff
        '''

        for i in self.fc_data.select_dtypes('float').columns:
            
            # plot scatterplot with 5% FDR coloring
            if qvalue == True:
                plt.figure(figsize = figsize)
                sns.scatterplot(x = self.fc_data[i], y = self.pv_data[i], color = 'lightgrey', alpha = 0.5,)
                sns.scatterplot(x = self.fc_data[self.qv_data.select_dtypes('float') < 0.05][i],
                                y = self.pv_data[self.qv_data.select_dtypes('float') < 0.05][i], color = 'lightcoral')
                sns.scatterplot(x = self.fc_data[(self.qv_data.select_dtypes('float') < 0.05) & (self.fc_data.select_dtypes('float') < 0)][i],
                                y = self.pv_data[(self.qv_data.select_dtypes('float') < 0.05) & (self.fc_data.select_dtypes('float') < 0)][i], color = 'cornflowerblue')
                # set q-value threshold
                idx_threshold = self.qv_data[self.qv_data[i] < 0.05][i].sort_values(ascending=False).index[0]
                threshold = self.pv_data.loc[idx_threshold][i]
                plt.axhline(threshold,ls = '--', color = 'lightgrey')
                
            else:
                plt.figure(figsize = figsize)
                sns.scatterplot(x = self.fc_data[i], y = self.pv_data[i], color = 'lightgrey', alpha = 0.5,)
                sns.scatterplot(x = self.fc_data[self.pv_data.select_dtypes('float') > 1.3][i],
                                y = self.pv_data[self.pv_data.select_dtypes('float') > 1.3][i], color = sns.color_palette("deep", 20)[1])
                sns.scatterplot(x = self.fc_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i],
                                y = self.pv_data[(self.pv_data.select_dtypes('float') > 1.3) & (self.fc_data.select_dtypes('float') < 0)][i], color = sns.color_palette("deep", 20)[2])
                # set p-value threshold 
                idx_threshold = self.pv_data[self.pv_data[i] > 1.3][i].sort_values(ascending=True).index[0]
                plt.axhline(1.3,ls = '--', color = 'lightgrey')
            
            # Gene annotation
            if annot == 'fc_cutoff' and qvalue == True:
                indices = self.fc_data[((self.fc_data[i] < lower_fc_cutoff) | (self.fc_data[i] > upper_fc_cutoff)) & (self.qv_data[i] <0.05)].index
                plt.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                plt.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')
                
            elif annot == 'fc_cutoff' and qvalue == False:
                indices = self.fc_data[((self.fc_data[i] < lower_fc_cutoff) | (self.fc_data[i] > upper_fc_cutoff)) & (self.pv_data[i] >1.3)].index
                plt.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                plt.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')

            elif annot == None: 
                indices = self.fc_data[(self.fc_data[gene_column].isin(gene_list)) & (self.pv_data[i]>1.3)].index

            sns.scatterplot(x = self.fc_data.loc[indices][i],
                            y = self.pv_data.loc[indices][i], color = 'lightblue')
            sns.despine()

            # annoation 

            texts = [plt.text(self.fc_data[i][idx],self.pv_data[i][idx], self.fc_data[gene_column][idx], ha='center', va='center', fontsize=9) for idx in indices]
            adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black'))

            plt.xlabel(f'log2 fold change ({i})')
            plt.ylabel(f'-log10 p-value ({i})')
            
            if annot == 'fc_cutoff' and vline == True and hline == False:
                plt.axhline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                plt.axhline(upper_fc_cutoff, ls = '--', color = 'lightgrey')
            
                plt.show()
            
            
            if annot == 'fc_cutoff' and vline == True and hline == True:
                plt.axhline(1, ls = '--', color = 'lightgrey')
                plt.axvline(upper_fc_cutoff, ls = '--', color = 'lightgrey')             
                plt.axvline(lower_fc_cutoff, ls = '--', color = 'lightgrey')
                
                plt.show()

            if savefig == True:
                plt.savefig(f'{i.replace("/", "_vs_")}_volcano.pdf', bbox_inches = "tight")