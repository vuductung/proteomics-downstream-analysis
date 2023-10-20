import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

import textwrap
import numpy as np
import pandas as pd


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .utils import is_jupyter_notebook

import plotly.io as pio

class Visualizer:
    """ Visualizer class for plotting."""

    def __init__(self):
        pass

    def volcano_plot(self, n_rows, n_cols, gene_list=None, figsize = (8,8), savefig = False, upper_fc_cutoff = 1, lower_fc_cutoff = 1,
                 annot = None, gene_column = 'Genes', qvalue=True):
    
        """Plot a volcano plot.

        Parameters
        ----------
        n_rows : int
            number of rows for the subplot.
        n_cols : int
            number of columns for the subplot.
        gene_list : list
            name of genes in a list for annotation in volcano plot (Default value = None).
        figsize :
            figure size. (Default value = (8)
        8) :
            
        savefig : boolean
            save the figure. (Default value = False)
        upper_fc_cutoff : int or float
            Set an upper fold change cutoff for annotation (Default value = 1).
        lower_fc_cutoff : int or float
            Set a lower fold change cutoff for annotation (Default value = 1).
        annot : str or Nonetype
            If annot == 'fc_cutoff' genes are annotated by fold change cutoff and gene_list.
            If None no annotation by fold change cutoff is applied, only gene_list. (Default value = None)
        gene_column : string
            What column to use the annotation with (Default value = 'Genes').
        qvalue : boolean
            If qvalue == True then 5% adjusted p-value cutoff is used.
            Else, 5% p-value cutoff is used (Default value = True).

        Returns
        -------

        
        """
        fig, ax = plt.subplots(n_rows, n_cols, figsize = figsize, layout='tight')

        for i, axes in zip(self.fc_data.select_dtypes('float').columns, np.array(ax).flatten()):
            
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
                texts = [axes.text(self.fc_data[i][idx], self.pv_data[i][idx], self.fc_data[gene_column][idx], ha='center', va='center', fontsize=9) for idx in set(indices)]
                adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black'), ax=axes)
            
            else:
                pass
            
            sample_annot = i.split('/')
            x1 = self.fc_data[i].max()
            x2 = self.fc_data[i].min()
            y = self.pv_data[i].max()

            axes.text(x1, y, sample_annot[0], fontsize=12,
                    bbox=dict(boxstyle='round', fc='w', ec='black', alpha=0.3))
            axes.text(x2, y, sample_annot[1],
                    fontsize=12,
                    bbox=dict(boxstyle='round', fc='w', ec='black', alpha=0.3))

            if savefig == True:
                fig.savefig(f'{i.replace("/", "_vs_")}_volcano.pdf', bbox_inches = "tight")

    def sign_prots_plot(self, normalized=False, figsize=(8, 5), savefig=False):

        """Plot number of significant proteins.

        Parameters
        ----------
        normalized : boolean
            If True displays number of significant proteins in % (Default value = False).
        figsize : tuple
            Figure size (Default value = (8,5).
        savefig : boolean
            If True saves figure (Default value = False).
        5) :
            

        Returns
        -------

        
        """

        sign_data = (self.qv_data.select_dtypes('float') < 0.05).sum(axis=0)
        norm_sign_data = ((self.qv_data.select_dtypes('float') < 0.05).sum(axis=0)/self.qv_data.shape[0])*100
        x = [textwrap.fill(i,8)  for i in sign_data.index]
        plt.figure(figsize=figsize)
        if normalized is True:
            sns.barplot(x=x, y=norm_sign_data.values)
            plt.ylabel('number of significant proteins in %')
        
        else:
            sns.barplot(x=x, y=sign_data.values)
            plt.ylabel('number of significant proteins')
            
            sns.despine()

        if savefig is True:
            plt.savefig('sign_prots_plot.pdf', transparent=True, bbox_inches='tight')

    def int_volcano_plot(self, n_rows, n_cols, height, width,
                         annot_genes =[], upper_fc_cutoff = None,
                         lower_fc_cutoff=None, qv_cutoff = False,
                         save=False, filename='int_volcano_plot.html'):

        indices = self.fc_data[self.fc_data['Genes'].isin(annot_genes)].index.to_list()

        fig = make_subplots(rows=n_rows, cols=n_cols)
        
        for i, col_name in enumerate(self.fc_data.select_dtypes(float).columns.tolist(), start=1):
            
            # Create a list of colors based on the qv and fc values
            color_list =[]
            fc_indices = []

            for idx in np.arange(self.fc_data.shape[0]):

                if self.qv_data.loc[idx, col_name] < 0.05 and self.fc_data.loc[idx, col_name] >0:
                    color_list.append('lightcoral')

                elif self.qv_data.loc[idx, col_name] <0.05 and self.fc_data.loc[idx, col_name] <0:
                    color_list.append('cornflowerblue')

                else:
                    color_list.append('lightgrey')
            
            # Create an annotation list based on fold change cutoff
            if qv_cutoff is True:
                if upper_fc_cutoff is not None and lower_fc_cutoff is None:
                    fc_indices = self.fc_data[(self.qv_data[col_name] < 0.05) &
                                            (self.fc_data[col_name] > upper_fc_cutoff)].index.to_list()
                
                elif lower_fc_cutoff is not None and upper_fc_cutoff is None:
                    fc_indices = self.fc_data[(self.qv_data[col_name] < 0.05) &
                                            (self.fc_data[col_name] < lower_fc_cutoff)].index.to_list()
                
                elif lower_fc_cutoff is not None and upper_fc_cutoff is not None:
                    fc_indices = self.fc_data[(self.qv_data[col_name] < 0.05) &
                                            ((self.fc_data[col_name] > upper_fc_cutoff) |
                                            (self.fc_data[col_name] < lower_fc_cutoff))].index.to_list()
                    
            if qv_cutoff is False:
                if upper_fc_cutoff is not None and lower_fc_cutoff is None:
                    fc_indices = self.fc_data[(self.fc_data[col_name] > upper_fc_cutoff)].index.to_list()
                
                elif lower_fc_cutoff is not None and upper_fc_cutoff is None:
                    fc_indices = self.fc_data[(self.fc_data[col_name] < lower_fc_cutoff)].index.to_list()
                
                elif lower_fc_cutoff is not None and upper_fc_cutoff is not None:
                    fc_indices = self.fc_data[((self.fc_data[col_name] > upper_fc_cutoff) |
                                               (self.fc_data[col_name] < lower_fc_cutoff))].index.to_list()
                    
            # Calculate row and col values for the subplots based on i
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1
            
            text = [gene+', '+description
                    for gene, description
                    in zip(self.fc_data['Genes'],
                           self.fc_data['First.Protein.Description'])]
            
            # Add the scatter plots to the subplots layout
            fig.add_trace(go.Scatter(x=self.fc_data[col_name],
                                     y=self.pv_data[col_name],
                                     mode='markers',
                                     marker=dict(color=color_list),
                                     text=text),
                                     row=row,
                                     col=col)

            fig.add_trace(go.Scatter(x=self.fc_data.loc[indices + fc_indices, col_name],
                                     y=self.pv_data.loc[indices + fc_indices, col_name],
                                     mode='markers',
                                     marker=dict(color='lightblue')),
                                     row=row,
                                     col=col)

            # Add annotations to the subplots
            for idx in (indices + fc_indices):
                fig.add_annotation(
                    dict(
                        x=self.fc_data.loc[idx, col_name],
                        y=self.pv_data.loc[idx, col_name],
                        text=self.fc_data.loc[idx, 'Genes'],
                        font={'color': 'black', 'size': 12},
                        xref=f'x{i}',  # use proper xref
                        yref=f'y{i}',  # use proper yref
                    ),
                    row=row,
                    col=col
                )
            
            y_max = self.pv_data[col_name].max()
            x_max = self.fc_data[col_name].max()
            x_min = self.fc_data[col_name].min()

            for coord, sample in zip([x_max, x_min], col_name.split('/')):
                fig.add_annotation(
                    dict(x=coord,
                        y=y_max,
                        text=sample,
                        font={'color': 'black', 'size': 16},
                        showarrow=False,
                        bgcolor='rgba(245, 222, 179, 0.2)', 
                        bordercolor="black", 
                        borderwidth=1,
                        borderpad=3,  
                        ),
                row=row,
                col=col    
                )

            fig.update_xaxes(title_text=f'log2 fold change')
            fig.update_yaxes(title_text=f'-log10 p-value')

        # Update the layout and traces
        fig.update_layout(template='simple_white',
                          height=height,
                          width=width,
                          hoverlabel=dict(
            bgcolor="white",
            font_size=16)
                        )
        fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
        if save == True:
            # Save the plot as an HTML file
            pio.write_html(fig, f'{filename}')

        # Show the plot
        if is_jupyter_notebook():
            fig.show()
        else:
            st.plotly_chart(fig, use_container_width=True)