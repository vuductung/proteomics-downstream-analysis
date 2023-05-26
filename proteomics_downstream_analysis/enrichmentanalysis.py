from collections import Counter
import textwrap
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.associations import dnld_assc
from goatools.semantic import semantic_similarity, TermCounts, \
    get_info_content, resnik_sim, lin_sim, deepest_common_ancestor

import gseapy as gp

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import textwrap

class EnrichmentAnalysis:

    def __init__(self):
        pass

    def array_enrichment_analysis(self, gene_list, organism):

        """ Perform GO term enrichment """
        term = ['GO_Biological_Process_2021',
                'GO_Molecular_Function_2021',
                'GO_Cellular_Component_2021']
        
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=term,
                         organism=organism,
                         outdir=None,
                         cutoff=0.5)

        # change dtype of enr
        enr = enr.results.astype({'Term': 'string'})

        # transform the pvalue
        enr_data = enr.sort_values('Adjusted P-value')
        enr_data['Adjusted P-value'] = - np.log10(enr_data['Adjusted P-value'])

        # calculator the overlap in %
        numerator = [float(i[0]) for i in enr_data['Overlap'].str.split('/')]
        denominator = [float(i[1]) for i in enr_data['Overlap'].str.split('/')]
        overlap = np.round(np.array(numerator) / np.array(denominator) * 100)
        enr_data['Overlap in %'] = overlap

        # partition the term
        enr_data['GO term ID'] = [i[1].split(')')[0]
                                  for i in enr_data['Term'].str.split('(')]
        enr_data['GO term ID'] = enr_data['GO term ID'].astype('string')
        enr_data['Term'] = enr_data['Term'].str.partition(' (')[0]

        # filter the data by FDR
        enr_data = enr_data[enr_data['Adjusted P-value'] > 1.3]
        enr_data = enr_data.sort_values('Combined Score', ascending=False) 
        
        enr_data_filt = self.filter_go_by_lin_sim(
            [enr_data[enr_data['Gene_set'] == i] for i in term])
        enr_data_filt = [enr_data_filt[enr_data_filt['Gene_set'] == i]
                         for i in term]
        
        for data in enr_data_filt:
            data['info content'] = [get_info_content(go_id, termcounts)
                                    for go_id in data['GO term ID']]
            data = data.sort_values('info content', ascending=False)
        self.go_data = enr_data_filt.copy()
        return enr_data_filt
    
    def plot_array_enrichment(self, go_data, figsize, top,
                              savefig=False, title=None):
    
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        
        for data, axes in zip(go_data, ax.flat):
            data_copy = data.copy()
            data_copy['Term'] = [textwrap.fill(i, 30)
                                 for i in data_copy['Term']]
            
            sns.scatterplot(data=data_copy.head(top),
                            x='Adjusted P-value',
                            y='Term',
                            size='Overlap in %',
                            hue='Combined Score', 
                            sizes=(20, 200),
                            palette='viridis_r',
                            ax=axes)
            plt.legend(loc='best')
            sns.despine()
        fig.tight_layout()
        
        if title is not None:
            fig.suptitle(title)
            fig.subplots_adjust(top=0.9)
            
        if savefig is True:
            fig.savefig('plot_array_enrichment.pdf', bbox_inches='tight',
                        transparent=True)

    def array_enrichment_analysis_plot(self, gene_list, organism, figsize, top,
                                       savefig=False, title=None):

        '''Perform enrichment analysis and plot
        Biological Process, Molecular Function and Cellular compartment'''
        # enrichment
        go_data = self.array_enrichment_analysis(gene_list=gene_list,
                                                 organism=organism)
        
        # plot enrichment
        self.plot_array_enrichment(go_data=go_data, figsize=figsize, top=top,
                                   savefig=savefig, title=title)
        
    def go_enricher(self, up_gene_list, down_gene_list, organism, go_term=0):
    
        go_datasets = []
        
        for gene_list in [up_gene_list, down_gene_list]:
        
            """ Perform GO term enrichment """
            term = ['GO_Biological_Process_2021',
                    'GO_Molecular_Function_2021',
                    'GO_Cellular_Component_2021'][go_term]
            enr = gp.enrichr(gene_list=gene_list,
                             gene_sets=term,
                             organism=organism, 
                             outdir=None,
                             cutoff=0.5)

            # change dtype of enr
            enr = enr.results.astype({'Term': 'string'})

            # transform the pvalue
            enr_data = enr.sort_values('Adjusted P-value')
            enr_data['Adjusted P-value'] = - np.log10(
                enr_data['Adjusted P-value'])

            # calculator the overlap in %
            numerator = [float(i[0])
                         for i in enr_data['Overlap'].str.split('/')]
            denominator = [float(i[1])
                           for i in enr_data['Overlap'].str.split('/')]
            overlap = np.round(np.array(numerator) / 
                               np.array(denominator) * 100)
            enr_data['Overlap in %'] = overlap

            # partition the term 
            enr_data['GO term ID'] = [i[1].split(')')[0]
                                      for i in enr_data['Term'].str.split('(')]
            enr_data['GO term ID'] = enr_data['GO term ID'].astype('string')
            enr_data['Term'] = enr_data['Term'].str.partition(' (')[0]

            # filter the data by FDR
            enr_data = enr_data[enr_data['Adjusted P-value'] > 1.3]
            enr_data = enr_data.sort_values('Combined Score', ascending=False) 
            
            go_datasets.append(enr_data)
        
        return go_datasets
    
    def filter_go_by_lin_sim(self, go_datasets):
    
        '''Filter data based on lin sematic similarity'''
        # define a threshold for similarity
        threshold = 0.7
        go_data = []
        
        for data in go_datasets:
            
            if data.shape[0] > 0:
                # remove similar elements
                unique_values = [data['GO term ID'].iloc[0]]

                for val in data['GO term ID']:
                    try:
                        sims = [lin_sim(
                            val, u_val, godag, termcounts) >= threshold
                            for u_val in unique_values
                            if lin_sim(val, u_val, godag, termcounts)
                            is not None]

                        if any(sims):
                            pass

                        else: 
                            unique_values.append(val)

                    except KeyError:
                        continue
                
                go_data.append(data[data['GO term ID'].isin(unique_values)])
    
        go_data = pd.concat(go_data, axis=0).reset_index(drop=True)
            
        return go_data

    def calculate_mean_fold_enrichment(self, go_data, fc_data):
    
        mean_fold_go_data = go_data.copy()
        
        mean_fold_go_data['Genes'] = mean_fold_go_data['Genes'].str.split(';')

        mean_fold_enrichment = []
        
        for genes in mean_fold_go_data['Genes']:
            mean_genes = fc_data[fc_data['Genes'].isin(genes)].mean(
                numeric_only=True)[0]
            mean_fold_enrichment.append(mean_genes)

        mean_fold_go_data['mean_fold_enrichment'] = mean_fold_enrichment
        self.go_data = mean_fold_go_data.copy()
        return mean_fold_go_data
    
    def go_term_scatterplot(self, mean_fold_go_data, n_go_terms,
                            savefig, addit_annot=None):
    
        mean_fold_go_data['Term'] = [i.replace(' ', '\n')
                                     for i in mean_fold_go_data['Term']]
        
        with sns.plotting_context('notebook'):
            fig, ax = plt.subplots(figsize=(8, 6))
            g = sns.scatterplot(data=mean_fold_go_data,
                                x='mean_fold_enrichment',
                                y='Adjusted P-value',
                                size='Overlap in %',
                                hue='Combined Score', 
                                sizes=(20, 200), 
                                ax=ax,
                                palette='viridis_r')
            
            up_go_data_idx = mean_fold_go_data[mean_fold_go_data['mean_fold_enrichment'] > 0].sort_values('Combined Score', ascending=False).head(n_go_terms).index.to_list()
            down_go_data_idx = mean_fold_go_data[mean_fold_go_data['mean_fold_enrichment'] < 0].sort_values('Combined Score', ascending=False).head(n_go_terms).index.to_list()

            if addit_annot is None:
                indices = up_go_data_idx + down_go_data_idx
            else:
                addit_annot_idx = mean_fold_go_data[mean_fold_go_data['Term'].str.contains(addit_annot)].index.to_list()
                indices = up_go_data_idx + down_go_data_idx + addit_annot_idx
            texts = [plt.text(mean_fold_go_data['mean_fold_enrichment'][idx],
                              mean_fold_go_data['Adjusted P-value'][idx],
                              mean_fold_go_data['Term'][idx],
                              ha='center',
                              va='center',
                              wrap=True,
                              fontsize=8,
                              bbox=dict(boxstyle='round', fc='w', ec='black',
                                        alpha=0.3)) for idx in indices]
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black',
                                               alpha=0.3))

            g.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3), ncol=1)

            norm = plt.Normalize(mean_fold_go_data['Combined Score'].min(),
                                 mean_fold_go_data['Combined Score'].max())
            cmap = sns.color_palette("viridis_r", as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm,)

            cax = fig.add_axes([ax.get_position().x1+0.05,
                                ax.get_position().y0, 0.06,
                                ax.get_position().height / 2])
            ax.figure.colorbar(sm, cax=cax, label="Cobmined Score")    

            sns.despine()
            if savefig is True:
                plt.savefig('GO_scatter.pdf', bbox_inches='tight',
                            transparent=True)
            plt.show()

    def enrichment_analysis_plot(self, up_gene_list, down_gene_list, organism,
                                 go_term, fc_data, n_go_terms=15,
                                 savefig=False):
        
        '''Plot enrichment analyis with fold change data'''

        go_datasets = self.go_enricher(up_gene_list=up_gene_list,
                                       down_gene_list=down_gene_list,
                                       organism=organism, go_term=go_term)

        go_data = self.filter_go_by_lin_sim(go_datasets=go_datasets)

        mean_fold_go_data = self.calculate_mean_fold_enrichment(
            go_data=go_data, fc_data=fc_data)

        self.go_term_scatterplot(mean_fold_go_data=mean_fold_go_data,
                                 n_go_terms=n_go_terms, savefig=savefig)

    def go_circle_plot(self, mean_fold_go_data, n_go_terms=10, savefig=False):

        from pycirclize import Circos
        from pycirclize.parser import Matrix
        
        mean_fold_go_data['Term'] = [textwrap.fill(i, 30)
                                     for i in mean_fold_go_data['Term']]
        
        upreg_go_data = mean_fold_go_data[mean_fold_go_data['mean_fold_enrichment'] > 0]
        downreg_go_data = mean_fold_go_data[mean_fold_go_data['mean_fold_enrichment'] < 0]

        for idx,  reg_go_data in enumerate([upreg_go_data, downreg_go_data]):

            test_data = reg_go_data.sort_values('Combined Score',
                                                ascending=False).head(n_go_terms).copy()

            unique_genes = set([gene for genes in test_data['Genes']
                                for gene in genes])

            matrix = pd.DataFrame([[gene, test_data['Term'].iloc[idx], 1]
                                   for gene in unique_genes
                                   for idx, gene_set in  enumerate(test_data['Genes'])
                                   if gene in gene_set], columns=["from", "to", "value"])
                    
            matrix_df = Matrix.parse_fromto_table(matrix.sort_values('to'))

            go_terms_sorted = list(matrix['to'].unique())
            sorted_genes = sorted(Counter(matrix['from']).items(), key=lambda x: x[1], reverse=True)
            sorted_genes = [i[0] for i in sorted_genes]

            circos = Circos.initialize_from_matrix(
                matrix_df,
                space=1,
                cmap="tab20",
                label_kws=dict(size=6, r=102, orientation="vertical"),
                link_kws=dict(direction=1, ec="black", lw=0),
                order = go_terms_sorted  + sorted_genes
            )
            fig = circos.plotfig()

            if idx == 0:
                plt.title('Upregulated proteins')

            else:
                plt.title('Downregulated proteins')
            if savefig == True:
                plt.savefig(f'{idx}_circle_plot.pdf', bbox_inches='tight')
