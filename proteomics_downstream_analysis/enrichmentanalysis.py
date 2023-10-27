import gseapy as gp
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.associations import dnld_assc
from goatools.semantic import semantic_similarity, TermCounts, \
    get_info_content, resnik_sim, lin_sim, deepest_common_ancestor

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import Counter
import textwrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from adjustText import adjust_text

import plotly.express as px
import plotly.io as pio


from .utils import is_jupyter_notebook, format_ytick_label

class EnrichmentAnalysis:
    """ This class encapsulates enrichment analysis methods """

    def __init__(self, obo_file=None, gaf_file=None):
        
        if obo_file is None and gaf_file is None:
            pass
        else:
            godag = GODag(obo_file)
            associations = read_gaf(gaf_file)
            termcounts = TermCounts(godag, associations)

            self.godag = godag
            self.termcounts = termcounts

    def array_enrichment_analysis(self, gene_list, organism,
                                  sematic_sim_filter=True):

        """
        Perform GO term enrichment

        Parameters
        ----------
        gene_list : list
            A list of genes to be used
            for enrichment analysis

        organism : str
            Organism to be used to do enrichment
            analysis. e.g 'human' or 'mouse'
        
        Returns
        -------
        enr_data_filt : list
            List of dataframes with GO_Biological_Process_2021,
            GO_Molecular_Function_2021
            GO_Cellular_Component_2021 enrichment data.
        """
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
        
        if sematic_sim_filter is True:
            enr_data_filt = self.filter_go_by_lin_sim(
                [enr_data[enr_data['Gene_set'] == i] for i in term])
            enr_data_filt = [enr_data_filt[enr_data_filt['Gene_set'] == i]
                            for i in term]
            
            for data in enr_data_filt:
                data['info content'] = [get_info_content(go_id, self.termcounts)
                                        for go_id in data['GO term ID']]
                data = data.sort_values('info content', ascending=False)
            
        else:
            enr_data_filt = [enr_data[enr_data['Gene_set'] == i] for i in term]
        
        self.go_data = enr_data_filt.copy()
        return enr_data_filt
    

    def plot_array_enrichment(self, go_data, figsize, top,
                              savefig=False):
        """

        Parameters
        ----------
        go_data : list
            List of dataframes with GO_Biological_Process_2021,
            GO_Molecular_Function_2021 and
            GO_Cellular_Component_2021 enrichment data.

        figsize : tuple
            Size of the figure e.g (10, 10)
        
        top : int
            top n terms to be plotted based on the combined score

        savefig : boolean
             If True saves the figure (Default value = False).

        Returns
        -------
        matplotlib.pyplot
            Plot of the enrichment analysis
        """

        titles = ['GO_Biological_Process',
                  'GO_Molecular_Function',
                  'GO_Cellular_Component']
        
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        
        for data, axes, title in zip(go_data, ax.flat, titles):
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
            
            axes.title.set_text(title)
            plt.legend(loc='best')
            sns.despine()
        fig.tight_layout()
            
        if savefig is True:
            fig.savefig('plot_array_enrichment.pdf', bbox_inches='tight',
                        transparent=True)
            
    def int_plot_array_enrichment(self, go_data, height=700, width=1500, top=100):

        n_cols = 3
        titles = ['Biological Process',
                  'Cellular Component',
                  'Molecular Function']
        fig = make_subplots(1, 3, horizontal_spacing = 0.2, subplot_titles=titles)

        cmin = np.min(pd.concat(go_data)['Combined Score'])
        cmax = np.max(pd.concat(go_data)['Combined Score'])

        for i, data in enumerate(go_data, start=1):
            
            data = data.iloc[:top].copy()

            # Calculate row and col values for the subplots based on i
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1

            # Add the scatter plots to the subplots layout
            fig.add_trace(go.Scatter(x=data['Adjusted P-value'],
                                    y=data['Term'],
                                    mode='markers',
                                    marker=dict(size=data['Overlap in %'],
                                                color=data['Combined Score'],
                                                colorscale='Viridis', 
                                                showscale=True,  
                                                cmin=cmin,  
                                                cmax=cmax),
                                    text=data['Genes'],
                                    hovertemplate = "Overlap in %: %{marker.size}<br>" +
                                                    "Genes: %{text}<br>"),
                                        row=row,
                                        col=col)

            fig.update_xaxes(title_text='Adjusted P-value')
            fig.update_yaxes(title_text='Term', 
                            tickmode='array',
                            tickvals=data['Term'],
                            ticktext=[format_ytick_label(tick) for tick in data['Term']],
                            row=row, col=col)

        # Update the layout and traces
        fig.update_layout(template='simple_white',
                        height=height,
                        width=width,
                        hoverlabel=dict(bgcolor="white",
                                        font_size=16))
        # Show the plot
        if is_jupyter_notebook():
            fig.show()
        else:
            st.plotly_chart(fig, use_container_width=True)

    def array_enrichment_analysis_plot(self, gene_list, organism, figsize, top,
                                       savefig=False):

        """
        Perform enrichment analysis and plot
        Biological Process, Molecular Function
        and Cellular compartment

        Parameters
        ----------
        gene_list : list
            A list of genes to be used for enrichment analysis
            
        organism : str
            Organism to be used to do enrichment
            analysis. e.g 'human' or 'mouse'

        figsize : tuple
            Figure size e.g (10, 10)
        
        top : int
            Top n terms to be plotted
            based on the combined score

        savefig : boolean
             If True save figure (Default value = False)

        Returns
        -------
        matplotlib.pyplot
        """
        # enrichment
        go_data = self.array_enrichment_analysis(gene_list=gene_list,
                                                 organism=organism)
        
        self.plot_array_enrichment(go_data=go_data, 
                                   figsize=figsize, top=top,
                                   savefig=savefig)
        return go_data
    
    def array_enrichment_analysis_int_plot(self, gene_list, organism, height=700, width=1500, top=10):

         # enrichment
        go_data = self.array_enrichment_analysis(gene_list=gene_list,
                                                 organism=organism)
        
        self.int_plot_array_enrichment(go_data=go_data, height=height, width=width, top=top)
        
        return go_data
    
    def go_enricher(self, up_gene_list, down_gene_list, organism, go_term=0):
        """
        Perform GO term enrichment analysis

        Parameters
        ----------
        up_gene_list : list
            List of upregulated genes

        down_gene_list : list
            List of downregulated genes
            
        organism : str
            Organism to be used to do enrichment
            analysis. e.g 'human' or 'mouse'

        go_term : int
             0 represents Biological process,
             1 represents Molecular function and
             2 represents cellular component
             (Default value = 0)

        Returns
        -------
        go_datasets : list
            List of dataframes with GO_Biological_Process_2021,
            GO_Molecular_Function_2021 and
            GO_Cellular_Component_2021 enrichment data.
        """
    
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
    
        """
        Filter data based on lin sematic similarity

        Parameters
        ----------
        go_datasets : list
            List of dataframes with upregulated and
            downregulated enriched GO terms
            
        Returns
        -------
        go_data : list
            List of dataframes with upregulated and
            downregulated enriched GO terms filtered by
            lin sematic similarity
        """
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
                            val, u_val, self.godag, self.termcounts) >= threshold
                            for u_val in unique_values
                            if lin_sim(val, u_val, self.godag, self.termcounts)
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
        """

        Parameters
        ----------
        go_data : list
            List of dataframes with upregulated and
            downregulated enriched GO terms filtered by
            lin sematic similarity
            
        fc_data : pandas.DataFrame
            Dataframe with fold change data

        Returns
        -------
        mean_fold_go_data : pandas.DataFrame
            Dataframe with mean fold enrichment data
        """
    
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
        """

        Parameters
        ----------
        mean_fold_go_data : pandas.DataFrame
            Go term data with mean fold enrichment data

        n_go_terms : int
            Number of go terms to be plotted for up and downregulated genes
            
        savefig : bool 
            If True save figure

        addit_annot : list
             Additional Go terms to be annotated (Default value = None)

        Returns
        -------
        matplotlib.pyplot.figure
            Scatterplot of mean fold enrichment vs. adjusted p-value
        """
    
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
        
        """
        Plot enrichment analyis with fold change data

        Parameters
        ----------
        up_gene_list : list
            List of upregulated genes

        down_gene_list : list
            List of downregulated genes

        organism : str
            Organism for enrichment analysis

        go_term : int
             0 represents Biological process,
             1 represents Molecular function and
             2 represents cellular component
             (Default value = 0)

        fc_data : pandas.DataFrame
            Dataframe with fold change data

        n_go_terms : int
             Numbers of go terms to be plotted (Default value = 15)
        savefig : bool
             If True save figure (Default value = False)

        Returns
        -------
        matplotlib.pyplot.figure
        """

        go_datasets = self.go_enricher(up_gene_list=up_gene_list,
                                       down_gene_list=down_gene_list,
                                       organism=organism, go_term=go_term)

        go_data = self.filter_go_by_lin_sim(go_datasets=go_datasets)

        mean_fold_go_data = self.calculate_mean_fold_enrichment(
            go_data=go_data, fc_data=fc_data)

        self.go_term_scatterplot(mean_fold_go_data=mean_fold_go_data,
                                 n_go_terms=n_go_terms, savefig=savefig)

    def go_circle_plot(self, mean_fold_go_data, n_go_terms=10, savefig=False):
        """

        Parameters
        ----------
        mean_fold_go_data : pandas.DataFrame
            Go term data with mean fold enrichment data
            
        n_go_terms : int
             Number of go terms to plot (Default value = 10)
        savefig : bool
             If True save figure (Default value = False)

        Returns
        -------
        circos plot
        """

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


    def _filter_go_data(self, go_data, godag):

        '''
        Filter the go_data by the go terms in the godag

        Parameters
        ----------
        go_data : pandas.DataFrame
            Dataframe with go term data
        godag: goatools.obo_parser.GODag
            GODag object

        Returns
        -------
        go_data : pandas.DataFrame
            Filtered dataframe with go term data
        '''

        return go_data[go_data['GO term ID'].isin(godag.keys())]

    def _calculate_semantic_similarity_between_go_terms(self, go_data, godag, termcounts):
        
        '''
        Calculate the semantic similarity between given go terms

        Parameters
        ----------
        go_data : pandas.DataFrame
            Dataframe with go term data
        godag: goatools.obo_parser.GODag
            GODag object
        termcounts: goatools.semantic.TermCounts
            TermCounts object

        Returns
        -------
        sem_sims : numpy.ndarray
            Array with semantic similarity data
        '''

        go_data = self._filter_go_data(go_data, godag)
        go_termids = go_data['GO term ID'].tolist()

        sem_sims = np.zeros((len(go_termids), len(go_termids)))
        for ax1, i in enumerate(go_termids):
            for ax2, j in enumerate(go_termids):
                sem_sims[ax1, ax2] = lin_sim(i, j, godag, termcounts)

        return sem_sims

    def _perform_mds(self, sem_sims):

        '''
        Perform mutlidimensional scaling

        Parameters
        ----------
        sem_sims : numpy.ndarray
            Array with semantic similarity data
        
        Returns
        -------
        X_transformed : numpy.ndarray
            Array with MDS scores
        '''
        
        np.random.seed(42)
        embedding = MDS(n_components=2,
                        normalized_stress='auto', 
                        dissimilarity='euclidean')

        X_transformed = embedding.fit_transform(np.nan_to_num(sem_sims, 0))

        return X_transformed

    def _cluster_mds_scores(self, x, n_clusters=25):
        
        '''
        Cluster the MDS scores

        Parameters
        ----------
        x : numpy.ndarray
            Array with MDS scores
        n_clusters : int
            Number of clusters (Default value = 25)

        Returns
        -------
        labels : numpy.ndarray
            Array with cluster labels
        hue : list
            Array with cluster labels as strings
        '''

        knn = KMeans(n_clusters=n_clusters, 
                    n_init=10,
                    random_state=42)
        output = knn.fit(x)

        labels = output.labels_
        hue = [str(i) for i in labels] 

        return labels, hue

    def _calculate_deepest_common_ancestor(self, go_data, labels, godag, ):

        '''
        Calculate the deepest common ancestor for each cluster

        Parameters
        ----------
        go_data : pandas.DataFrame
            Dataframe with go term data
        labels : numpy.ndarray or list
            Array with cluster labels
        godag: goatools.obo_parser.GODag
            GODag object

        Returns
        -------
        deepest_common_ancestors : list
            List with deepest common ancestors
        '''

        deepest_common_ancestors = []
        for i in range(len(np.unique(labels))):
            id_cluster = go_data[labels == i]['GO term ID'].tolist()
            go_root = deepest_common_ancestor(id_cluster, godag)
            dca = godag[go_root].name
            deepest_common_ancestors.append(dca)
        return deepest_common_ancestors

    def _calculate_center(self, x, labels):
        
        '''
        Calculate the center of each cluster for plotting

        Parameters
        ----------
        x : numpy.ndarray
            Array with MDS scores

        labels : numpy.ndarray or list
            Array with cluster labels
        
        Returns
        -------
        center : list
            List with cluster centers
        '''
        
        center = []
        for i in range(len(np.unique(labels))):
            x_cluster = x[labels == i]
            center.append(np.mean(x_cluster, axis=0))
        return center

    def _customwrap(self, s, width=15):

        '''
        Wrap string with <br> tag

        Parameters
        ----------
        s : str
            String to be wrapped
        width : int
            Width of the string (Default value = 15)
        '''

        return "<br>".join(textwrap.wrap(s, width=width))

    def plot_enrichment_mds(self, x, hue, names, centers, pvalues, title=None,
                            figsize=(8,8), savefig=True,
                            dir='enrichment_mds.pdf'):
        
        '''
        Plot the enrichment MDS scores

        Parameters
        ----------
        x : numpy.ndarray
            Array with MDS scores
        hue : numpy.ndarray or list
            Array with cluster labels
        names : list
            List with cluster names
        centers : list
            List with cluster centers
        pvalues : numpy.ndarray or list
            Array with pvalues
        title : str
            Title of the plot (Default value = None)
        figsize : tuple
            Figure size (Default value = (8,8))
        savefig : bool
            If True save figure (Default value = True)
        dir : str
            Directory to save the figure (Default value = 'enrichment_mds.pdf')
        
        Returns
        -------
        matplotlib.pyplot.figure
        '''

        plt.figure(figsize=figsize)
        sns.scatterplot(x=x[:,0],
                        y=x[:,1], 
                        hue=hue,
                        size=pvalues,
                        sizes=(50, 200),
                        )
        plt.xlabel('MDS1')
        plt.ylabel('MDS2')
        plt.legend().set_visible(False)
        texts = [plt.text(center[0], center[1], names[idx], ha='center', va='center', fontsize=9)
                for idx, center in enumerate(centers)]    
        adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black'))

        if savefig == True:
            plt.savefig(dir, bbox_inches='tight')
        plt.title(title)
        plt.show()
    
    def prepare_data_for_enrichment_mds(self, go_data, godag, termcounts, n_clusters):

        # filter the go_data to only contain go terms in the godag
        go_data = self._filter_go_data(go_data, godag)

        # calculate the semantic similarity between given go terms
        sem_sims = self._calculate_semantic_similarity_between_go_terms(go_data, godag, termcounts)

        # perform mds on the semantic similarity matrix
        X_transformed = self._perform_mds(sem_sims)

        # cluster the mds scores
        labels, hue = self._cluster_mds_scores(X_transformed, n_clusters)

        # calculate the center of each cluster
        centers = self._calculate_center(X_transformed, labels)

        # calculate the deepest common ancestor of each cluster
        dca = self._calculate_deepest_common_ancestor(go_data, labels, godag)

        # get the names of the go terms and the pvalues
        names = [textwrap.fill(i, 15) for i in dca]
        pvalues = go_data['Adjusted P-value'].tolist()

        return X_transformed, hue, labels, names, centers, pvalues

    def create_plotly_data(self, x, hue, go_data, godag):

        '''
        Create the data for the interactive plots based on the MDS scores

        Parameters
        ----------
        x : numpy.ndarray
            Array with MDS scores
        hue : numpy.ndarray or list
            Array with cluster labels
        go_data : pandas.DataFrame
            Dataframe with go term data

        Returns
        -------
        data : pandas.DataFrame
            Dataframe with MDS scores, cluster labels, go terms and pvalues
        '''
        go_data = self._filter_go_data(go_data, godag)
        data = pd.DataFrame(x, columns=['MDS1', 'MDS2'])
        data['hue'] = hue
        data['Term'] = go_data['Term'].tolist()
        data['Adjusted P-value'] = go_data['Adjusted P-value'].tolist()

        return data

    def plot_interactive_enrichment_mds(self, data, labels, centers, names,
                                        title=None, figsize=(700, 700),
                                        save=False, dir='int_enrichment.html'):
        
        '''
        Plot the interactive enrichment MDS scores

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with MDS scores, cluster labels, go terms and pvalues
        labels : numpy.ndarray or list
            Array with cluster labels
        centers : list
            List with cluster centers
        names : list
            List with cluster names
        title : str
            Title of the plot (Default value = None)
        figsize : tuple
            Figure size (Default value = (700,700))
        save : bool
            If True save figure (Default value = False)
        dir : str
            Directory to save the figure
            (Default value = 'int_enrichment.html')
        
        Returns
        -------
        plotly figure
        '''
        fig = px.scatter(data,
                        x='MDS1',
                        y='MDS2',
                        color='hue',
                        size='Adjusted P-value',
                        hover_data={'Term': True,
                                    'MDS1': False, # hide
                                    'MDS2': False, # hide
                                    'hue':False, # hide
                                    'Adjusted P-value':False}, # hide
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=title)

        fig.update_layout(template='simple_white',
                                height=figsize[0],
                                width=figsize[1],
                                showlegend=False)

        for i in range(len(np.unique(labels))):
            fig.add_annotation(x=centers[i][0],
                                y=centers[i][1],
                                text=list(map(self._customwrap, names))[i],
                                showarrow=True,
                                arrowhead=0)

        if save == True:
            # Save the plot as an HTML file
            pio.write_html(fig, f'{dir}')

        fig.show()
        