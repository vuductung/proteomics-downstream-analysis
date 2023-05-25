import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap as mp

from adjustText import adjust_text

from plotly.subplots import make_subplots
import plotly.express as px


class DimensionalityReduction:

    def _pca(self, data=None):

        '''Generate a PCA'''

        if isinstance(data, pd.DataFrame):
            df = data.select_dtypes(include =['float64']).T.copy()

        else:
            # prepare the dataframe for PCA
            df = self.data.select_dtypes(include =['float64']).T.copy()

        # prepare features variable (Genes)
        features = list(np.arange(df.shape[1]))
        df.columns = features
        
        # prepare target column (sample types)
        df['target'] = df.index
        df = df.reset_index(drop= True)

        # Separating out the features
        x = df.loc[:, features].values
        
        # Separating out the target
        y = df.loc[:,['target']].values
        
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                                ,columns = ['principal component 1', 'principal component 2'])
        
        finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
        
        return pca, finalDf

    def pca_plot(self, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):
        
        """Plot PCA for one or more datasets."""

        if n_rows == 1 and n_cols == 1:
            pca_data = [self._pca()]

        else:
            pca_data = [self._pca(i) for i in self.datasets]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for data_set, axes, title in zip(pca_data, np.array(ax).flatten(), titles):
            sns.scatterplot(data=data_set[1], x='principal component 1', y='principal component 2', hue='target', ax=axes)
            pc1 = "%.2f" % (data_set[0].explained_variance_ratio_[0] * 100)
            pc2 = "%.2f" % (data_set[0].explained_variance_ratio_[1] * 100)
            axes.set(xlabel=f'PC1 {pc1}%', ylabel=f'PC2 {pc2}%')
            axes.set_title(title)
            axes.legend(loc='best')
            sns.despine()

        fig.tight_layout()

        if savefig:
            fig.savefig('pca_plots.pdf', bbox_inches='tight', transparent=True)
    
    def int_pca_plot(self, n_rows=1, n_cols=1, titles=[''], height=500, width=700):
    
        if n_rows == 1 and n_cols == 1:
            pca_data = [self._pca()[1]]

        else:
            pca_data = [self._pca(i)[1] for i in self.datasets]

        for i in pca_data:
            i['idx'] = i.index

        plots = [px.scatter(dataset, x="principal component 1", y="principal component 2", 
                            color="target", hover_data=['idx'], template='simple_white') for dataset in pca_data]

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
        row_col_number = [(i, j) for i in range(1, n_rows+1) for j in range(1, n_cols+1)]

        for plot, row_col in zip(plots, row_col_number):
            for idx in range(len(plot['data'])):
                fig.append_trace(plot['data'][idx], row=row_col[0], col=row_col[1])

        fig.update_layout(height=height, width=width, template='simple_white')
        fig.update_traces(marker=dict(size=12,
                                    line=dict(width=2,
                                                color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_xaxes(title_text='PC1')
        fig.update_yaxes(title_text='PC2')
        fig.show()

    def top_loadings(self, k):
        
        '''Get k features that explain most of variance in PC1 & 2'''
        
        X = np.array(self.data.select_dtypes('float').T)

        # analysis
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2).fit(X_scaled)        

        # results
        loading = pca.components_.T
                    
        topx = (loading ** 2)[:, 0].argsort()[-k:]
        topy = (loading ** 2)[:, 1].argsort()[-k:]
        top = np.append(topx, topy)
        
        return top
    
    def min_var_top_loadings(self, k, variance):

        float_data = self.data.select_dtypes('float').T
        float_data = StandardScaler().fit_transform(float_data)

        pca_with_variance = PCA(n_components=variance).fit(float_data)

        loading = pca_with_variance.components_.T

        top_features = [np.abs(loading)[:, component].argsort()[-k:] for component in range(loading.shape[1])]

        top_features = [value for top_feature in top_features for value in top_feature]

        return top_features

    def biplot(self, k=5, n_rows=1, n_cols=1, titles=[''], figsize=(8,3), savefig=False):
        
        targets = []
        features = []
        scores = []
        pvars = []
        tops = []
        arrows = []

        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]

        else:
            datasets = self.datasets.copy()

        for data in datasets:

            # data
            X = np.array(data.select_dtypes('float').T)
            target = data.select_dtypes('float').columns.to_list()
            feature = data['Genes'].to_list()

            # analysis
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2).fit(X_scaled)
            X_reduced = pca.transform(X_scaled)

            # results
            score = X_reduced[:, :2]
            loading = pca.components_[:2].T
            pvar = pca.explained_variance_ratio_[:2] * 100
                    
            topx = (loading ** 2)[:, 0].argsort()[-k:]
            topy = (loading ** 2)[:, 1].argsort()[-k:]
            top = np.append(topx,topy)
            arrow = loading[top]

        #         top = (loading * pvar).sum(axis=1).argsort()[-k:]
        #         arrow = loading[top]
            
            arrow /= np.sqrt((arrow ** 2).sum(axis=0))
            arrow *= np.abs(score).max(axis=0)
            
            targets.append(target)
            features.append(feature)
            scores.append(score)
            pvars.append(pvar)
            tops.append(top)
            arrows.append(arrow)
            
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for target, feature, score, pvar, top, arrow, title, axes  in zip(targets, features, scores, pvars, tops, arrows, titles, np.array(ax).flatten()):

            sns.scatterplot(x=score[:, 0], y=score[:, 1], hue=target, ax=axes)
            axes.set_title(title)
            sns.despine()

            for j in arrow:
                axes.arrow(0, 0, *j, color='k', width=0.5, ec='none',
                        length_includes_head=True, alpha=0.5, )
            
            texts = [axes.text(*(j), feature[i], ha='center', va='center') for i, j in zip(top, arrow)]
            adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black', alpha=0.5), ax=axes)
            axes.set(xlabel=f'PC1 {pvar[0]:.2f}%', ylabel=f'PC2 {pvar[1]:.2f}%')

        fig.tight_layout()

        if savefig == True:
            fig.savefig('biplot.pdf', bbox_inches='tight')

    def _umap(self, data=None, n_neighbors=3):

        '''Generate a UMAP plot'''
        if isinstance(data, pd.DataFrame):
            df = data.select_dtypes(include =['float64']).T.copy()
        else:
            df = self.data.select_dtypes(include =['float64']).T.copy()

        # prepare features variable (Genes)
        features = list(np.arange(df.shape[1]))
        df.columns = features
        
        # prepare target column (sample types)
        df['target'] = df.index
        df = df.reset_index(drop= True)

        # Separating out the features
        x = df.loc[:, features].values
 
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        reducer = mp.UMAP(n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(x)
        embedding_data = pd.DataFrame(data = embedding
                                ,columns = ['UMAP1', 'UMAP2'])
        
        finalDf = pd.concat([embedding_data, df[['target']]], axis = 1)
        
        return finalDf
    
    def umap_plot(self, n_neighbors=3, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):
        
        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]

        else:
             datasets = self.datasets.copy()

        # plot the pca
        umap_datasets = [self._umap(data=i, n_neighbors=n_neighbors) for i in datasets]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for umap_data, axes, title in zip(umap_datasets, np.array(ax).flatten(), titles):
            sns.scatterplot(data = umap_data, x = 'UMAP1', y = 'UMAP2', hue = 'target', ax=axes)
            axes.set_title(title)
            axes.legend(loc='best')
            sns.despine()
        
        fig.tight_layout()
        
        if savefig == True:
            fig.savefig('umap_figs.pdf', bbox_inches='tight', transparent=True)

    def _tsne(self, data=None, perplexity=10, learning_rate='auto', early_exaggeration=12):

        '''t-SNE dimensionality reduction'''

        if isinstance(data, pd.DataFrame):
            float_data = data.select_dtypes('float').T
        
        else:
            float_data = self.data.select_dtypes('float').T
        
        features = float_data.index.to_list()
        float_data = StandardScaler().fit_transform(float_data)
        tsne_data = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=early_exaggeration, random_state=42).fit_transform(float_data)
        tsne_data = pd.DataFrame(data=tsne_data, columns =['TSNE 1', 'TSNE 2'])
        tsne_data['features'] = features
        
        return tsne_data

    def tsne_plot(self, perplexity=10, learning_rate='auto', early_exaggeration=12, n_rows=1, n_cols=1, titles=[''], figsize=(5, 5), savefig=False):

        '''Plot t-SNE'''

        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]
        
        else:
            datasets = self.datasets.copy()

        tsne_datasets = [self._tsne(data=i, perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=early_exaggeration) for i in datasets]

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for tsne_data, title, axes in zip(tsne_datasets, titles, np.array(ax).flatten()):
            sns.scatterplot(data=tsne_data, x='TSNE 1', y='TSNE 2', hue='features', ax=axes)
            axes.set_title(title)
            sns.despine()

        if savefig == True:
            fig.savfig('t_SNE_plot.pdf', bbox_inches='tight', transparent=True)

    def sequential_pca_tsne_plot(self, variance=0.8, perplexity=10, learning_rate='auto', early_exaggeration=12, n_rows=1, n_cols=1, titles=[''], figsize=(5,5), savefig=False):

        '''Plot tSNE after PCA dimensionality reduction'''

        if n_rows == 1 and n_cols == 1:
            datasets = [self.data]

        else:
            datasets = self.datasets.copy()

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for data, title, ax in zip(datasets, titles, np.array(ax).flatten()):
            float_data = data.select_dtypes('float').T
            features = float_data.index.to_list()
            float_data = StandardScaler().fit_transform(float_data)
            pca_reduced_data = PCA(n_components=variance).fit_transform(float_data)

            pca_data = pd.DataFrame(pca_reduced_data, index=features).T

            tsne_data = self._tsne(data=pca_data, perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=early_exaggeration)

            sns.scatterplot(data=tsne_data, x='TSNE 1', y='TSNE 2', hue='features', ax=ax)
            ax.set_title(title)
            sns.despine()

        if savefig == True:
            fig.savefig('seq_pca_t_SNE_plot.pdf', bbox_inches='tight', transparent=True)
