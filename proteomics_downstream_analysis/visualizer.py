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
from scipy.stats import pearsonr

class Visualizer:
    """Visualizer class for plotting."""

    def __init__(self):
        pass

    def volcano_plot(
        self,
        fc_data,
        pv_data,
        qv_data,
        n_rows,
        n_cols,
        gene_list=None,
        figsize=(8, 8),
        filepath=False,
        upper_fc_cutoff=1,
        lower_fc_cutoff=1,
        annot=None,
        gene_column="Genes",
        qvalue=True,
        palette=None,
        title=None
    ):
        """
        Plot a volcano plot.

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

        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, layout="tight")

        if title:
            fig.suptitle(title)
        cols = fc_data.select_dtypes("float").columns
        axs = np.array(ax).flatten()
        indices = False
        if palette is None:
            palette = sns.color_palette("coolwarm", 2)
        for i, axes in zip(
            cols, axs
        ):
            
            qv_mask = qv_data[i].isna()
            fc_mask = fc_data[i].isna()
            pv_mask = pv_data[i].isna()
            mask = ~(qv_mask | fc_mask | pv_mask)

            if qvalue:
                    # plot the non significant datapoints lightgrey
                    non_sign_mask = qv_data[i] > 0.05
                    sns.scatterplot(
                        x=fc_data[non_sign_mask & mask][i],
                        y=pv_data[non_sign_mask & mask][i],
                        color="lightgrey",
                        alpha=0.5,
                        ax=axes,
                        linewidth=0,
                        rasterized=True
                    )

                    # plot the significant datapoints with pos fold change
                    sign_mask = qv_data[i]  < 0.05
                    pos_fc_mask =  fc_data[i] > 0
                    sns.scatterplot(
                        x=fc_data[
                            sign_mask & pos_fc_mask & mask
                        ][i],
                        y=pv_data[
                           sign_mask & pos_fc_mask & mask
                        ][i],
                        color=palette[1],
                        ax=axes,
                        linewidth=0,
                        rasterized=True
                    )

                    # plot the significant datapoints with neg fold change
                    neg_fc_mask =  fc_data[i] < 0
                    sns.scatterplot(
                        x=fc_data[
                           sign_mask & neg_fc_mask * mask
                        ][i],
                        y=pv_data[
                            sign_mask & neg_fc_mask * mask
                        ][i],
                        color=palette[0],
                        ax=axes,
                        linewidth=0,
                        rasterized=True,
                    )

                    axes.set_xlabel("log2 fold change", fontsize=10)
                    axes.set_ylabel("-log10 p-value", fontsize=10)

                    # set q-value threshold
                    if sign_mask.sum() > 0:
                        threshold = pv_data[qv_data[i] < 0.05][i].sort_values().values[0]
                        axes.axhline(threshold, ls="--", color="lightgrey")

                    if annot == "fc_cutoff":
                        indices = fc_data[
                            (
                                (fc_data[i] < lower_fc_cutoff)
                                | (fc_data[i] > upper_fc_cutoff)
                                | (fc_data["Genes"].isin(gene_list))
                            )
                            & sign_mask & mask
                        ].index
                        axes.axvline(lower_fc_cutoff, ls="--", color="lightgrey")
                        axes.axvline(upper_fc_cutoff, ls="--", color="lightgrey")

                    else:
                        annot_mask = fc_data[gene_column].isin(gene_list)
                        indices = fc_data[annot_mask & mask].index

            else:
                # plot all non significant datapoints
                non_sign_mask = pv_data[i] < 1.3
                sns.scatterplot(
                    x=fc_data[i][non_sign_mask & mask],
                    y=pv_data[i][non_sign_mask & mask],
                    color="lightgrey",
                    alpha=0.5,
                    ax=axes,
                    linewidth=0,
                    rasterized=True
                )
                # plot all significant datapoints with pos fold change
                sign_mask = pv_data[i] > 1.3
                pos_fc_mask = fc_data[i] > 0
                sns.scatterplot(
                    x=fc_data[sign_mask & mask & pos_fc_mask][i],
                    y=pv_data[sign_mask & mask & pos_fc_mask][i],
                    color="lightgrey", #palette[0],
                    ax=axes,
                    linewidth=0,
                    rasterized=True
                )

                # plot all significant datapoints with neg fold change
                neg_fc_mask = fc_data[i] < 0
                sns.scatterplot(
                    x=fc_data[
                       sign_mask & mask & neg_fc_mask
                    ][i],
                    y=pv_data[
                       sign_mask & mask & neg_fc_mask
                    ][i],
                    color="lightgrey", #palette[1],
                    ax=axes,
                    linewidth=0,
                    rasterized=True
                )

                axes.set_xlabel("log2 fold change", fontsize=10)
                axes.set_ylabel("-log10 p-value", fontsize=10)

                # set p-value threshold
                axes.axhline(1.3, ls="--", color="lightgrey")

                if annot == "fc_cutoff":
                    indices = fc_data[
                        (
                            (fc_data[i] < lower_fc_cutoff)
                            | (fc_data[i] > upper_fc_cutoff)
                            | (fc_data["Genes"].isin(gene_list))
                        )
                        & sign_mask & mask
                    ].index
                    axes.axvline(lower_fc_cutoff, ls="--", color="lightgrey")
                    axes.axvline(upper_fc_cutoff, ls="--", color="lightgrey")

                else:
                    annot_mask = fc_data[gene_column].isin(gene_list)
                    indices = fc_data[annot_mask & mask].index

            sns.scatterplot(x=fc_data[i][indices],
                            y=pv_data[i][indices],
                            color='black',
                            ax=axes,
                            rasterized=True, 
                            facecolors='none',
                            linewidth=0.5,
                            edgecolor="black")
        
            # annotation
            if indices.any():
                texts = [
                    axes.text(
                        fc_data[i][idx],
                        pv_data[i][idx],
                        fc_data[gene_column][idx],
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    for idx in set(indices)
                ]
                adjust_text(
                    texts, arrowprops=dict(arrowstyle="-", color="black", linewidth=0.5), ax=axes
                )

            sample_annot = i.split("/")
            x_min = fc_data[i].min()
            x_max = fc_data[i].max()
            x1 = x_max/2
            x2 = x_min/2

            y = pv_data[i].max() - 0.5

            axes.text(
                x1,
                y,
                sample_annot[0],
                fontsize=10,
                bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.3),
            )
            axes.text(
                x2,
                y,
                sample_annot[1],
                fontsize=10,
                bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.3),
            )

            if filepath:
                fig.savefig(filepath, bbox_inches="tight", transparent=True)

    def sign_prots_plot(self, qv_data, normalized=False, figsize=(8, 5), savefig=False):
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

        sign_data = (qv_data.select_dtypes("float") < 0.05).sum(axis=0)
        norm_sign_data = (
            (qv_data.select_dtypes("float") < 0.05).sum(axis=0)
            / qv_data.shape[0]
        ) * 100
        x = [textwrap.fill(i, 8) for i in sign_data.index]
        plt.figure(figsize=figsize)
        if normalized is True:
            sns.barplot(x=x, y=norm_sign_data.values)
            plt.ylabel("number of significant proteins in %")

        else:
            sns.barplot(x=x, y=sign_data.values)
            plt.ylabel("number of significant proteins")

            sns.despine()

        if savefig is True:
            plt.savefig("sign_prots_plot.pdf", transparent=True, bbox_inches="tight")

    def int_volcano_plot(
        self,
        fc_data,
        pv_data,
        qv_data,
        n_rows,
        n_cols,
        height,
        width,
        annot_genes=[],
        upper_fc_cutoff=None,
        lower_fc_cutoff=None,
        qv_cutoff=False,
        save=False,
        filename="int_volcano_plot.html",
    ):
        indices = fc_data[fc_data["Genes"].isin(annot_genes)].index.to_list()

        fig = make_subplots(rows=n_rows, cols=n_cols)

        for i, col_name in enumerate(
            fc_data.select_dtypes(float).columns.tolist(), start=1
        ):
            # Create a list of colors based on the qv and fc values
            color_list = []
            fc_indices = []

            for idx in np.arange(fc_data.shape[0]):
                if (
                    qv_data.loc[idx, col_name] < 0.05
                    and fc_data.loc[idx, col_name] > 0
                ):
                    color_list.append("lightcoral")

                elif (
                    qv_data.loc[idx, col_name] < 0.05
                    and fc_data.loc[idx, col_name] < 0
                ):
                    color_list.append("cornflowerblue")

                else:
                    color_list.append("lightgrey")

            # Create an annotation list based on fold change cutoff
            if qv_cutoff is True:
                if upper_fc_cutoff is not None and lower_fc_cutoff is None:
                    fc_indices = fc_data[
                        (qv_data[col_name] < 0.05)
                        & (fc_data[col_name] > upper_fc_cutoff)
                    ].index.to_list()

                elif lower_fc_cutoff is not None and upper_fc_cutoff is None:
                    fc_indices = fc_data[
                        (qv_data[col_name] < 0.05)
                        & (fc_data[col_name] < lower_fc_cutoff)
                    ].index.to_list()

                elif lower_fc_cutoff is not None and upper_fc_cutoff is not None:
                    fc_indices = fc_data[
                        (qv_data[col_name] < 0.05)
                        & (
                            (fc_data[col_name] > upper_fc_cutoff)
                            | (fc_data[col_name] < lower_fc_cutoff)
                        )
                    ].index.to_list()

            if qv_cutoff is False:
                if upper_fc_cutoff is not None and lower_fc_cutoff is None:
                    fc_indices = fc_data[
                        (fc_data[col_name] > upper_fc_cutoff)
                    ].index.to_list()

                elif lower_fc_cutoff is not None and upper_fc_cutoff is None:
                    fc_indices = fc_data[
                        (fc_data[col_name] < lower_fc_cutoff)
                    ].index.to_list()

                elif lower_fc_cutoff is not None and upper_fc_cutoff is not None:
                    fc_indices = fc_data[
                        (
                            (fc_data[col_name] > upper_fc_cutoff)
                            | (fc_data[col_name] < lower_fc_cutoff)
                        )
                    ].index.to_list()

            # Calculate row and col values for the subplots based on i
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1

            text = [
                gene + ", " + description
                for gene, description in zip(
                    fc_data["Genes"], fc_data["First.Protein.Description"]
                )
            ]

            # Add the scatter plots to the subplots layout
            fig.add_trace(
                go.Scatter(
                    x=fc_data[col_name],
                    y=pv_data[col_name],
                    mode="markers",
                    marker=dict(color=color_list),
                    text=text,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=fc_data.loc[indices + fc_indices, col_name],
                    y=pv_data.loc[indices + fc_indices, col_name],
                    mode="markers",
                    marker=dict(color="lightblue"),
                ),
                row=row,
                col=col,
            )

            # Add annotations to the subplots
            for idx in indices + fc_indices:
                fig.add_annotation(
                    dict(
                        x=fc_data.loc[idx, col_name],
                        y=pv_data.loc[idx, col_name],
                        text=fc_data.loc[idx, "Genes"],
                        font={"color": "black", "size": 12},
                        xref=f"x{i}",  # use proper xref
                        yref=f"y{i}",  # use proper yref
                    ),
                    row=row,
                    col=col,
                )

            y_max = pv_data[col_name].max()
            x_max = fc_data[col_name].max()
            x_min = fc_data[col_name].min()

            for coord, sample in zip([x_max, x_min], col_name.split("/")):
                fig.add_annotation(
                    dict(
                        x=coord,
                        y=y_max,
                        text=sample,
                        font={"color": "black", "size": 16},
                        showarrow=False,
                        bgcolor="rgba(245, 222, 179, 0.2)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=3,
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(title_text=f"log2 fold change")
            fig.update_yaxes(title_text=f"-log10 p-value")

        # Update the layout and traces
        fig.update_layout(
            template="simple_white",
            height=height,
            width=width,
            hoverlabel=dict(bgcolor="white", font_size=16),
        )
        fig.update_traces(marker=dict(size=8), selector=dict(mode="markers"))
        if save:
            # Save the plot as an HTML file
            pio.write_html(fig, f"{filename}")

        # Show the plot
        if is_jupyter_notebook():
            fig.show()
        else:
            st.plotly_chart(fig, use_container_width=True)

    def scatterplot_with_annotation(
        self,
        x,
        y,
        labels=None,
        pval_cutoff=None,
        lower_coef_cutoff=None,
        upper_coef_cutoff=None,
        annotation=None,
    ):
        sns.scatterplot(x=x, y=y)
        plt.axvline(lower_coef_cutoff, color="lightgrey", ls="--")
        plt.axvline(upper_coef_cutoff, color="lightgrey", ls="--")
        plt.axhline(pval_cutoff, color="lightgrey", ls="--")

        plt.ylabel("-log10 p-value")
        plt.xlabel("coefficient")

        if annotation:
            upper_mask = x > upper_coef_cutoff
            lower_mask = x < lower_coef_cutoff
            pval_mask = y > pval_cutoff
            mask = (upper_mask & pval_mask) | (lower_mask & pval_mask)

            x_coords = x[mask]
            y_coords = y[mask]
            annotation = labels[mask]

            texts = [
                plt.text(x_coord, y_coord, annot, ha="center", va="center", fontsize=9)
                for x_coord, y_coord, annot in zip(x_coords, y_coords, annotation)
            ]
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black"))

        plt.show()

    
    def create_pie_chart(self, data, title, colors=None):

        """
        Create a pie chart. Input data should
        be a Series with the categories as index.
        Like this: 
        data = pd.Series([10, 20, 30], index=['A', 'B', 'C'])

        Parameters
        ----------
        data : pd.Series
            A Series with the categories as index.
        
        title : str
            Title of the pie chart.
        
        colors : list
            List of colors to use in the pie chart. 
            If None, a default color palette is used. 
            Default is None.

        Returns
        -------
        fig, ax : tuple
            A tuple with the figure and axis objects.
        """

        sizes = data.values
        labels = data.index
        
        # Create a consistent color palette
        if colors is None:
            colors = sns.color_palette(n_colors=len(labels))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Function to format the labels with percentages and absolute values
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val:d})'
            return my_autopct
        
        # Create the pie chart
        wedges, _, autotexts = ax.pie(sizes, labels=None, colors=colors,
                                        autopct=make_autopct(sizes),
                                        startangle=90, pctdistance=0.75,
                                        wedgeprops=dict(edgecolor='black', linewidth=1.5))
        
        # Enhance the appearance of annotations
        plt.setp(autotexts, size=8, weight="bold")
        
        # Add a title
        ax.set_title(title, size=16, fontweight='bold')
        
        # Add a legend
        ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        
        return fig, ax
    
    def regression_plot(self, x, y, fc_data, cutoff_data, gene_list, pval, min=1, figsize=(8, 8), title=None):

        """
        Create a regression plot with annotation.

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        fc_data : pd.DataFrame
            DataFrame with fold change data.
        cutoff_data : pd.DataFrame
            DataFrame with p-values or q-values.
        gene_list : list
            List of genes to annotate.
        pval : bool
            If True, p-values are used. If False, q-values are used.
        min : int
            Minimum number of significant datapoints/comparisons for regression line.
            between 1 and 2. Default is 1.
        """

        gene_list_mask = fc_data["Genes"].isin(gene_list)

        if pval:
            cutoff_mask = (cutoff_data[[x, y]].select_dtypes(float)>1.3).sum(axis=1) >= min
        
        else:
            cutoff_mask = (cutoff_data[[x, y]].select_dtypes(float)<0.05).sum(axis=1)>= min

        plt.figure(figsize=figsize)

        if title:
            plt.title(title)
        plt.text(fc_data[x].min()-0.3, fc_data[y].min()-0.3,
                 f"Pearson r: {pearsonr(fc_data[cutoff_mask][x], fc_data[cutoff_mask][y])[0]:.2f}",
                 fontsize=8,
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle="round", fc="w", ec="black", alpha=0.3)
                 )
        plt.axhline(y=0, color='lightgrey', linestyle='--')
        plt.axvline(x=0, color='lightgrey', linestyle='--')

        sns.scatterplot(data=fc_data[~cutoff_mask],
                    x=x,
                    y=y,
                    color='lightgrey',
                    linewidth=0,
                    )

        sns.regplot(data=fc_data[cutoff_mask],
                x=x,
                y=y,
                color='lightblue',
                ci=False,
                truncate=False,
                line_kws={'linestyle': '--',
                        'linewidth':2},
                )
        sns.scatterplot(x=fc_data[x][gene_list_mask],
                    y=fc_data[y][gene_list_mask],
                    facecolors= 'none',
                    linewidth=0.5,
                    edgecolor="black")

        indices = fc_data[gene_list_mask].index
        texts =  [plt.text(fc_data[x][idx], fc_data[y][idx], fc_data['Genes'][idx], ha='center', va='center', fontsize=8) for idx in indices]
        adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'black', linewidth=0.5), )

        plt.xlabel(f"log2 fold change \n ({x})")
        plt.ylabel(f"log2 fold change \n ({y})")

