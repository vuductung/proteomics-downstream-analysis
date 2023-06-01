import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

class ContaminationAnalysis:

    """ This class contains methods for detecting and removing contamination in a 2D dataframe. """
    
    def __init__(self):
        pass
    
    def zscore_outlier_plot(self, n_row, n_col, height=3, width=8, experimental_group=True, savefig=False):
    
        """
        Detecting outliers in a 2D dataframe using robust zscore and 3.5/-3.5 cutoff (upper and lower outliers)

        Parameters
        ----------
        n_row : int
            Number of rows in the subplot
            
        n_col : int
            Number of columns in the subplot
            
        height : int
            Height of the figure
             (Default value = 3)
        width : int
            Width of the figure
             (Default value = 8)
        experimental_group : bool
            If True robust zscore will be calculated across
            experimental group. If false, across entire dataframe
             (Default value = True)

        savefig : bool
            If True, figure will be saved as pdf
             (Default value = False)

        Returns
        -------
        outliers : list
            List of outliers with sample name and sample measurement order
        
        """
        # calculate zscore across each experimental group
        if experimental_group == True: 
            testing_data = self.data.select_dtypes('float')
            
            # prepare dataset for zscore calculation. Retain column names and corresponding index
            columns=[*zip(testing_data.columns, range(len(testing_data.columns)))]
            testing_data.columns = pd.MultiIndex.from_tuples(columns)
            testing_data = testing_data.sort_index(level=0, axis=1)

            frequency_list = []
            
            # calculate zscore across each experimental group
            for col in  sorted(list(set(i[0] for i in testing_data.columns))):
                
                median = testing_data[col].median(axis=1).values.reshape(-1,1)
                mad = stats.median_abs_deviation(testing_data[col], axis=1).reshape(-1,1)

                zscore_data = (0.6745 * (testing_data[col]-median))/mad
                above = (zscore_data > 3.5).sum(axis=0)
                below = (zscore_data < -3.5).sum(axis=0)

                sum_below_above = above + below

                frequency_list = frequency_list + list(sum_below_above.values)
            
            # create outlier frequency dataframe
            frequ_data = pd.DataFrame(np.column_stack([frequency_list, [i[1] for i in testing_data.columns], [i[0] for i in testing_data.columns]]), 
                                        columns=['frequency', 'sample measurment order', 'sample name'])

            frequ_data['frequency'] = frequ_data['frequency'].astype('float')
            
            outliers = []
            
            fig, ax = plt.subplots(n_row, n_col)
            fig.set_figheight(height)
            fig.set_figwidth(width)

            for col_name, axes in zip(sorted(list(set(i[0] for i in testing_data.columns))), ax.flat):
                barplot_data = frequ_data[frequ_data['sample name']==col_name]
                
                # calculate cutoff for each experimental group
                q1 = np.percentile(barplot_data['frequency'], 25)
                q3 = np.percentile(barplot_data['frequency'], 75)
                iqr = q3-q1
                upper_limit = q3 + 1.5*iqr
                
                # collect outliers
                outlier_data = barplot_data[barplot_data['frequency']>upper_limit]
                outlier = [(sample, idx) for sample, idx in zip(outlier_data['sample name'], outlier_data['sample measurment order'])]
                outliers += outlier
                
                # create a subplot
                axes.set_title(col_name)
                sns.barplot(data=barplot_data, y='frequency', x='sample measurment order', color='lightgrey', ax=axes)
                axes.axhline(y=upper_limit, color='lightgrey', linestyle='--',)
                plt.setp(axes, ylabel='outlier frequency')
                sns.despine()
            
            fig.tight_layout() 
            
            if savefig == True:
                fig.savefig('zscore_outlier_plot.pdf', bbox_inches='tight')
            
            print(outliers)
            
            return outliers
        
        # calculate zscore across entire dataframe
        else:
            testing_data = self.data.select_dtypes('float')

            median = testing_data.median(axis=1).values.reshape(-1,1)
            mad = stats.median_abs_deviation(testing_data, axis=1).reshape(-1,1)
            zscore_data = (0.6745 * (testing_data-median))/mad
            above = (zscore_data > 3.5).sum(axis=0)
            below = (zscore_data < -3.5).sum(axis=0)

            # create frequency data
            sum_below_above = above + below
            sum_below_above_data = pd.DataFrame(sum_below_above, columns=['value'])
            sum_below_above_data = sum_below_above_data.reset_index().rename(columns={'index':'strain'})

            # calculate cutoff 
            q1 = np.percentile(sum_below_above_data['value'], 25)
            q3 = np.percentile(sum_below_above_data['value'], 75)
            iqr = q3-q1
            upper_limit = q3 + 1.5*iqr
            
            # plot data
            plt.figure(figsize=(12,6))
            sns.barplot(y=sum_below_above_data['value'], x=sum_below_above_data.index, hue=sum_below_above_data['strain'], )
            plt.axhline(y=upper_limit, color='lightgrey', linestyle='--')
            plt.xlabel('sample measurement order')
            plt.ylabel('outlier frequency')
            sns.despine()
            if savefig == True:
                fig.savefig('zscore_outlier_plot.pdf', bbox_inches='tight')
            
            plt.show()

    def drop_zscore_outlier(self):
    
        """
        Detecting outliers in a 2D dataframe using robust zscore and 3.5/-3.5 cutoff (upper and lower outliers)
                
        Returns
        -------
        self.data: pandas dataframe
            dataframe with outliers removed

        """
        # calculate zscore across each experimental group
        testing_data = self.data.select_dtypes('float')

        # prepare dataset for zscore calculation. Retain column names and corresponding index
        columns=[*zip(testing_data.columns, range(len(testing_data.columns)))]
        testing_data.columns = pd.MultiIndex.from_tuples(columns)
        testing_data = testing_data.sort_index(level=0, axis=1)

        frequency_list = []

        # calculate zscore across each experimental group
        for col in sorted(list(set(i[0] for i in testing_data.columns))):
            
            median = testing_data[col].median(axis=1).values.reshape(-1,1)
            mad = stats.median_abs_deviation(testing_data[col], axis=1).reshape(-1,1)

            zscore_data = (0.6745 * (testing_data[col]-median))/mad
            above = (zscore_data > 3.5).sum(axis=0)
            below = (zscore_data < -3.5).sum(axis=0)

            sum_below_above = above + below

            frequency_list = frequency_list + list(sum_below_above.values)

        # create outlier frequency dataframe
        frequ_data = pd.DataFrame(np.column_stack([frequency_list, [i[1] for i in testing_data.columns], [i[0] for i in testing_data.columns]]), 
                                    columns=['frequency', 'sample measurment order', 'sample name'])

        frequ_data['frequency'] = frequ_data['frequency'].astype('float')

        outliers = []

        for col_name in sorted(list(set(i[0] for i in testing_data.columns))):
            barplot_data = frequ_data[frequ_data['sample name']==col_name]
            
            # calculate cutoff for each experimental group
            q1 = np.percentile(barplot_data['frequency'], 25)
            q3 = np.percentile(barplot_data['frequency'], 75)
            iqr = q3-q1
            upper_limit = q3 + 1.5*iqr
            
            # collect outliers
            outlier_data = barplot_data[barplot_data['frequency']>upper_limit]
            outlier = [(sample, idx) for sample, idx in zip(outlier_data['sample name'], outlier_data['sample measurment order'])]
            outliers += outlier

        outliers_idx = [int(i[1]) + 5 for i in outliers]
        inliers = [i for i in range(len(self.data.columns)) if i not in outliers_idx]
        self.data = self.data.iloc[:, inliers]

        return self.data
    
    def assess_blood_contam(self, contam_panel, kind='scatter', shift=3, figsize=(12, 3), savefig=False):

        """Assess blood contamination

        Parameters
        ----------
        contam_panel : pandas dataframe
            dataframe containing the list of blood contamination proteins
            
        kind : str
            plot type. Options: 'scatter' or 'histplot' (Default value = 'scatter')
             (Default value = 'scatter')

        shift : int
            shift the x-axis by a certain number of units (Default value = 3)
             (Default value = 3)

        figsize : tuple
            figure size (Default value = (12, 3)
             (Default value = (12, 3)
            
        savefig :
             (Default value = False)

        Returns
        -------
        plot
        """

        contam_data_list = []
        for i in contam_panel['Type'].unique():
            
            # calculate the sum of all proteins intensities and all contamination intensities
            
            total_protein_sum = self.data.select_dtypes('float').sum(axis=0)
            contams = contam_panel[contam_panel['Type'] == i]['Gene names']
            total_contam_sum = self.data[self.data['Genes'].isin(contams)].sum(axis=0, numeric_only=True)
            
            # calculate the contamination ratio 
            contam_data = pd.DataFrame(total_contam_sum/total_protein_sum*100).T
            contam_data['contam_type'] = i
            contam_data_list.append(contam_data)
        
        # prepare dataframe for plotting
        cont_data_concat = pd.concat(contam_data_list)
        column_names = [str(i) for i in range(total_protein_sum.shape[0])]
        cont_data_concat.columns = column_names + ['contam_type']
        cont_data_concat_melt = cont_data_concat.melt(id_vars='contam_type')
        
        if kind == 'scatter':
        
            for i in cont_data_concat['contam_type']:

                # create an unbiased cutoff using upper limit in boxplot (75th percentile + iqr * 1.5)
                q1 = np.percentile(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], 25)
                q3 = np.percentile(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], 75)
                iqr = q3-q1    
                upper_limit = q3 + 1.5*iqr

                # create plot for each contamination type 
                plt.figure(figsize=figsize)
                plt.title(i)
                sns.scatterplot(data=cont_data_concat_melt[cont_data_concat_melt['contam_type']==i], x= 'variable', y='value', hue='contam_type')
                plt.legend(loc='upper right')
                plt.xlabel('sample_number')
                plt.ylabel('contamination in %')
                plt.axhline(y=upper_limit, linestyle='--')
                
                if savefig == True:
                    plt.savefig('blood_contam_plot.pdf', bbox_inches='tight')
                plt.show()
                
        elif kind == 'hist' and shift != None:
            
            # assessing data quality using contam panel and biased cutoff (mean + 1 x STD)
            for i in cont_data_concat['contam_type']:
                mean = np.mean(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'])
                std = np.std(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'])
                plt.title(i)
                sns.histplot(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], bins=15)
                plt.axvline(x=mean, color='lightgrey', linestyle='--')
                plt.axvline(x=mean + std*shift, color='red', linestyle='--')
                plt.show()
        
        else:
            
            # assessing data quality using contam panel and unbiased cutoff (upper limit in boxplot)
            for i in cont_data_concat['contam_type']:
                q1 = np.percentile(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], 25)
                q3 = np.percentile(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], 75)
                iqr = q3-q1    
                upper_limit = q3 + 1.5*iqr
                plt.title(i)
                sns.histplot(cont_data_concat_melt[cont_data_concat_melt['contam_type']==i]['value'], bins=15)
                plt.axvline(x=upper_limit, color='lightgrey', linestyle='--')
                plt.show()
