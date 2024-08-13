import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from pathlib import Path


class ContaminationAnalysis:
    """
    This class contains methods for detecting and removing contamination in a dataframe (2D array).
    """

    def __init__(self):
         # Get the directory of the current file
        current_dir = Path(__file__).parent.resolve()
        
        # Construct the path to the Excel file
        filepath = current_dir / ".." / "data" / "contamination" / "contam_panel.xlsx"
        self.panel = pd.read_excel(filepath)

        self.from_orig_to_sorted_indices = None
        self.from_sorted_to_orig_indices = None

    def outlier(self, data, kind="zscore", remove=False, contam_type="RBC"):
        """
        docstring
        An outlier algorithm using the z-score and the IQR to detect outliers

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed

        kind : string
            what kind of outlier detection algorithm to use. 'zscore' for
            zscore algorithm, 'contamination' to detect e.g. RBC contamination
            and 'missing values' for detecting outliers based on the numbers
            of missing valuesrot

        remove : boolean
            if True removes outliers from dataset

        panel: pd.DataFrame
            a panel that includes the kind of contamination and the Protein.Ids
            of each contamination

        type = string
            What kind of contamination should be used e.g. 'RBC' for red blood
            cell

        Returns
        -------

        array1: np.array
            outliers with their index (excluding the string columns)
        array2 : np.array
            outliers with boolean values (True is an outlier and False
            is inlier)

        or

        data : pd.DataFrame
            data without outliers

        """
        outlier_master_mask = np.array([], dtype=bool)
        sorted_data = self._sort_by_column_names(data).reset_index()
        columns = sorted_data.select_dtypes(float).columns.unique()

        for i in columns:
            if kind == "zscore":
                outlier_mask = self._compute_zscore_outliers(sorted_data[i])[-1]
                outlier_master_mask = np.concatenate((outlier_master_mask, outlier_mask))

            elif kind == "contamination":
                outlier_mask = self._compute_contamination_outlier(
                    sorted_data[["Genes", i]], contam_type
                )[-1]
                outlier_master_mask = np.concatenate((outlier_master_mask, outlier_mask))

            elif kind == "missing values":
                outlier_mask = self._compute_missing_values_outlier(sorted_data[i])[-1]
                outlier_master_mask = np.concatenate((outlier_master_mask, outlier_mask))

        _ = self._sort_by_column_names(data) # update from_sorted_to_orig_indices attribute
        outlier_master_mask_resorted =  outlier_master_mask[self.from_sorted_to_orig_indices]

        if remove:
            inlier_indices = np.where(~(outlier_master_mask_resorted))[0]
            return self._remove_outliers(data, inlier_indices)

        else:
            return outlier_master_mask_resorted

    def outlier_plot(
        self,
        data,
        plot="bar",
        kind="zscore",
        contam_type="RBC",
        figsize=(20, 5),
        filepath=None,
    ):
        groups = data.select_dtypes(float).columns.unique()
        len_groups = len(groups)
        fig, ax = plt.subplots(1, len_groups, figsize=figsize)

        if kind == "zscore":
            output = [self._compute_zscore_outliers(data[i])[1:3] for i in groups]

        elif kind == "contamination":
            output = [
                self._compute_contamination_outlier(data[["Genes", i]], contam_type)[:2]
                for i in groups
            ]

        elif kind == "missing values":
            output = [self._compute_missing_values_outlier(data[i])[:2] for i in groups]

        # plot the data
        for values, axes, group in zip(output, ax.flat, groups):
            outliers = values[0]
            upper_lim = values[1]

            if plot == "bar":
                sns.barplot(
                    x=[*range(len(outliers))], y=outliers, color="lightgrey", ax=axes, width=1
                )

                axes.set_title(f"{group}")
                axes.set_ylabel("outlier frequency")
                axes.set_xticks([])
                axes.axhline(y=upper_lim, color="red", linestyle="--")

            if plot == "hist":
                sns.histplot(x=outliers, color="lightgrey", ax=axes)

                axes.set_title(f"{group}")
                axes.set_ylabel("outlier frequency")
                axes.axvline(x=upper_lim, color="red", linestyle="--")

        fig.tight_layout()
        if filepath:
            fig.savefig(filepath, bbox_inches="tight", transparent=True)
            
    def _compute_zscore_outliers(self, data):
        # sort data
        data = self._sort_by_column_names(data)

        # calculate robus zscores
        robust_zscores = self._robust_zscore(data)

        # get the protein outliers
        protein_outliers = self._number_of_protein_outliers(robust_zscores)

        # get the upper limit (boxplot)
        upper_lim = self._upper_limit(protein_outliers)

        # get the mask to filter outliers
        outlier_mask = self._get_outlier_mask(upper_lim, protein_outliers)

        return robust_zscores, protein_outliers, upper_lim, outlier_mask
    
    def _compute_contamination_outlier(self, data, contam_type="RBC"):
        # sort data
        data = self._sort_by_column_names(data)

        # calculate rbc to total protein ratio
        rbc_total_ratio = self._compute_rbc_total_ratio(data, contam_type)

        # get the upper limit (boxplot)
        upper_lim = self._upper_limit(rbc_total_ratio)

        # get the mask to filter outliers
        mask = self._get_outlier_mask(upper_lim, rbc_total_ratio)

        return rbc_total_ratio, upper_lim, mask
    
    def _compute_missing_values_outlier(self, data):
        # sort data
        data = self._sort_by_column_names(data)

        # count missing values
        nan_values = self._count_missing_values(data)

        # get the upper limit (boxplot)
        upper_lim = self._upper_limit(nan_values)

        # get the mask to filter outliers
        mask = self._get_outlier_mask(upper_lim, nan_values)

        return nan_values, upper_lim, mask

    def _sort_by_column_names(self, data, id_data=None, col_name=None):

        if id_data is None:
            col_names = data.select_dtypes("float").columns
            ids = list(np.arange(len(col_names)))
            id_data = pd.DataFrame({"col_name": col_names,
                                    "ID": ids})
            sorted_id_data = id_data.sort_values("col_name")
        
        else:
            sorted_id_data = id_data.sort_values(col_name)

        self.from_orig_to_sorted_indices = sorted_id_data.index.tolist()
        self.from_sorted_to_orig_indices = sorted_id_data.reset_index().sort_values("index").index.tolist()

        # sort data by column names
        non_float_columns = data.select_dtypes(exclude="float").reset_index().columns.tolist()

        reindexed_data = data.reset_index().set_index(non_float_columns)

        reindexed_data_sorted = reindexed_data.sort_index(axis=1)

        return reindexed_data_sorted
    
    def _robust_zscore(self, data):
        # get the robust zscore
        data = data.select_dtypes("float")
        median = data.median(axis=1).values.reshape(-1, 1)
        mad = stats.median_abs_deviation(data, axis=1, nan_policy="omit").reshape(-1, 1)
        robust_zscores = 0.6745 * (data - median) / mad
        return robust_zscores
    
    def _number_of_protein_outliers(self, array):
        # compute the number of outliers for each protein
        protein_outliers = (np.abs(array) > 3.5).sum(axis=0)
        return protein_outliers.values

    def _upper_limit(self, protein_outliers):
        # compute the upper limit for outliers
        q1 = np.percentile(protein_outliers, 25)
        q3 = np.percentile(protein_outliers, 75)
        iqr = q3 - q1
        upper_lim = q3 + 3 * iqr

        return upper_lim
    
    def _compute_rbc_total_ratio(self, data, contam_type="RBC"):
        # calculat the RBC to total protein ratio
        total = data.sum(axis=0, numeric_only=True)
        contam = self.panel[self.panel["Type"] == contam_type]["Gene names"].tolist()
        rbc_sum = data[data.reset_index()["Genes"].isin(contam).values].sum(axis=0, numeric_only=True)
        rbc_total_ratio = rbc_sum / total

        return rbc_total_ratio.values

    def _get_outlier_mask(self, upper_lim, protein_outliers):
        # get the mask to filter outliers
        mask = protein_outliers >= upper_lim

        return mask
    
    def _count_missing_values(self, data):

        nan_values = data.select_dtypes(float).isna().sum(axis=0).values

        return nan_values
    
    def _remove_outliers(self, data, inlier_indices):
        # set all string columns to index
        reindexed_data = data.set_index(data.select_dtypes("string").columns.tolist())
        inlier_data = reindexed_data.iloc[:, inlier_indices]
        return inlier_data
    
    def _set_all_str_to_index(self, data):
        # set all string columns to index
        data = data.set_index(data.select_dtypes("string").columns.tolist())
        return data

    def _missing_values_outlier(self, data, experimental=True, remove=False):
        if experimental:
            master_mask = np.array([], dtype=bool)
            for i in data.select_dtypes(float).columns.unique():
                _, _, mask = self._compute_missing_values_outlier(data[i])
                master_mask = np.concatenate((master_mask, mask))

        else:
            _, _, master_mask = self._compute_missing_values_outlier(data)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes("string").columns)

        if remove:
            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]
            return data

        else:
            return np.where(master_mask)[0] + number_of_string_cols

    def zscore_outlier(self, data, experimental=True, remove=False):
        # sort data
        data = self._sort_by_column_names(data)

        # compute the robust zscore to find outliers
        # within experimental group or all data
        if experimental:
            master_mask = np.array([], dtype=bool)

            for i in data.select_dtypes(float).columns.unique():
                _, _, _, mask = self._compute_zscore_outliers(data[i])
                master_mask = np.concatenate((master_mask, mask))

        else:
            _, _, _, master_mask = self._compute_zscore_outliers(data)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes("string").columns)

        if remove:
            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]

            return data

        else:
            return np.where(master_mask)[0] + number_of_string_cols
    
    def contamination_outlier(
        self, data, contam_type="RBC", remove=False, experimental=True
    ):
        # compute contamination outliers
        if experimental:
            master_mask = np.array([], dtype=bool)
            for i in data.select_dtypes(float).columns.unique():
                _, _, mask = self._compute_contamination_outlier(
                    data[["Genes", i]], contam_type
                )
                master_mask = np.concatenate((master_mask, mask))

        else:
            _, _, master_mask = self._compute_contamination_outlier(data, contam_type)

        # get number of string cols to correct the inliers/outliers index
        number_of_string_cols = len(data.select_dtypes("string").columns)

        if remove:
            inliers = np.where(~(master_mask))[0]
            inliers = inliers + number_of_string_cols
            inliers = np.concatenate((np.arange(number_of_string_cols), inliers))
            data = data.iloc[:, inliers]
            return data

        else:
            return np.where(master_mask)[0] + number_of_string_cols

    def contamination_outlier_plot(self, data, experimental=True, type="RBC"):
        # compute the robust zscore to find outliers
        if experimental:
            groups = data.select_dtypes(float).columns.unique()
            len_groups = len(groups)
            fig, ax = plt.subplots(1, len_groups, figsize=(20, 5))

            for group, axes in zip(groups, ax.flat):
                rbc_total_ratio, upper_lim, _ = self._compute_contamination_outlier(
                    data[group], type
                )

                # plot the data
                sns.barplot(
                    x=[*range(len(rbc_total_ratio))],
                    y=rbc_total_ratio,
                    color="lightgrey",
                    ax=axes,
                )

                axes.set_ylabel("outlier frequency")
                axes.axhline(y=upper_lim, color="red", linestyle="--")

            fig.tight_layout()

    def missing_values_outlier_plot(self, data, experimental):
        # compute the robust zscore to find outliers
        if experimental:
            groups = data.select_dtypes(float).columns.unique()
            len_groups = len(groups)
            fig, ax = plt.subplots(1, len_groups, figsize=(20, 5))

            for group, axes in zip(groups, ax.flat):
                nan_values, upper_lim, _ = self._compute_missing_values_outlier(
                    data[group]
                )

                # plot the data
                sns.barplot(
                    x=[*range(len(nan_values))],
                    y=nan_values,
                    color="lightgrey",
                    ax=axes,
                )

                axes.set_ylabel("outlier frequency")
                axes.axhline(y=upper_lim, color="red", linestyle="--")

            fig.tight_layout()
