import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

class MachineLearning:

    """
    Machine Learning methods
    """
    def __init__(self):
        self.scorings = ["roc_auc", "accuracy", "precision", "recall", "f1"]

    def plot_confusion_matrix(self, predicted_labels_list, y_test_list, figsize=(2, 2)):

        """
        This function prints and plots the confusion matrix.
        """
        
        cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure(figsize=figsize)
        self._generate_confusion_matrix(cnf_matrix, classes=[0, 1])

    def _generate_confusion_matrix(self, cnf_matrix, classes):
        cnf_matrix_raw = cnf_matrix.copy()
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cnf_matrix.max() / 2.0

        for i, j in itertools.product(
            range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
        ):
            plt.text(
                j,
                i,
                f"{cnf_matrix_raw[i , j]} {cnf_matrix[i , j]*100:.2f}%",
                horizontalalignment="center",
                color="white" if cnf_matrix[i, j] > thresh else "black",
                size=8,
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        return cnf_matrix
    
    def cross_validate_w_no_of_features(self, model, cv, x_train, y_train, scorings):
        """
        This method is used to cross validate the model with different number of features.
        Parameters
        ----------
        model : *type*
            Model to evaluate
        cv : *type*
            cv object
        scorings : list
            List of scoring functions.
        x_train : np.array
            x_train. Shape: (n_samples, n_features)
            Features must be sorted in descending
            order of importance.
        y_train : np.array
            y_train
        Returns
        -------
        list
            List of results with different number of features.
        """
        results_w_diff_scores = []

        # Use all available cores, leave one free
        n_jobs = -2  # -1 would use all cores, -2 leaves one core free

        for scoring in scorings:
            print(f"Scoring: {scoring}")
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._incrementally_add_one_feature_for_cross_val)(idx, model, x_train, y_train, cv, scoring)
                for idx in tqdm(range(1, x_train.shape[1] + 1))
            )
            
            results_w_diff_scores.append({
                'scoring': scoring,
                'results': results
            })

        results = self._collect_data_from_no_of_feat_cvs(results_w_diff_scores, scorings)
        return results

    
    def _incrementally_add_one_feature_for_cross_val(self, idx, model, x_train, y_train, cv, scoring):
        
        result = cross_validate(
                                estimator=model,
                                X=x_train[:, :idx],
                                y=y_train,
                                cv=cv,
                                scoring=scoring,
                                return_train_score=True
                            )
        return {
            'n_features': idx,
            'train_score_mean': np.mean(result['train_score']),
            'test_score_mean': np.mean(result['test_score']),
            'train_score_std': np.std(result['train_score']),
            'test_score_std': np.std(result['test_score']),
        }
    
    def calculate_combined_scores(self, train_roc_aucs, test_roc_aucs):
        scores = {}
        
        # Normalized Difference Score
        norm_diff = [test * (1 - (train - test) / train) for train, test in zip(train_roc_aucs, test_roc_aucs)]
        scores['normalized_difference'] = norm_diff
        
        # F1-like Score
        overfitting = [train - test for train, test in zip(train_roc_aucs, test_roc_aucs)]
        non_overfitting = [1 - o for o in overfitting]
        f1_like = [2 * (test * non_over) / (test + non_over) for test, non_over in zip(test_roc_aucs, non_overfitting)]
        scores['f1_like'] = f1_like
        
        # Area Under the Margin
        aum = np.trapz(y=np.array(train_roc_aucs) - np.array(test_roc_aucs), x=range(1, len(train_roc_aucs) + 1))
        aum_score = [test - (aum / len(train_roc_aucs)) for test in test_roc_aucs]
        scores['area_under_margin'] = aum_score

        return scores

    def _collect_data_from_no_of_feat_cvs(self, datasets, scorings):
        return {score: pd.DataFrame(data["results"]) for data, score in zip(datasets, scorings)}
    
    def plot_no_of_feat_cv_results(self, results):
        """
        Plot the results of cross validation
        with different number of features.

        Parameters
        ----------
        results : list
            List of results with different number of features.
        """
        for key in results.keys():
            data_to_plot = results[key]
            plt.figure(figsize=(2, 2))
            plt.plot(data_to_plot["n_features"], data_to_plot["train_score_mean"], label="Train")
            plt.fill_between(data_to_plot["n_features"].values,
                            data_to_plot["train_score_mean"].values - data_to_plot["train_score_std"].values,
                            data_to_plot["train_score_mean"].values + data_to_plot["train_score_std"].values,
                            alpha=0.2)
            
            plt.plot(data_to_plot["n_features"], data_to_plot["test_score_mean"], label="Test")
            plt.fill_between(data_to_plot["n_features"].values,
                            data_to_plot["test_score_mean"].values - data_to_plot["test_score_std"].values,
                            data_to_plot["test_score_mean"].values + data_to_plot["test_score_std"].values,
                            alpha=0.2)
                            
            plt.xlabel("Number of features")
            plt.ylabel(key)
            plt.ylim(0.5, 1)
            plt.legend()
            plt.show()
    
    def group_proteins_by_value(self, protein_dict):
        grouped = {}
        for protein, value in protein_dict.items():
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(protein)
        return grouped

    def get_feature_importance(self, model, X, y, n_repeats=10):
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
        return result.importances_mean
    
    def align_two_dataframes_based_on_gene_names(self, data1, data2, selected_features_mask):
        """This method aligns two dataframes based on common Gene names.
        data1 represents the train data while data2 the test data. Before
        alignment a selected_feature_mask has to be created. The idea behind
        creating a selected_feature_mask for the train data before aligning
        the train and test data is to make ensure unbiased feature selection
        only based on the train data. Alignment and filtering is done subsequently.

        Parameters
        ----------
        data1 : pd.DataFrame
            Train data. Has to have "Genes" column.
        data2 : pd.DataFram
            Test data. Has to have "Genes" column.
        selected_features_mask : np.array
            mask of selected features.

        Returns
        -------
        pd.DataFrame
            train and test data aligned based on common Gene names.
        """        
        selected_genes = data1[selected_features_mask]["Genes"].tolist()

        data2_filt_by_selected_genes = data2[data2["Genes"].isin(selected_genes)]
        data2_filt_by_selected_genes = data2_filt_by_selected_genes.groupby("Genes").mean()
        data2_filt_by_selected_genes = data2_filt_by_selected_genes.sort_index()

        data1_data2_intersec_proteins = data2_filt_by_selected_genes.index.values

        data1_filt_by_selected_genes = data1[data1["Genes"].isin(data1_data2_intersec_proteins)].groupby("Genes").mean()
        data1_filt_by_selected_genes = data1_filt_by_selected_genes.sort_index()

        return data2_filt_by_selected_genes, data1_filt_by_selected_genes