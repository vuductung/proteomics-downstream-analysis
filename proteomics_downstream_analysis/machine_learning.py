import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import pandas as pd

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
        model : _type_
            Model to evaluate
        cv : _type_
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
        results = []
        results_w_diff_scores = []

        for scoring in scorings:
            results = []
            for idx in range(1, x_train.shape[1] + 1):
                result = cross_validate(
                    estimator=model,
                    X=x_train[:, :idx],
                    y=y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True
                )
                results.append({
                    'n_features': idx,
                    'train_score_mean': np.mean(result['train_score']),
                    'test_score_mean': np.mean(result['test_score']),
                    'train_score_std': np.std(result['train_score']),
                    'test_score_std': np.std(result['test_score']),
                })
            results_w_diff_scores.append({
                'scoring': scoring,
                'results': results
            })
        results = self._collect_data_from_no_of_feat_cvs(results_w_diff_scores, scorings)
        
        return results

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