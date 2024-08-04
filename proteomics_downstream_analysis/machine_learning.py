import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class MachineLearning:

    """
    Machine Learning methods
    """
    def __init__(self):
        pass

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
