import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

class BaselineModel:
    def __init__(self, y_train, X_val, y_val):
        """
        Initialize the BaselineModel with training and validation data.

        Parameters
        ----------
        y_train : pd.Series or np.ndarray
            Target values for the training set.
        X_val : pd.DataFrame or np.ndarray
            Features of the validation set.
        y_val : pd.Series or np.ndarray
            Target values for the validation set.
        """
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.y_mode = y_train.mode()[0]  # Mode of the target variable

    def compute_baseline_predictions(self):
        """
        Compute baseline predictions based on the mode of y_train.

        Returns
        -------
        np.ndarray
            Array filled with the mode value for the validation set.
        """
        y_shape = len(self.X_val)
        return np.full(y_shape, self.y_mode)

    def calculate_auroc(self):
        """
        Calculate the AUROC for the baseline model.

        Returns
        -------
        float
            The AUROC score.
        """
        y_base_proba = np.full(len(self.y_val), 1.0 if self.y_mode == 1 else 0.0)
        return roc_auc_score(self.y_val, y_base_proba)

    def plot_roc_curve(self):
        """
        Plot the ROC curve for the baseline model.
        """
        y_base_proba = np.full(len(self.y_val), 1.0 if self.y_mode == 1 else 0.0)
        fpr, tpr, _ = roc_curve(self.y_val, y_base_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Baseline Model')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate(self):
        """
        Run the evaluation of the baseline model, calculate AUROC, and plot the ROC curve.

        Returns
        -------
        float
            The AUROC score.
        """
        auroc = self.calculate_auroc()
        print(f'AUROC score for the baseline model: {auroc:.6f}')
        self.plot_roc_curve()
        return auroc
