import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy import stats



pval_transf = lambda x: -np.log10(x)
int_transf = lambda x: np.log2(x)



class Confounder:

    def __init__(self):

        pass


    # some function

    def remove_confounder_with_linear_regression(self, data, confounder, model='linear'):

        """
        Remove confounder from data using linear regression.
        Parameters:
        
            data: pd.DataFrame
                dataframe with proteins as columns and samples as rows
            confounder: pd.DataFrame, np.array, list
                dataframe with confounder as colums and samples as rows
                or np.array or list with confounder values
        Returns:
            data: pd.DataFrame
                data with confounders removed
        """

        # create model 
        if model == 'linear':
            model = LinearRegression()
        elif model == 'lasso':
            model = Lasso()
        elif model == 'ridge':
            model = Ridge()
        else:
            raise ValueError('Model not supported.')      
        for protein in data.columns:
            # fit & prediction
            model.fit(confounder, data[protein])
            predicted = model.predict(confounder)

            # remove confounder
            data[protein] = data[protein] - predicted

        return data.T


    def calculate_r2_scores_for_multiple_targets(self, X, y, degrees=None, model='linear'):

        # create model 
        if model == 'linear':
            model = LinearRegression()
        elif model == 'lasso':
            model = Lasso()
        elif model == 'ridge':
            model = Ridge()
        elif model == 'random_forest':
            model = RandomForestRegressor()
        else:
            raise ValueError('Model not supported.')

        # create r2 data
        r2_scores = np.zeros(y.shape[1])
        pvals = np.zeros(y.shape[1])
        # create polynomial features
        if degrees:
            X = PolynomialFeatures(degree=degrees).fit_transform(X)

        # fit and prediction
        for index, id in enumerate(tqdm(y.columns)):
            model.fit(X, y[id])
            y_pred = model.predict(X)
            r2_scores[index] = r2_score(y[id], y_pred)
            pvals[index] = f_regression(X, y[id])[1]
        return r2_scores, pvals

    def combine_data(self, a, b, col_names=None):

        if col_names:
            if b.ndim > 1:
                for idx, col_name in  enumerate(col_names):
                    a[col_name] = b[:, idx]
            else:
                a[col_names] = b

        else:
            if b.ndim > 1:
                for idx in np.arange(len(col_names)):
                    a[idx] = b[:, idx]
            else:
                a[0] = b

        return a

    def data_preprocessing_for_confounder_removal(self, data, confounder):
        confounder = pd.get_dummies(confounder)
        data = data.select_dtypes(float).T
        return data, confounder

    def transform_only_for_numeric_data(self, data, func):
        for col in data.columns:
            if data[col].dtype == 'float64':
                data[col] = data[col].fillna(0).apply(func)

        return data.replace([np.inf, -np.inf], np.nan)
    
    def get_beta_and_pval_for_each_protein(self, X, y):

        pvals = np.zeros((y.shape[1], X.shape[1]))
        coefs = np.zeros((y.shape[1], X.shape[1]))

        for index in range(y.shape[1]):
            y_idx = y[:, index]

            # remove missing values and check if there is at least one
            # sample per confounder
            for group in range(X.shape[1]):
                if self._check_for_min_samples_per_group(X, y_idx, group):
                    
                    X_masked = X[self.mask]
                    y_masked = y_idx[self.mask]

                    pval, coef = self._get_beta_and_pvals(X_masked, y_masked)
                    pvals[index, :] = pval[1:]
                    coefs[index, :] = coef

                else:
                    pvals[index, :] = np.nan
                    coefs[index, :] = np.nan

        pvals = self._replace_0_with_min(pvals)

        return pvals, coefs
    
    def _check_for_min_samples_per_group(self, X, y):

        for group in range(X.shape[1]):

            self.mask = np.isfinite(y)

            _, b = np.unique(X[self.mask][:, group], return_counts=True, axis=0)

        return (b > 1).all()
    
    def _get_beta_and_pvals(self, X, y):

        model = self._fit_lin_model(X, y)

        predictions = model.predict(X)

        try:
            p_values = self._calculate_p_value_for_coefs(X, y, predictions, model)

        except np.linalg.LinAlgError:
            p_values = np.full((X.shape[1] + 1, ), fill_value=np.nan)

        coefs = model.coef_

        return p_values, coefs
    
    def _fit_lin_model(self, X, y):

        model = LinearRegression()
        model.fit(X, y)

        return model

    def _calculate_p_value_for_coefs(self, X,  y, predictions, model):

        newX = np.append(np.ones((len(X), 1)), X, axis=1)
        mse = (np.sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

        var_b = mse * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())

        var_b = np.maximum(var_b, 0)
        sd_b = np.sqrt(var_b)

        params = np.append(model.intercept_, model.coef_)
        ts_b = params / sd_b
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

        return p_values

    def _replace_0_with_min(self, pvals):
        pvals_filt = pvals[~(pvals == 0.0).any(axis=1)]
        mins = np.nan_to_num(pvals_filt, nan=1).min(axis=0)

        for col in range(pvals.shape[1]):
            pvals[:, col] = np.where(pvals[:, col] == 0.0, mins[col], pvals[:, col])

        return pvals
