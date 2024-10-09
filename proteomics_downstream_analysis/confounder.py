import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy.stats import combine_pvalues

pval_transf = lambda x: -np.log10(x)
int_transf = lambda x: np.log2(x)


class Confounder:

    def __init__(self):

        pass

    def encoder(self, data):
        cat_variable_mask = ~((data.dtypes == 'float') | (data.dtypes == 'int')).values
        cat_variables = data.iloc[:,  cat_variable_mask]
        n_unique_vals = cat_variables.nunique()

        get_dummies_col = (n_unique_vals > 2).values

        dummies = pd.get_dummies(cat_variables.iloc[:, get_dummies_col])

        label_encoder = LabelEncoder()

        mappings = {}
        for col in cat_variables.iloc[:, ~get_dummies_col].columns:
            dummies.loc[:, col] = label_encoder.fit_transform(cat_variables[col])
            mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        encoded_data = pd.concat([data.iloc[:, ~cat_variable_mask], dummies.astype(int)], axis=1)

        return encoded_data, mappings

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
                if self._check_for_min_samples_per_group(X, y_idx):
                    
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


    def linear_regression_with_confounder(self, protein_intensity_data, confounders, iterations, variable,):

        """
        Perform linear regression with confounder variables.

        Parameters
        ----------
        protein_intensity_data : pd.DataFrame
            pd.DataFrame containing protein intensity data.
            Index has to be the gene names, columns have to be the samples.
            Protein intensity has to be log2 transformed.
        confounders : pd.DataFrame
            Dataframe containing confounder variables.
            Index has to be the sample names,
            columns have to be the confounder variables.
            Confounders have to be all numerical values,
            and standardization is not required. Will
            be done internally.
        iterations : int
            Number of iterations to perform.
        variable : str
            Name of the confounder variable.

        Returns
        -------
        coef_data : pd.DataFrame
            Dataframe containing coefficients.
        pval_data : pd.DataFrame        
            Dataframe containing p-values.
        qval_data : pd.DataFrame    
            Dataframe containing q-values.
        """

        # remove missing values
        finite_mask = np.isfinite(confounders).all(axis=1).values # create mask to remove missing values from input
        selected_confounders = confounders[finite_mask].reset_index(drop=True)
        genes = protein_intensity_data.index
        protein_intensity_data = protein_intensity_data.T[finite_mask].reset_index(drop=True).values
        
        seeds = []
        three_d_pvalues = []
        three_d_betas = []

        for it in tqdm(range(iterations)):
            seeds.append(it)

            np.random.seed(it)

            # check if the size of sample > pos/neg classes
            pos_class = selected_confounders[selected_confounders[variable]==1]
            non_pos_class = selected_confounders[selected_confounders[variable]==0]
            pos_class_size = pos_class.shape[0]
            non_pos_class_size = non_pos_class.shape[0]

            sample_size = np.min([pos_class_size, non_pos_class_size])
            pos_class_sample = pos_class.sample(sample_size)
            non_pos_class_sample = non_pos_class.sample(sample_size)

            X_sampled = pd.concat([non_pos_class_sample, pos_class_sample])

            # sample X and y based on the indices
            sampled_indices = X_sampled.index
            y_sampled = protein_intensity_data[sampled_indices]

            # Ensure correct initialization shape
            pvals = []
            coefs = []

            confounder_cols = X_sampled.columns

            scaler = StandardScaler()
            y = np.nan_to_num(y_sampled, 0)
            X_std = scaler.fit_transform(X_sampled)
            X_const = sm.add_constant(X_std)

            for protein in range(y.shape[1]):

                # remove missing values
                finite_mask = np.isfinite(y[:, protein])
                y_filt = y[finite_mask,  protein]
                X_filt = X_const[finite_mask]
                
                try:
                    # Fit the model
                    model = sm.OLS(y_filt, X_filt).fit()
                    # Store the p-values and coefficients
                    pvals.append(model.pvalues)
                    coefs.append(model.params)

                except Exception as e: # for any kind of error, set all to nan (will be updated later)
                    pvals.append(np.full((4, ) ,np.nan))
                    coefs.append(np.full((4, ) ,np.nan))

            pvalues = np.vstack(pvals)[:, :, np.newaxis]
            betas = np.vstack(coefs)[:, :, np.newaxis]

            three_d_pvalues.append(pvalues)
            three_d_betas.append(betas)

        pvals_concat = np.concatenate(three_d_pvalues, axis=2)
        betas_concat = np.concatenate(three_d_betas, axis=2)

        # pvals_combined = np.median(pvals_concat, axis=2)
        pvals_combined = combine_pvalues(pvals_concat, method="tippett", axis=2)[1]
        betas_combined = np.median(betas_concat, axis=2)

        # PVALUE ADJUSTMENT
        qvals = np.full(pvals_combined.flatten().shape, np.nan)
        mask = np.isfinite(pvals_combined.flatten()) 
        qvals[mask] = fdrcorrection(pvals_combined.flatten()[mask])[1]
        qvalues = qvals.reshape(pvals_combined.shape)

        # CREATE DATAFRAME FOR PVALUE, COEFFICIENTS AND QVALUES 
        cols = np.array(confounder_cols) + "/" + np.array("no " + confounder_cols)

        pval_data = pd.DataFrame(data=-np.log10(pvals_combined)[:, 1:], columns=cols)
        coef_data = pd.DataFrame(data=betas_combined[:, 1:], columns=cols)
        qval_data = pd.DataFrame(data=qvalues[:, 1:], columns=cols)

        pval_data["Genes"] = genes
        coef_data["Genes"] = genes
        qval_data["Genes"] = genes

        return coef_data, pval_data, qval_data
