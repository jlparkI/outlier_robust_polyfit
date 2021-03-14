import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

class robust_polyfit():


    def __init__(self, max_iter=500, tol=1e-3, polyorder = 1, df = 4):
        self.check_user_specs(max_iter, tol, polyorder, df)
        self.weights, self.var = None, 1.0
        self.polyorder = polyorder
        self.df = 4
        self.max_iter = max_iter
        self.converged = False
        self.tol = tol
        
    #Check the conditions the user selected. It's sort of arbitrary that we only
    #fit linear, quadratic or cubic right now but...high degree polynomials are
    #inherently ill-conditioned and we want to avoid promising the user a good
    #fit we may not be able to deliver; 3 seemed like a good cutoff.
    def check_user_specs(self, max_iter, tol, polyorder, df):
        if max_iter < 1:
            raise ValueError("The number of iterations must be positive and > 1.")
        if tol < 0:
            raise ValueError("The tolerance must be > 0.")
        if df < 1:
            raise ValueError("A Student's T distribution has df >= 1.")
        if polyorder not in [1,2,3]:
            raise ValueError("Currently this class only fits linear, quadratic or cubic"
                    "functions, i.e. only polyorder 1, 2 and 3 are currently accepted.")

    def check_fitting_input(self, x, y, yvar):
        if isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False:
            raise ValueError("The input x and y data should be numpy arrays.")
        if x.dtype != "float64" or y.dtype != "float64":
            raise ValueError("The input x and y arrays should be of type float64.")
        if yvar < 1e-9:
            raise ValueError("y has very low variance and may have only one or two "
                    "unique values. Please check your input.")
        if len(x.shape) > 1:
            raise ValueError("The x-array should be flat (one dimension only). Please "
                    "try again.")
        #This is a little bit arbitrary -- just primarily want to catch situations
        #where the user accidentally does something silly like trying to fit a 
        #quadratic with two
        #datapoints etc. It's hard to imagine there are very many situations where
        #a user would want to fit this to a very small number of datapoints --
        #at that point, why are you doing robust regression -- you don't know that
        #your "outliers" really are outliers!
        if x.shape[0] < 10:
            raise ValueError("You are trying to fit using a very small number of "
                    "datapoints! This class requires that the number of datapoints be "
                    ">= 10. In general, if you are using robust regression "
                    "it is because you think that some of your datapoints are outliers, "
                    "and it's unwise to try to make that determination from a very small "
                    "dataset.")

    #Builds the vandermonde matrix, and sets columns to be unit norm if caller so
    #specifies (true during training, false during prediction). 
    def get_vandermonde_mat(self, x, get_norms = False):
        vmat = np.empty((x.shape[0], self.polyorder + 1))
        vmat[:,0] = 1.0
        vmat[:,1] = x
        for i in range(1, self.polyorder):
            vmat[:,i+1] = x * vmat[:,i]
        if get_norms == False:
            return vmat
        norms = np.sqrt(np.square(vmat).sum(0))
        norms[norms == 0] = 1.0
        return vmat / norms[np.newaxis,:], norms


    def fit(self, x, y, special_linfit = False):
        self.check_fitting_input(x,y, np.var(y))
        X_, norms = self.get_vandermonde_mat(x, get_norms=True)
        self.weights = self.get_starting_params(x, y)
        
        #Ordinarily in EM we iterate until the lower bound converges.
        #In THIS case, however, we can end up with very slow convergence
        #because the variance is being adjusted. To avoid this, we define 
        #convergence using both lower bound AND weights. If the weights 
        #are not changing and only lower bound is, we have converged.
        lower_bound = -np.inf
        old_weights = np.full(self.polyorder+1, fill_value=-np.inf)
        preds = self.internal_predict(X_)
        #Crude estimate of the scale which is refined during fitting.
        self.var = np.sum((preds - y)**2) / (y.shape[0])
        for i in range(self.max_iter):
            resp, current_bound = self.e_step(y, preds)
            preds = self.m_step(X_, y, resp)
            change = current_bound - lower_bound
            weight_change = np.max(np.abs((old_weights - self.weights) 
                                / self.weights))
            if np.abs(change) < self.tol or weight_change < 1e-2:
                self.converged = True
                self.weights = self.weights / norms
                break
            lower_bound = current_bound
            old_weights = self.weights
        if self.converged == False:
            raise ValueError("Fitting did not converge! Try increasing tol or "
                    "increasing max_iter.")


    def weighted_linreg(self, X, y, resp = None):
        resp_sqrt = np.sqrt(resp)
        q, r = np.linalg.qr(resp_sqrt[:,np.newaxis] * X)
        target = np.matmul(q.T, (resp_sqrt * y)[:,np.newaxis] )
        self.weights = solve_triangular(r, target).T


    def internal_predict(self, X):
        return np.sum(X * self.weights, axis=1)


    def e_step(self, y, preds):
        maha_dist = self.get_maha_dist(y, preds)
        resp = (self.df + 1) / (self.df + maha_dist)
        current_bound = -np.log(self.var) - 0.5 * np.sum(resp * maha_dist)
        return resp, current_bound

    def m_step(self, X, y, resp):
        self.weighted_linreg(X, y, resp)
        preds = self.internal_predict(X)
        self.var = (1 / X.shape[0]) * np.sum(resp * (y - preds)**2)
        return preds

    def get_maha_dist(self, y, preds):
        return (y-preds)**2 / self.var
    
    
    def get_starting_params(self, x, y):
        return np.polyfit(x, y, deg=self.polyorder)[np.newaxis,:]
    
    def get_coefs(self):
        return self.weights.flatten()

    def predict(self, x):
        if self.converged == False:
            raise ValueError("Model not fitted yet!")
        X_ = self.get_vandermonde_mat(x)
        return np.sum(X_ * self.weights, axis=1)

