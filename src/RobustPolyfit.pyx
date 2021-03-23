import numpy as np
from scipy.linalg import solve_triangular
cimport numpy as np

ctypedef np.float64_t FLOAT64

cdef class robust_polyfit():
    cdef public float var
    cdef public bint converged
    cdef public float tol
    cdef public np.ndarray weights
    cdef public int polyorder
    cdef public int df
    cdef public int max_iter
    cdef private bint user_spec_start_weights

    def __init__(self, int max_iter=500, float tol=1e-2, 
            int polyorder = 1, int df = 1, starting_weights=None):
        self.check_user_specs(max_iter, tol, polyorder, df)
        self.weights = np.empty((1,polyorder+1), dtype=np.float64)
        self.var = 1.0
        self.polyorder = polyorder
        self.df = df
        self.max_iter = max_iter
        self.converged = False
        self.tol = tol
        if starting_weights is not None:
            if isinstance(starting_weights, np.ndarray) == False:
                raise ValueError("Starting weights, if supplied, must be a numpy array.")
            if starting_weights.shape[0] != 1 or starting_weights.shape[1] != polyorder + 1:
                raise ValueError("Starting weights, if supplied, must be of shape "
                        "(1,polyorder+1)")
            if starting_weights.dtype != "float64":
                raise ValueError("Starting weights, if supplied, must be of dtype float64")
            self.user_spec_start_weights = True
            self.weights = starting_weights
        else:
            self.user_spec_start_weights = False
        
    #Check the conditions the user selected. It's sort of arbitrary that we only
    #fit linear, quadratic or cubic right now but...high degree polynomials are
    #inherently ill-conditioned and we want to avoid promising the user a good
    #fit we may not be able to deliver; 3 seemed like a good cutoff.
    cdef check_user_specs(self, max_iter, tol, polyorder, df):
        if max_iter < 1:
            raise ValueError("The number of iterations must be positive and > 1.")
        if tol < 0:
            raise ValueError("The tolerance must be > 0.")
        if df < 1:
            raise ValueError("A Student's T distribution has df >= 1.")
        if polyorder not in [1,2,3]:
            raise ValueError("Currently this class only fits linear, quadratic or cubic"
                    "functions, i.e. only polyorder 1, 2 and 3 are currently accepted.")

    cdef check_fitting_input(self, x, y, yvar):
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
    cdef get_vandermonde_mat(self, np.ndarray[FLOAT64, ndim=1] x, 
            bint get_norms = False):
        cdef np.ndarray[FLOAT64, ndim=2] vmat = np.empty((x.shape[0], self.polyorder + 1),
                            dtype=np.float64)
        cdef np.ndarray[FLOAT64, ndim=1] norms = np.empty((self.polyorder+1),
                dtype=np.float64)
        vmat[:,0] = 1.0
        vmat[:,1] = x
        cdef int i = 0
        for i in range(1, self.polyorder):
            vmat[:,i+1] = x * vmat[:,i]
        if get_norms == False:
            return vmat
        norms = np.sqrt(np.square(vmat).sum(0))
        norms[norms==0] = 1.0
        return vmat / norms[np.newaxis,:], norms


    def fit(self, np.ndarray[FLOAT64, ndim=1] x, 
            np.ndarray[FLOAT64, ndim=1] y):
        self.check_fitting_input(x,y, np.var(y))
        X_, norms = self.get_vandermonde_mat(x, get_norms=True)
        if self.user_spec_start_weights == False:
            self.weights = self.get_starting_params(x, y)

        cdef np.ndarray lower_bound = np.full((1), fill_value=-np.inf, dtype=np.float64)
        cdef np.ndarray change = np.empty((1), dtype=np.float64)
        cdef np.ndarray current_bound = np.empty((1), dtype=np.float64)
        cdef np.ndarray resp = np.empty((y.shape[0]), dtype=np.float64)
        cdef double weight_change = 0
        cdef np.ndarray old_weights = np.full((1,self.polyorder+1),
                fill_value = -np.inf, dtype=np.float64)
        cdef np.ndarray preds = np.empty((y.shape[0]), dtype=np.float64)
        
        preds = self.internal_predict(X_)
        #Crude estimate of the scale which is refined during fitting.
        self.var = np.var(y)
        
        cdef int i = 0
        #Ordinarily in EM we iterate until the lower bound converges.
        #In THIS case, however, we can occasionally have slow convergence
        #because the variance is being adjusted. To avoid this, we define 
        #convergence using both lower bound AND weights. If the weights 
        #are not changing and only lower bound is, we have converged.
        for i in range(self.max_iter):
            resp, current_bound[0] = self.e_step(y, preds)
            preds = self.m_step(X_, y, resp)
            change[0] = current_bound[0] - lower_bound[0]
            weight_change = np.max(np.abs(old_weights / self.weights) - 1 )
            if np.abs(change[0]) < self.tol or weight_change < 1e-3:
                self.converged = True
                self.weights = self.weights / norms
                break
            lower_bound = current_bound
            old_weights = self.weights
        if self.converged == False:
            raise ValueError("Fitting did not converge! Try increasing tol or "
                    "increasing max_iter.")


    cdef weighted_linreg(self, np.ndarray[FLOAT64, ndim=2] X, 
                np.ndarray[FLOAT64, ndim=1] y, 
                np.ndarray[FLOAT64, ndim=1] resp):
        cdef np.ndarray[FLOAT64, ndim=1] resp_sqrt = np.empty((resp.shape[0]),
                    dtype = np.float64)
        resp_sqrt = np.sqrt(resp)
        q, r = np.linalg.qr(resp_sqrt[:,np.newaxis] * X)
        target = np.matmul(q.T, (resp_sqrt * y)[:,np.newaxis] )
        self.weights = solve_triangular(r, target).T


    cdef internal_predict(self, np.ndarray[FLOAT64, ndim=2] X):
        return np.sum(X * self.weights, axis=1)


    cdef e_step(self, np.ndarray[FLOAT64, ndim=1] y, 
                    np.ndarray[FLOAT64, ndim=1] preds):
        cdef np.ndarray[FLOAT64, ndim=1] maha_dist = np.empty((y.shape[0]),
                    dtype=np.float64)
        cdef np.ndarray[FLOAT64, ndim=1] resp = np.empty((y.shape[0]), 
                    dtype=np.float64)
        cdef float current_bound
        maha_dist = self.get_maha_dist(y, preds)
        resp = (self.df + 1) / (self.df + maha_dist)
        current_bound = -np.log(self.var) - 0.5 * np.sum(resp * maha_dist)
        return resp, current_bound

    cdef m_step(self, np.ndarray[FLOAT64, ndim=2] X, 
            np.ndarray[FLOAT64, ndim=1] y, 
            np.ndarray[FLOAT64, ndim=1] resp):
        cdef np.ndarray[FLOAT64, ndim=1] preds = np.empty((y.shape[0]),
                        dtype=np.float64)
        cdef float n = X.shape[0]
        
        self.weighted_linreg(X, y, resp)
        preds = self.internal_predict(X)
        self.var = (1 / n) * np.sum(resp * (y - preds)**2)
        return preds

    cdef get_maha_dist(self, np.ndarray[FLOAT64, ndim=1] y, 
            np.ndarray[FLOAT64, ndim=1] preds):
        return (y-preds)**2 / self.var
    
    
    cdef get_starting_params(self, np.ndarray[FLOAT64, ndim=1] x, 
            np.ndarray[FLOAT64, ndim=1] y):
        return np.polyfit(x, y, deg=self.polyorder)[np.newaxis,:]
    
    def get_coefs(self):
        return self.weights.flatten()

    def predict(self, x):
        if self.converged == False:
            raise ValueError("Model not fitted yet!")
        X_ = self.get_vandermonde_mat(x)
        return np.sum(X_ * self.weights, axis=1)



