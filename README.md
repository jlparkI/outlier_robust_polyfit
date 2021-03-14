# outlier_robust_polyfit
<br>
- [Summary](#summary)
- [Installation](#installation)
- [Usage](#usage)
<br>
## Summary
<br>
This package fits linear, quadratic or cubic polynomials with Student's 
T-distributed errors using the EM algorithm. It's 2x as fast as
Scipy's siegelslopes for fitting robust linear models to small datasets
(10 - 20 datapoints) and an order of magnitude faster for larger 
datasets (>=100 datapoints). It yields a robust fit for datasets
where 25% of datapoints are outliers (use with caution for datasets
where the "outlier" population may be greater than 25%).
<br>
![example](https://github.com/jlparki/outlier_robust_polyfit/blob/main/resources/example_1.png)
<br>
![example](https://github.com/jlparki/outlier_robust_polyfit/blob/main/resources/example_2.png)
<br>
## Installation
<br>
    pip install robustpolyfit
<br>
## Usage
<br>
    from RobustPolyfit import robust_polyfit
    
    class robust_polyfit(int max_iter=500, float tol=1e-2,
        int polyorder = 1, int df = 1)
<br>
- **max_iter**    The maximum number of iterations
- **tol**         Tolerance for convergence (change in lower bound).
    - Setting this to a higher value reduces the number of iterations, leading
to faster convergence (and in some cases a poorer fit).
- **polyorder**   The degree of the polynomial. Allowed values are 1, 2 and 3.
- **df**          The degrees of freedom of the Student's t-distribution.
    - A smaller value leads to a higher level of tolerance for outliers. df=1
is the smallest allowed value. Large values for df, e.g. >> 5, are essentially
equivalent to normally distributed error and do not offer any benefit compared
to standard least squares fitting.
<br>
    
