# outlier_robust_polyfit

- [Summary](##Summary)
- [Installation](##Installation)
- [Usage](##Usage)

## Summary

This package fits linear, quadratic or cubic polynomials with Student's 
T-distributed errors using the EM algorithm. It's 2x as fast as
Scipy's siegelslopes for fitting robust linear models to small datasets
(10 - 20 datapoints) and an order of magnitude faster for larger 
datasets (>=100 datapoints). It yields a robust fit for datasets
where 25% of datapoints are outliers (use with caution for datasets
where the "outlier" population may be greater than 25%).

![example](https://github.com/jlparki/outlier_robust_polyfit/blob/main/resources/example_1.png)

![example](https://github.com/jlparki/outlier_robust_polyfit/blob/main/resources/example_2.png)



## Installation



## Usage


