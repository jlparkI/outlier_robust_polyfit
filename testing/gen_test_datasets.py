import numpy as np, argparse, sys
sys.path.append("..")
from src.RobustPolyfit.RobustPolyfit import robust_polyfit
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

def main():
    parser = argparse.ArgumentParser(description="Test the RobustPolyfit module.")
    parser.add_argument("noisefrac", type=float, default=0.1, help="The fraction of the generated "
            "datapoints that are outliers.")
    parser.add_argument("numpoints", type=int, default=100, 
            help="The total number of datapoints.")
    parser.add_argument("sigma", type=float, default=3, help="Scale value for the Cauchy distribution "
            "used to generate outliers.")
    parser.add_argument("--linear", type=int, help="Run specified number of tests using "
            "polyorder 1.")
    parser.add_argument("--quad", type=int, help="Run specified number of tests using "
            "polyorder 2.")
    parser.add_argument("--cubic", type=int, help="Run specified number of tests using "
            "polyorder 3.")
    parser.add_argument("--siegelslopes", action="store_true", help="Compare on time with "
            "the scipy siegelslopes function (polyorder 1 only).")
    args = parser.parse_args()
    if args.linear is not None:
        set_up_test(args.noisefrac, args.numpoints, args.sigma, args.linear,
                args.siegelslopes, 1)
    elif args.quad is not None:
        set_up_test(args.noisefrac, args.numpoints, args.sigma, args.quad,
                args.siegelslopes, 2)
    elif args.cubic is not None:
        set_up_test(args.noisefrac, args.numpoints, args.sigma, args.cubic,
                args.siegelslopes, 3)

def set_up_test(noisefrac, numpoints, sigma, num_iter, use_siegel, polyorder):
    x, y = [], []
    for i in range(num_iter):
        xnew = np.linspace(-10,10,numpoints)
        
        #Add Gaussian noise to the randomly generated polynomial.
        if polyorder == 1:
            slope, intcpt = np.random.uniform(-10,10), np.random.uniform(-10,10)
            ynew = slope*xnew + intcpt + norm.rvs(0,1, size=numpoints)
        elif polyorder == 2:
            slope1, slope2, intcpt = np.random.uniform(-5,5),\
                    np.random.uniform(-10, 10), np.random.uniform(-10,10)
            ynew = slope1*xnew**2 + slope2*xnew + intcpt + norm.rvs(0,1, size=numpoints)
        elif polyorder == 3:
            slope1, slope2, slope3, intcpt = np.random.uniform(-2,2),\
                np.random.uniform(-5, 5), np.random.uniform(-10,10), np.random.uniform(-10,10)
            ynew = slope1*xnew**3 + slope2*xnew**2 + slope3*xnew + intcpt + norm.rvs(0,1, size=numpoints)
        
        idx = np.random.choice(numpoints, size=int(noisefrac*numpoints), replace=False)
        #Add outliers but in one direction only to create a skewed distribution.
        ynew[idx] = ynew[idx] * (1 + np.random.uniform(0.3,4,size=idx.shape[0]))
        x.append(xnew)
        y.append(ynew)
    if polyorder > 1:
        modelfit(x, y, use_siegel=False, polyorder=polyorder, noisefrac=noisefrac)
    else:
        modelfit(x, y, use_siegel, polyorder=polyorder, noisefrac=noisefrac)



def modelfit(x, y, use_siegel, polyorder, noisefrac):
    robmods = []
    start_time = time.time()
    for i in range(len(x)):
        robmod = robust_polyfit(polyorder=polyorder)
        robmod.fit(x[i], y[i])
        robmods.append(robmod)
    end_time = time.time()
    average_time = (end_time - start_time) / len(x)

    if use_siegel:
        from scipy.stats import siegelslopes
        sslopes, ssints = [], []
        start_time = time.time()
        for i in range(len(x)):
            sslope, sint = siegelslopes(y[i], x[i])
            sslopes.append(sslope)
            ssints.append(sint)
        end_time = time.time()
        average_siegel_time = (end_time - start_time) / len(x)
        print("Average siegelslopes time: %s"%average_siegel_time)
    print("Average fit time: %s"%average_time)
    for i in range(len(x)):
        fig = plt.figure()
        plt.scatter(x[i], y[i], s=10, label="raw data")
        plt.plot(x[i], robmods[i].predict(x[i]), color="black", linestyle="dashed",
                label="Student T reg fit")
        coefs = np.polyfit(x[i], y[i], deg=polyorder)
        if polyorder == 1:
            plt.plot(x[i], coefs[0]*x[i] + coefs[1], color="red", linestyle="dashed",
                    label="Standard least squares fit")
        if polyorder == 2:
            plt.plot(x[i], coefs[0]*x[i]**2 + coefs[1]*x[i] + coefs[2], 
                    color="red", linestyle="dashed",
                    label="Standard least squares fit")
        if polyorder == 3:
            plt.plot(x[i], coefs[0]*x[i]**3 + coefs[1]*x[i]**2 + coefs[2]*x[i] + coefs[3], 
                    color="red", linestyle="dashed",
                    label="Standard least squares fit")

        if use_siegel:
            plt.plot(x[i], sslopes[i]*x[i] + ssints[i], color="green", linestyle="dashed",
                    label="siegelslopes fit")
        plt.title("Robust regression fit, polyorder %s,\noutlier fraction %s"
                %(polyorder, noisefrac))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig("Polyorder_%s_%s_noisefrac_%s.png"%(polyorder, i, noisefrac))
        plt.close()



if __name__ == "__main__":
    main()
