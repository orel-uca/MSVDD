__author__ = 'vblancoOR'
__docs__ = 'MSVDD Instance and Solution Classes'

class Instance(object):
    """A class to store the data and parameters for Gurobi model instances.
        - data: Original data points (size: total_samples, d).
        - ntrain: Number of training samples.
        - nval: Number of validation samples.
        - ntest: Number of test samples.
        - y: Labels for the data (regular/outliers).
        - p: Number of clusters.
        - C: Regularization parameter.
        - kernel: Kernel type used for calculations.
        - sigma: Parameter for the RBF kernel (if applicable).
        - degree: Degree of the polynomial kernel (if applicable).
    """
    def __init__(self, data, ntrain, nval, ntest, y, p=1, C=0.1, kernel='linear', sigma=None, degree=None):
        
        self.data=data
        self.ntrain=ntrain
        self.nval=nval
        self.ntest=ntest
        self.y=y
        self.p=p
        self.C=C
        self.kernel=kernel
        self.sigma=sigma
        self.degree=degree


class Solution(object):
    """A class to store the Gurobi solution model and tupledicts of variables
        - x: Training data points used (size: n, d).
        - data: Original data points (size: total_samples, d).
        - ntrain: Number of training samples.
        - nval: Number of validation samples.
        - ntest: Number of test samples.
        - y: Labels for the data.
        - p: Number of clusters.
        - C: Regularization parameter.
        - kernel: Kernel type used for calculations.
        - sigma: Parameter for the RBF kernel (if applicable).
        - degree: Degree of the polynomial kernel (if applicable).
        - obj: Objective value of the solution.
        - runtime: Time taken to compute the solution.
        - nodes: Number of nodes explored (if applicable).
        - gap: Optimality gap (if applicable).
        - c: Centers of the clusters (size: p, d).
        - R: Square radii of the clusters (size: p,).
        - xi: Slack variables (size: n, p).
        - z: Allocation variables (size: n, p).
        - alpha: Coefficients for each cluster (size: n, p).
    """
    def __init__(self, x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, obj, runtime, nodes, gap=None, c=None, R=None, xi=None, z=None, alpha=None):
        
        self.x=x
        self.data=data
        self.ntrain=ntrain
        self.nval=nval
        self.ntest=ntest
        self.y=y
        self.p=p
        self.C=C
        self.kernel=kernel
        self.sigma=sigma
        self.degree=degree
        self.obj=obj
        self.runtime=runtime
        self.nodes=nodes
        self.gap=gap
        self.c=c
        self.R=R
        self.xi=xi
        self.z=z
        self.alpha=alpha        
