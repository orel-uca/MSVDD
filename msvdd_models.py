__authors__ = ['vblancoOR', 'raulpaez']
__docs__ = 'MSVDD Models: Primal and Dualized formulations using Gurobipy'

import numpy as np
from gurobipy import Model, GRB, quicksum
from msvdd_class import Solution
from scipy.spatial.distance import cdist
from utilities import Kernel

def Primal_MSVDD(instance):

    data = instance.data
    y = instance.y
    p = instance.p
    C = instance.C
    kernel  = instance.kernel
    sigma = instance.sigma
    degree = instance.degree
    
    ntrain = int(instance.ntrain)
    nval = int(instance.nval)
    ntest = int(instance.ntest)
    
    train = np.array(range(ntrain), dtype='i')
    x = data[train, :]

    P = range(p)
    
    a = cdist(x,x)
    bigM = float(np.max(a)**2)
    
    n = x.shape[0]
    d = x.shape[1]

    P = range(p)
    N = range(n)
    D = range(d)

    model = Model()
    
    ## Vars
    c = model.addVars(p, d, lb=-GRB.INFINITY, name="c")# centers
    R  = model.addVars(p, ub=float(bigM/4), name="R")# radii
    xi = model.addVars(n, ub=float(bigM/4), name="xi")# error
    z = model.addVars(n, p, vtype=GRB.BINARY, name="z")# assginment
    
    ## Auxiliary vars to linearize distance
    v = {}
    for i in N:
        for j in P:
            for k in D:
                v[i,j,k] = x[i,k]-c[j,k]
    
    ## Obj
    obj = quicksum(R[j] for j in P) + C*quicksum(xi[i] for i in N)
    model.setObjective(obj, GRB.MINIMIZE)
    
    ## Ctrs
    for i in N:
        for j in P:
            model.addConstr(quicksum(v[i,j,k]*v[i,j,k] for k in D) <= R[j] + xi[i] + bigM*(1.0-z[i,j]), name="R3")

    for i in N:
        model.addConstr(quicksum(z[i,j] for j in P )== 1, name="R4")

    model.addConstrs((R[j] <= R[j+1] for j in range(p-1)), name="Sim")
    
    model.update()
    
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = 3600
    model.setParam("ScaleFlag", 2)

    model.optimize()

    if model.Status == GRB.Status.OPTIMAL:
        
        zsol=np.empty((n,p))
        for j in P:
            for i in N:
                zsol[i,j]=z[i,j].X

        csol=np.array([[c[j,l].X for l in D] for j in P])

        Rsol=np.array([R[j].x for j in P])
        xisol = [xi[i].x for i in N]
        
        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, model.ObjVal, model.Runtime, model.NodeCount, model.MIPGap, csol, Rsol, xisol, zsol)
    
    elif (model.Status == GRB.INFEASIBLE):
        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
    
    else:
        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, None, None, None, model.Fingerprint)

    return sol

def Dualized_MSVDD(instance):
    
    data = instance.data
    y = instance.y
    p = instance.p
    C = instance.C
    kernel = instance.kernel
    sigma = instance.sigma
    degree = instance.degree 
    
    ntrain = int(instance.ntrain)
    nval = int(instance.nval)
    ntest = int(instance.ntest)
    
    train = np.array(range(ntrain), dtype='i')
    
    x = data[train, :]
    
    a = cdist(x,x)
    
    bigM = 1+a.sum()*C**2
    
    n = x.shape[0]
    
    P = range(p)
    N = range(n)

    model = Model()
        
    ## Vars
    R  = model.addVars(p, name="R")
    xi = model.addVars(n, name="xi")
    z = model.addVars(n,p, vtype=GRB.BINARY, name="z")
    alpha = model.addVars(n,p, ub=C, name="alpha")

    K = Kernel(x, x, kernel=kernel, sigma=sigma, degree=degree)
    
    ## Obj
    obj = quicksum(R[j] for j in P) + C * quicksum(xi[i] for i in N)
    model.setObjective(obj, GRB.MINIMIZE)
    
    ## Ctrs
    for i in N:
        model.addConstr(quicksum(z[i,j] for j in P) == 1, name="R8")
        
    for i in N:
        for j in P:
            model.addConstr(z[i,j]*K[i,i] - 2*quicksum(alpha[k,j] * K[i,k] for k in N) + quicksum(alpha[k,j] * alpha[l,j] * K[k,l] for k in N for l in N) <= R[j] + xi[i] + bigM*(1-z[i,j]), name="R12")
            model.addConstr(alpha[i,j] <= C*z[i,j], name="R13")
            
    for j in P:
        model.addConstr(quicksum(alpha[i,j] for i in N) <= 1+1e-6, name="R14-1")
        model.addConstr(quicksum(alpha[i,j] for i in N) >= 1-1e-6, name="R14-2")
    
    model.addConstrs((R[j] <= R[j+1] for j in range(p-1)), name="Sim")
    
    model.update()
    
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = 3600
    model.setParam("ScaleFlag", 2)

    model.optimize()

    if model.Status == GRB.Status.OPTIMAL:

        alphasol=np.empty((n,p))
        for j in P:
            for i in N:
                alphasol[i,j]=alpha[i,j].X
                
        zsol=np.empty((n,p))
        for j in P:
            for i in N:
                zsol[i,j]=z[i,j].X

        xisol = [xi[i].x for i in N]
        Rsol=np.array([R[j].x for j in P])

        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, model.ObjVal, model.Runtime, model.NodeCount, model.MIPGap, None, Rsol, xisol, zsol, alphasol)
        
    elif (model.Status == GRB.INFEASIBLE):
        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, None, model.Runtime, model.NodeCount, model.Fingerprint)
    
    else:
        sol = Solution(x, data, ntrain, nval, ntest, y, p, C, kernel, sigma, degree, None, None, None, model.Fingerprint)

    return sol
