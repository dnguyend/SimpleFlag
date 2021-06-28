import numpy as np
from numpy import trace
from SimpleFlag.manifolds.RealFlagOpt import RealFlagOpt


def l1(A):
    return np.max(np.abs(A))


if False:
    dvec = np.array([4, 3, 2])
    alp = np.array([1, .8])
    man = RealFlagOpt(dvec, alp)

    Y = man.rand()
    omg = man._rand_ambient()
    xi = man.randvec(Y)
    print(man.J(Y, xi))
    print(man.J(Y, man.proj(Y, omg)))

    print(l1(omg - man.g(Y, man.g_inv(Y, omg))))
    
    s1 = man.proj_g_inv(Y, omg)
    s2 = man.proj(Y, man.g_inv(Y, omg))
    print(l1(s1 - s2))
    print((man.inner(Y, omg, xi) - man.inner(Y, man.proj(Y, omg), xi)))

    eta = man.randvec(Y)    
    print(l1(man.proj(Y, man.GammaH(Y, xi, eta) -
                      man.GammaH(Y, eta, xi))))

    n = dvec.sum()
    p = dvec[1:].sum()

    egrad = np.random.randn(n, p)
    ehess = np.random.randn(n, p)

    from ManNullRange.manifolds.RealFlag import RealFlag
    pp = dvec.shape[0] - 1    
    alpha = np.random.randint(1, 10, (pp, pp+1)) * .1
    alpha[:, 0] = alp[0]
    alpha[:, 1:] = alp[1]

    man1 = RealFlag(dvec, alpha)

    print(man.rhess02(Y, xi, eta, egrad, ehess) -
          man1.rhess02(Y, xi, eta, egrad, ehess))
    
    s1 = man1.ehess2rhess(Y, egrad, ehess, xi)
    s2 = man.ehess2rhess(Y, egrad, ehess, xi)
    
    print(man.rhess02(Y, xi, eta, egrad, ehess))
    print(man1.rhess02(Y, xi, eta, egrad, ehess))
    
    print(man.inner(Y, s1, eta))

    def r02(man, Y, xi, eta, egrad, ehess):
        alf = man.alpha[1] / man.alpha[0]
        
        def Pi0(a):
            return a - Y@(Y.T@a)
        rr1 = Y@man.flag_sym(xi.T@eta) +\
            (1-alf)*Pi0(xi@(eta.T@Y) + eta@(xi.T@Y))
        print(np.trace(ehess@eta.T) - np.trace(
            egrad.T@Y@man.flag_sym(xi.T@eta) +\
            (1-alf)*egrad.T@Pi0(xi@(eta.T@Y) + eta@(xi.T@Y))))
        print(np.trace(ehess@eta.T) - np.trace(
            man.flag_sym(Y.T@egrad).T@xi.T@eta) -
              (1-alf)*np.trace(Pi0(egrad).T@xi@(eta.T@Y) + Pi0(egrad).T@eta@(xi.T@Y)))

        print(np.trace(ehess@eta.T) - np.trace(
            eta.T@xi@man.flag_sym(Y.T@egrad)) -
              (1-alf)*np.trace(Y@Pi0(egrad).T@xi@eta.T +
                               Pi0(egrad)@Y.T@xi@eta.T))
        ss = ehess - xi@man.flag_sym(Y.T@egrad) -\
            (1-alf)*(Y@egrad.T@Pi0(xi) + Pi0(egrad)@Y.T@xi)
        # print(man.proj_g_inv(Y, ss))
        print(np.sum(ss*eta))

    r02(man, Y, xi, eta, egrad, ehess)
    
    print(man.inner(Y, s2, eta))
    print(l1(s1 - s2))
    
    print(np.trace(egrad.T@Y@man.flag_sym(xi.T@eta)))
    print(np.trace((Y.T@egrad).T@man.flag_sym(xi.T@eta)))
    print(np.trace((Y.T@egrad).T@man.flag_sym(xi.T@eta)))        
    print(np.trace((xi.T@eta).T@man.flag_sym(Y.T@egrad)))
    print(np.trace((eta.T@xi)@man.flag_sym(Y.T@egrad)))        
    

def num_deriv(man, W, xi, func, dlt=1e-7):
    Wnew = W + dlt*xi
    return (func(Wnew) - func(W))/dlt

    
def optim_test():
    from ManNullRange.tests.test_tools import random_orthogonal, make_sym_pos
    from ManNullRange.manifolds.RealFlag import RealFlag
    
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable

    n = 1000

    # problem Tr(sum AXBX^T)
    for i in range(1):
        dvec = np.array([0, 30, 2, 1])
        dvec[0] = n - dvec[1:].sum()
        # dvec = np.array([0, 3, 2, 2])
        d = dvec[1:].sum()
        dvec[0] = n - d
        D = np.random.randint(1, 10, n) * 0.02 + 1
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T

        alp = np.array([1, np.random.randint(1, 10)*.1])
        mm = RealFlagOpt(dvec, alp=alp)
        pp = dvec.shape[0] - 1    
        alpha = np.random.randint(1, 10, (pp, pp+1)) * .1
        alpha[:, 0] = alp[0]
        alpha[:, 1:] = alp[1]
        man = RealFlag(dvec, alpha)
        
        Lbd = mm.make_lbd()
        Lbd += np.abs(Lbd[0]) + 1
        
        @Callable
        def cost(X):
            return trace(A @ (X * Lbd[None, :]) @ X.T)
        
        @Callable
        def egrad(X):
            return 2*A @ (X * Lbd[None, :])

        @Callable
        def ehess(X, H):
            return 2*A @ (H * Lbd[None, :])

        if False:
            X = man.rand()
            xi = man.randvec(X)
            d1 = num_deriv(man, X, xi, cost)
            d2 = trace(egrad(X) @ xi.T)
            print(l1(d1-d2))

            # test retraction.
            dlt = 1e-7
            print((man.exp(X, dlt*xi) - X)/dlt-xi)
            print(ehess(X, xi) - (egrad(X+dlt*xi) - egrad(X))/dlt)
                    
        prob = Problem(
            man, cost, egrad=egrad)
        XInit = man.rand()

        prob = Problem(
            man, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)

        prob1 = Problem(
            mm, cost, egrad=egrad, ehess=ehess)

        opt1 = solver.solve(prob1, x=XInit, Delta_bar=250)
        
        print(cost(opt))

        if False:
            min_val = 1e190
            # min_X = None
            for i in range(100):
                Xi = man.rand()
                c = cost(Xi)
                if c < min_val:
                    # min_X = Xi
                    min_val = c
                if i % 1000 == 0:
                    print('i=%d min=%f' % (i, min_val))
            print(min_val)
        alpha_c = alpha.copy()
        alpha_c[:] = 1
        man1 = RealFlag(dvec, alpha=alpha_c)
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)
        alpha_c5 = alpha_c.copy()
        alpha_c5[:, 1:] = .5
        man1 = RealFlag(dvec, alpha=alpha_c5)
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)


def optim_q_test():
    from ManNullRange.tests.test_tools import random_orthogonal
    
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable

    n = 1000
    np.random.seed(0)
    
    for i in range(1):
        dvec = np.array([0, 30, 20, 10])
        # dvec = np.array([0, 3, 2, 2])
        dvec[0] = n - dvec[1:].sum()
        
        d = dvec[1:].sum()
        dvec[0] = n - d
        D = np.random.randint(1, 100, n) * 0.2
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T

        alp = np.array([1, np.random.randint(1, 20)*.1])
        man = RealFlagOpt(dvec, alp=alp)
        pp = dvec.shape[0] - 1
        
        Lbd = np.concatenate(
            [dvec[a]*[a] for a in np.arange(1, pp+1)])        
        
        @Callable
        def cost(X):
            U = ((X.T@A)@X)*Lbd[None, :]
            return trace(U@U)
        
        @Callable
        def egrad(X):
            AXL = (A@X)*Lbd[None, :]
            return 4*AXL@(X.T@AXL)

        @Callable
        def ehess(X, H):
            AHL = (A@H)*Lbd[None, :]
            AXL = (A@X)*Lbd[None, :]
            return 4*AHL@(X.T@AXL) + 4*AXL@(H.T@AXL) + 4*AXL@(X.T@AHL)
                
        if False:
            X = man.rand()
            xi = man.randvec(X)
            d1 = num_deriv(man, X, xi, cost)
            d2 = trace(egrad(X) @ xi.T)
            print(l1(d1-d2))

            # test retraction.
            dlt = 1e-7
            print((man.exp(X, dlt*xi) - X)/dlt-xi)
            print(l1(ehess(X, xi) - (egrad(X+dlt*xi) - egrad(X))/dlt))
                    
        prob = Problem(
            man, cost, egrad=egrad)
        XInit = man.rand()

        print('doing alp %s' % alp)
        prob = Problem(
            man, cost, egrad=egrad, ehess=ehess, verbosity=1)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)

        from pymanopt.solvers import SteepestDescent
        prob = Problem(man, cost=cost, egrad=egrad, verbosity=1)
        solvS = SteepestDescent(
            maxtime=10000, maxiter=1000000, mingradnorm=1e-8)

        print('Doing SteepestDescent')
        opt2 = solvS.solve(prob, x=XInit)

        for alx in range(1, 60):
            alpha_c = alp.copy()
            alpha_c[1] = .1 + alx*.02
            man1 = RealFlagOpt(dvec, alp=alpha_c)
            prob = Problem(man1, cost, egrad=egrad, ehess=ehess, verbosity=1)

            solver = TrustRegions(maxtime=100000, maxiter=100)
            opt = solver.solve(prob, x=XInit, Delta_bar=250)
            print('alp =%s cost =%f' % (alpha_c, cost(opt)))                        
            
        alpha_c5 = alpha_c.copy()
        alpha_c5[1] = .5
        man1 = RealFlagOpt(dvec, alp=alpha_c5)
        prob = Problem(man1, cost, egrad=egrad, ehess=ehess, verbosity=1)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        print('doing alp %s' % alpha_c5)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)
        

def parse():
    with open('test_res.txt') as fi:
        lns = fi.readlines()
    ret = {}
    cnt = 0

    def parse_line(alist):
        save_item = []
        for a in alist:
            try:
                save_item.append(float(a))
            except Exception as e:
                pass
        return save_item
    
    for i in range(len(lns)):
        if lns[i].startswith('Terminated'):
            save_item = parse_line(lns[i].split())
        elif lns[i].startswith('alp'):
            ret[cnt] = {'al': float(lns[i].split()[2][:-1]),
                        'items': save_item}
            cnt += 1
    lnr = len(ret)
    dat = np.zeros((lnr, 3))
    for i in sorted(ret):
        dat[i, 0] = ret[i]['al']
        dat[i, 1] = ret[i]['items'][0]
        dat[i, 2] = ret[i]['items'][1]        
        
