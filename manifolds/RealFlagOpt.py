from __future__ import division
import numpy as np
import numpy.linalg as la
from numpy.random import randn
from numpy import zeros, zeros_like, trace
from scipy.linalg import expm, expm_frechet, null_space

from NaturalTangent.manifolds.tools import sbmat
from ManNullRange.manifolds.NullRangeManifold import NullRangeManifold
from ManNullRange.manifolds.tools import (sym)


def _calc_dim(dvec):
    s = 0
    for i in range(1, len(dvec)):
        for j in range(i):
            s += dvec[i]*dvec[j]
    return s

    
class RealFlagOpt(NullRangeManifold):
    """ This actually is Stiefel, with H and GammaH, B and Augmented
    
    """
    def __init__(self, dvec, alp=None,
                 log_stats=False,
                 log_method=None):
        self.dvec = np.array(dvec)
        self.n = dvec.sum()
        self.d = dvec[1:].sum()
        self._dimension = _calc_dim(dvec)
        self._codim = self.d * self.n - self._dimension
        self._point_layout = 1
        cs = dvec[:].cumsum() - dvec[0]
        self._g_idx = dict((i+1, (cs[i], cs[i+1]))
                           for i in range(cs.shape[0]-1))
        p = self.dvec.shape[0]-1
        self.p = p

        if alp is None:
            self.alpha = np.array([1, 1.])
        else:
            self.alpha = np.array(alp)
            
        self.log_stats = log_stats
        if log_method is None:
            self.log_method = 'trust-krylov'
        elif log_method.lower() in ['trust-ncg', 'trust-krylov']:
            self.log_method = log_method.lower()
        else:
            raise(ValueError(
                'log method must be one of trust-ncg or trust-krylov'))
        self.log_gtol = None
        self.lbd = self.make_lbd()

    def make_lbd(self):
        dd = self.dvec.shape[0]

        def coef(a, dd):
            if dd % 2 == 0:
                if a < dd // 2 - 1:
                    return - (dd // 2) + a + 1
                else:
                    return - (dd // 2) + a + 2
            else:
                if a < (dd-1)//2:
                    return -(dd-1)//2 + a
                else:
                    return -(dd-1)//2 + a + 1

        dsum = self.dvec[1:].cumsum()
        lbd = np.concatenate([np.ones(self.dvec[a+1])*coef(a, dd)
                              for a in range(dsum.shape[0])])
        # return .5 + lbd / lbd.sum()
        return lbd        

    def inner(self, X, Ba, Bb):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        al = self.alpha
        return al[0]*np.sum(Ba*Bb) + (al[1] - al[0])*np.sum((X.T@Ba)*(X.T@Bb))

    def __str__(self):
        self._name = "Real flag manifold dimension vector=(%s) alpha=%s" % (
            self.dvec, str(self.alpha))
        return self._name

    def base_inner_ambient(self, eta1, eta2):
        return np.sum(eta1 * eta2)

    def euclidean_dist(self, X, Y):
        """ Euclidean distance. Useful to compare two
        elememt
        """
        YTX = Y.T@X
        return np.sqrt(2*(np.sum(self.lbd * self.lbd) - np.trace(
            (YTX*self.lbd[None, :])@(YTX.T*self.lbd[None, :]))))

    @property
    def typicaldist(self):
        return np.sqrt(sum(self._dimension))

    def base_inner_E_J(self, a1, a2):
        raise np.sum([np.sum(a1[x]*a2[x]) for x in a1])
    
    def g(self, X, omg):
        alf = self.alpha
        return alf[0]*omg + (alf[1]-alf[0]) *\
            X @ (X.T @ omg)
        
    def g_inv(self, X, omg):
        ialp = 1/self.alpha
        return ialp[0]*omg + (ialp[1]-ialp[0]) * X @ (X.T @ omg)        
    
    def D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1]-alf[0]) * (xi @ (X.T @ eta) + X @ (xi.T @ eta))
    
    def contract_D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1] - alf[0])*(eta @ xi.T + xi @ eta.T) @ X

    def christoffel_form(self, X, xi, eta):
        ret = 0.5*self.D_g(X, xi, eta)
        ret += 0.5*self.D_g(X, eta, xi)
        ret -= 0.5*self.contract_D_g(X, xi, eta)
        return ret
        
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T

    def J(self, X, eta):
        ret = {}
        ph = self.dvec
        gidx = self._g_idx
        alpha = self.alpha
        p = self.dvec.shape[0]-1
        for r in range(p, 0, -1):
            if ph[r] == 0:
                continue
            r_g_beg, r_g_end = gidx[r]
            for s in range(p, 0, -1):
                if ph[s] == 0:
                    continue
                s_g_beg, s_g_end = gidx[s]
                if r == s:
                    if r >= 1:
                        rr = 1
                    else:
                        rr = 0
                    ret[r, r] = alpha[rr]*X[:, r_g_beg:r_g_end].T @\
                        eta[:, r_g_beg:r_g_end]

                elif s > r:
                    ret[r, s] = eta[:, r_g_beg:r_g_end].T @\
                        X[:, s_g_beg:s_g_end]

                    ret[r, s] += X[:, r_g_beg:r_g_end].T @\
                        eta[:, s_g_beg:s_g_end]
        return ret    

    def flag_sym(self, Xomg):
        """ 
        """
        ret = sym(Xomg)
        p = self.p
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret[bt:et, bt:et] = Xomg[bt:et, bt:et]
        return ret    

    def proj(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        return U - X@(self.flag_sym(X.T@U))

    def proj_g_inv(self, X, omg):
        ial = 1/self.alpha
        return ial[0]*omg + (ial[1] - ial[0])*X@(X.T@omg) -\
            ial[1]*X@self.flag_sym(X.T@omg)

    def D_proj(self, X, xi, U):
        """deriv of projection. U is in ambient
        return one in tangent
        """
        return - xi@(self.flag_sym(X.T@U)) -\
            X@(self.flag_sym(xi.T@U))
    
    def zerovec(self, X):
        return zeros_like(X)

    def retr(self, X, eta):
        """ Use expt and svd. reason for this is to avoid drift
        """
        # X1 = self.exp(X, eta)
        u, _, vh = la.svd(X + eta, full_matrices=False)
        return u @ vh

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        O, _ = la.qr(randn(
            self.n, self.d))
        return O
    
    def randvec(self, X):
        U = self.proj(X, randn(*X.shape))
        U = U / self.norm(X, U)
        return U

    def _rand_ambient(self):
        return randn(self.n, self.d)

    def _vec(self, E):
        return E.reshape(-1)

    def _unvec(self, vec):
        return vec.reshape(self.n, self.d)

    def GammaH(self, X, xi, omg):
        alf = self.alpha[1] / self.alpha[0]
        
        def Pi0(a):
            return a - X@(X.T@a)
        return X@self.flag_sym(xi.T@omg) + (1-alf)*Pi0(xi@(omg.T@X) +
                                                       omg@(xi.T@X))
    
    def ehess2rhess(self, X, egrad, ehess, H):
        alf = self.alpha[1] / self.alpha[0]
    
        PioEgrad = egrad - X@(X.T@egrad)
        omg = ehess - H@self.flag_sym(X.T@egrad) - (1-alf)*(
            PioEgrad@(X.T@H) + X@(PioEgrad.T@H))
        return self.proj_g_inv(X, omg)

    def rhess02(self, X, xi, eta, egrad, ehess):
        alf = self.alpha[1] / self.alpha[0]
        
        def Pi0(a):
            return a - X@(X.T@a)
        rr1 = X@self.flag_sym(xi.T@eta) +\
            (1-alf)*Pi0(xi@(eta.T@X) + eta@(xi.T@X))
        return np.sum(ehess*eta) - np.sum(egrad*rr1)

    def exp(self, Y, eta):
        """ Geodesics, the formula involves matrices of size 2d

        Parameters
        ----------
        Y    : a manifold point
        eta  : tangent vector
        
        Returns
        ----------
        gamma(1), where gamma(t) is the geodesics at Y in direction eta

        """
        K = eta - Y @ (Y.T @ eta)
        Yp, R = la.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T @eta
        x_mat = sbmat([[2*alf*A, -R.T],
                      [R, zeros((self.d, self.d))]])
        return sbmat([Y, Yp]) @ expm(x_mat)[:, :self.d] @ expm((1-2*alf)*A)

    def log(self, X, X1, show_steps=False, init_type=0):
        """
        This is Stiefel geodesics
        Only use init_type 0 now = may change in the future
        ret_aligned: return the aligment
        """
        if init_type != 0:
            print("Will init with zero vector. Other options are not yet available")

        alf = self.alpha[1]/self.alpha[0]
        d = self.dvec[1:].sum()
        sqrt2 = np.sqrt(2)
        
        def getQ():
            """ algorithm: find a basis in linear span of Y Y1
            orthogonal to Y
            """
            u, s, v = np.linalg.svd(
                np.concatenate([X, X1], axis=1), full_matrices=False)
            k = (s > 1e-14).sum()
            good = u[:, :k]@v[:k, :k]
            qs = null_space(X.T@good)
            Q, _ = np.linalg.qr(good@qs)
            return Q
        
        # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
        # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
        Q = getQ()
        k = Q.shape[1]
        p = self.p
        lbd = self.lbd

        def asym(mat):
            return 0.5*(mat - mat.T)
        
        def vec(A, R):
            # for A, take all blocks [ij with i > j]
            lret = []
            for r in range(1, p+1):
                gdc = self._g_idx                
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(r+1, p+1):
                    if s <= r:
                        continue
                    bs, es = gdc[s]
                    lret.append(A[br:er, bs:es].reshape(-1)*sqrt2)

            lret.append(R.reshape(-1))
            return np.concatenate(lret)

        def unvec(avec):
            A = np.zeros((d, d))
            R = np.zeros((k, d))
            gdc = self._g_idx
            be = 0
            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(r+1, p+1):
                    if s <= r:
                        continue
                    bs, es = gdc[s]
                    dr = er - br
                    ds = es - bs
                    A[br:er, bs:es] = (avec[be: be+dr*ds]/sqrt2).reshape(dr, ds)
                    A[bs:es, br:er] = - A[br:er, bs:es].T
                    be += dr*ds
            R = avec[be:].reshape(k, d)
            return A, R

        XQ = np.array(np.bmat([X, Q]))
        # X2 = XQ.T@X1@X1.T@XQ
        X2 = XQ.T@(X1*lbd[None, :])@X1.T@XQ
        
        def dist(v):
            #  = (dist0a(v) - d)*2
            alf = self.alpha[1] / self.alpha[0]
            A, R = unvec(v)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            exh = expm(x_mat)
            ex = expm((1-2*alf)*A)
            Mid = (ex*lbd[None, :])@ex.T
            return (- trace(X2@exh[:, :d]@Mid@exh[:, :d].T))

        def jac(v):
            alf = self.alpha[1] / self.alpha[0]
            gdc = self._g_idx
            A, R = unvec(v)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            exh = expm(x_mat)
            ex = expm((1-2*alf)*A)

            blk = np.zeros_like(exh)
            blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
            blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@X2@exh[:, :d]

            fexh = 2*expm_frechet(x_mat, blk@X2)[1]
            fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]            
                fexh[br:br, br:br] = 0
                fex[br:br, br:br] = 0

            return vec(
                (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),
                fexh[d:, :d] - fexh[:d, d:].T)    
        
        def make_vec(xi):
            return vec(X.T@xi, Q.T@xi)

        def hessp(v, xi):
            dlt = 1e-8
            return (jac(v+dlt*xi) - jac(v))/dlt

        def conv_to_tan(A, R):
            return X@A + Q@R

        from scipy.optimize import minimize
        # A0, R0 = make_init()
        # x0 = vec(A0, R0)
        adim = (self.dvec[1:].sum()*self.dvec[1:].sum() -
                (self.dvec[1:]*self.dvec[1:]).sum()) // 2
        tdim = d*k + adim

        x0 = np.zeros(tdim)
        
        def printxk(xk):
            print(la.norm(jac(xk)), dist(xk))

        if show_steps:
            callback = printxk
        else:
            callback = None
        res = {'fun': np.nan, 'x': np.zeros_like(x0),
               'success': False,               
               'message': 'minimizer exception'}
        try:
            if self.log_gtol is None:
                res = minimize(dist, x0, method=self.log_method,
                               jac=jac, hessp=hessp, callback=callback)
            else:
                res = minimize(dist, x0, method=self.log_method,
                               jac=jac, hessp=hessp, callback=callback,            
                               options={'gtol': self.log_gtol})
        except Exception:
            pass
        
        stat = [(a, res[a]) for a in res.keys() if a not in ['x', 'jac']]
        A1, R1 = unvec(res['x'])
        ret_mat = conv_to_tan(A1, R1)
        if self.log_stats:
            return ret_mat, stat
        else:
            return ret_mat


