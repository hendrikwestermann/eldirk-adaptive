import numpy as np
from numba import njit


@njit(fastmath=True)
def element(meta, deriv, ecoords, ielem, ex, weight, exn, shape, ekrk):
    ejac = np.zeros((meta.ndof*meta.nenode, meta.ndof*meta.nenode))
    eres = np.zeros((meta.ndof*meta.nenode))

    eta, theta = ex[0, :], ex[1,:]

    alpha, omega = 0.04, 7.5
    a1, a2, a3 = 1.0, 0.95, 1.35
    d, e = 0.45, 0.05

    beta_eta = 0.01
    eye = np.array([[1,0],[0,1]])
    gamma = 2.0

    tau_eta = beta_eta*omega**2
    tau_theta = 25.0

    for igaus in range(meta.ngaus):
        xjacm = np.zeros((meta.ndim, meta.ndim))
        for i in range(meta.ndim):
            for j in range(meta.ndim):
                for k in range(meta.nenode):
                    xjacm[i,j] += deriv[i,k,igaus]*ecoords[j,k]

        detjacb = xjacm[0,0]*xjacm[1,1]-xjacm[1,0]*xjacm[0,1]
        xjaci = np.linalg.inv(xjacm)

        cartd = np.zeros((meta.ndim,meta.nenode))
        for i in range(meta.ndim):
            for j in range(meta.nenode):
                for k in range(meta.ndim):
                    cartd[i,j] += xjaci[i,k]*deriv[k,j,igaus]

        etagp, thetagp = 0.0, 0.0
        for a in range(meta.nenode):
            etagp += shape[a,igaus]*eta[a]
            thetagp += shape[a,igaus]*theta[a]

        gradeta = np.zeros((meta.ndim))
        gradtheta = np.zeros((meta.ndim))
        for a in range(meta.nenode):
            for k in range(meta.ndim):
                gradeta[k] += cartd[k,a]*eta[a]
                gradtheta[k] += cartd[k,a]*theta[a]

        normgradtheta = (gradtheta[0]**2+gradtheta[1]**2)**0.5

        n = gradtheta/normgradtheta
        gog = np.outer(gradtheta,gradtheta)
        c = np.tanh(gamma*normgradtheta)

        # crystallinity
        for a in range(meta.nenode):
            row = meta.ndof*a
            eres[row] += shape[a,igaus]*omega**2*(etagp-1.0) /tau_eta
            eres[row] += shape[a,igaus]*2*d*etagp*normgradtheta**2 /tau_eta
            for k in range(meta.ndim):
                eres[row] += cartd[k,a]*alpha**2*gradeta[k] /tau_eta

            for b in range(meta.nenode):
                col = meta.ndof*b
                ejac[row,col] += shape[a,igaus]*omega**2*shape[b,igaus] /tau_eta
                ejac[row,col] += shape[a,igaus]*2*d*normgradtheta**2*shape[b,igaus] /tau_eta
                for k in range(meta.ndim):
                    ejac[row,col] += cartd[k,a]*alpha**2*cartd[k,b] /tau_eta
                    ejac[row,col+1] += shape[a,igaus]*4*d*etagp*gradtheta[k]*cartd[k,b] /tau_eta

        if normgradtheta != 0.0:
            for a in range(meta.nenode):
                row = meta.ndof*a
                eres[row] += shape[a,igaus]*(a1+2*a2*etagp+3*a3*etagp**2)*normgradtheta /tau_eta
                for b in range(meta.nenode):
                    col = meta.ndof*b
                    ejac[row,col] += shape[a,igaus]*(2*a2+6*a3*etagp)*shape[b,igaus]*normgradtheta /tau_eta
                    for k in range(meta.ndim):
                        ejac[row,col+1] += shape[a,igaus]*(a1+2*a2*etagp+3*a3*etagp**2)*gradtheta[k]/normgradtheta*cartd[k,b] /tau_eta

        # orientation
        for a in range(meta.nenode):
            row = meta.ndof*a
            for k in range(meta.ndim):
                eres[row+1] += cartd[k,a]*2*(d*etagp**2+e)*gradtheta[k] /tau_theta
            for b in range(meta.nenode):
                col = meta.ndof*b
                for k in range(meta.ndim):
                    ejac[row+1,col+1] += cartd[k,a]*2*(d*etagp**2+e)*cartd[k,b] /tau_theta
                    ejac[row+1,col] += cartd[k,a]*4*(d*etagp)*gradtheta[k]*shape[b,igaus] /tau_theta

        if normgradtheta != 0.0:
            for a in range(meta.nenode):
                row = meta.ndof*a
                for k in range(meta.ndim):
                    eres[row+1] += cartd[k,a]*(a1*etagp+a2*etagp**2+a3*etagp**3)*n[k]*c /tau_theta
                for b in range(meta.nenode):
                    col = meta.ndof*b
                    for k in range(meta.ndim):
                        ejac[row+1,col] += cartd[k,a]*(a1+2*a2*etagp+3*a3*etagp**2)*shape[b,igaus]*n[k]*c /tau_theta
                        for l in range(meta.ndim):
                            ejac[row+1,col+1] += cartd[k,a]*(a1*etagp+a2*etagp**2+a3*etagp**3) \
                                *(eye[k,l]*c/normgradtheta-c/normgradtheta**3*gog[k,l]+(1-c**2)*gamma/normgradtheta**2*gog[k,l])*cartd[l,b] /tau_theta

        # slope
        for idof in range(meta.ndof):
            for a in range(meta.nenode):
                row = meta.ndof*a
                for b in range(meta.nenode):
                    eres[row+idof] += shape[a,igaus]*shape[b,igaus]*ekrk[idof,b]

    eres *= detjacb*weight[igaus]
    ejac *= detjacb*weight[igaus]

    return eres, ejac
