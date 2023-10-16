import numpy as np
from numba import njit, prange
import kwc
import pypardiso as pyp


@njit
def rungekutta(meta, cells, x, xn, shape, deriv, coords, weight, kval, abut, krk, valM, istage):
    dxh = np.zeros((meta.nnode*meta.ndof))
    for jstage in range(meta.nstage):
        dxh = dxh + abut[istage,jstage]*krk[:,jstage]*meta.dtime
    x = xn + dxh

    sval, res = assembly(meta, krk, coords, cells, kval, x, xn, istage, shape, deriv, weight)

    iter = meta.ndof**2*meta.nenode**2*meta.nelem
    for k in kval:
        sval[iter] = sval[k]
        iter += 1

    val = sval*abut[istage,istage]*meta.dtime + valM

    return val, res


@njit(parallel=True)
def assembly(meta, krk, coords, cells, kval, x, xn, istage, shape, deriv, weight):
    sres = np.zeros((meta.ndof*meta.nnode))
    sval = np.zeros((meta.nval))

    eres_buffer = np.zeros((meta.ndof*meta.nenode, meta.nelem))

    for ielem in prange(meta.nelem):
        ecoords = np.zeros((meta.ndim, meta.nenode))
        ex = np.zeros((meta.ndof, meta.nenode))
        exn = np.zeros((meta.ndof, meta.nenode))
        ekrk = np.zeros((meta.ndof, meta.nenode))
        for a in range(meta.nenode):
            for i in range(meta.ndim):
                ecoords[i, a] = coords[cells[ielem, a], i]
            for i in range(meta.ndof):
                ex[i, a] = x[meta.ndof*cells[ielem, a]+i]
                exn[i, a] = xn[meta.ndof*cells[ielem, a]+i]
                ekrk[i, a] = krk[meta.ndof*cells[ielem, a]+i, istage]

        eres, ejac = kwc.element(meta, deriv, ecoords, ielem, ex, weight, exn, shape, ekrk)
        eres_buffer[:,ielem] = eres

        for a in range(meta.nenode):
            for b in range(meta.nenode):
                for i in range(meta.ndof):
                    for k in range(meta.ndof):
                        iter = k+meta.ndof*(i+meta.ndof*(b+meta.nenode*(a+meta.nenode*(ielem))))
                        sval[iter] = ejac[meta.ndof*a+i, meta.ndof*b+k]

    for ielem in range(meta.nelem):
        for a in range(meta.nenode):
            for i in range(meta.ndof):
                rw = meta.ndof*cells[ielem, a]+i
                sres[rw] += eres_buffer[meta.ndof*a+i, ielem]

    return sval, sres


def post_newton(meta, krk, coords, cells, mdof, sdof, kval, x, xn, shape, deriv, weight, fdof, abut, bbut, M):

    for istage in range(meta.nstage,meta.nstage+meta.nstageadd):
        dxh = np.zeros((meta.nnode*meta.ndof))
        for jstage in range(meta.nstage+meta.nstageadd):
            dxh = dxh + abut[istage,jstage]*krk[:,jstage]*meta.dtime
        xh = xn + dxh

        buffer = np.zeros((meta.ndof*meta.nnode,meta.nstage))
        _, res = assembly(meta, buffer, coords, cells, kval, xh, xn, 0, shape, deriv, weight)

        res[mdof[0,:]] += res[sdof[0,:]]
        res[mdof[1,:]] += res[sdof[1,:]]

        dk = np.zeros((meta.nnode*meta.ndof))
        dk[fdof] = pyp.spsolve(M, -res[fdof])

        dk[sdof[0,:]] = dk[mdof[0,:]]
        dk[sdof[1,:]] = dk[mdof[1,:]]
        krk[:,istage] += dk

    nbut = bbut.shape[0]
    dk = np.zeros((meta.nnode*meta.ndof, nbut))
    for ibut in range(nbut):
        for istage in range(meta.nstage+meta.nstageadd):
            dk[:,ibut] = dk[:,ibut] + bbut[ibut,istage]*krk[:,istage]

    return dk


@njit
def scaling(meta, kval, coords, cells, deriv, shape, weight):
    valM = np.zeros((meta.nval))

    for ielem in prange(meta.nelem):
        ecoords = np.zeros((meta.ndim, meta.nenode))

        for a in range(meta.nenode):
            for i in range(meta.ndim):
                ecoords[i, a] = coords[cells[ielem, a], i]

        eM = np.zeros((meta.ndof*meta.nenode, meta.ndof*meta.nenode))
        for igaus in range(meta.ngaus):
            xjacm = np.zeros((meta.ndim, meta.ndim))
            for j in range(meta.ndim):
                for k in range(meta.ndim):
                    for a in range(meta.nenode):
                        xjacm[j,k] += deriv[j,a,igaus]*ecoords[k,a]

            detjacb = xjacm[0,0]*xjacm[1,1]-xjacm[1,0]*xjacm[0,1]

            for idof in range(meta.ndof):
                for a in range(meta.nenode):
                    row = meta.ndof*a
                    for b in range(meta.nenode):
                        col = meta.ndof*b
                        eM[row+idof,col+idof] += shape[a,igaus]*shape[b,igaus]*weight[igaus]*detjacb

        for a in range(meta.nenode):
            for b in range(meta.nenode):
                for i in range(meta.ndof):
                    for k in range(meta.ndof):
                        iter = k+meta.ndof*(i+meta.ndof*(b+meta.nenode*(a+meta.nenode*(ielem))))
                        valM[iter] = eM[meta.ndof*a+i, meta.ndof*b+k]

    iter = meta.ndof**2*meta.nenode**2*meta.nelem
    for k in kval:
        valM[iter] = valM[k]
        iter += 1
    return valM


@njit
def sfr(ndim, ngaus, nenode, xi):
    if ndim == 2:
        if nenode == 4:
            shape = np.zeros((nenode, ngaus))
            deriv = np.zeros((ndim, nenode, ngaus))
            for igaus in range(ngaus):
                s, t = xi[0, igaus], xi[1, igaus]
                st = s*t
                buffershape = np.array([1.0-t-s+st, 1.0-t+s-st, 1.0+t+s+st, 1.0+t-s-st])*0.25
                bufferderiv = np.array([[-1.0+t, 1.0-t, 1.0+t, -1.0-t], [-1.0+s, -1.0-s, 1.0+s, 1.0-s]])*0.25
                shape[:, igaus] = buffershape
                deriv[:, :, igaus] = bufferderiv
        elif nenode == 9:
            shape = np.zeros((nenode, ngaus))
            deriv = np.zeros((ndim, nenode, ngaus))
            for igaus in range(ngaus):
                s, t = xi[0, igaus], xi[1, igaus]
                st = s*t
                buffershape = np.array([0.25*(s**2-s)*(t**2-t), 0.25*(s**2+s)*(t**2-t), 0.25*(s**2+s)*(t**2+t), 0.25*(s**2-s)*(t**2+t),
                                        0.5*(1-s**2)*(t**2-t), 0.5*(s**2+s)*(1-t**2), 0.5*(1-s**2)*(t**2+t), 0.5*(s**2-s)*(1-t**2), (1-s**2)*(1-t**2)])
                bufferderiv = np.array([[0.25*(2*s-1)*(t**2-t), 0.25*(2*s+1)*(t**2-t), 0.25*(2*s+1)*(t**2+t), 0.25*(2*s-1)*(t**2+t),
                                        0.5*(-2*s)*(t**2-t), 0.5*(2*s+1)*(1-t**2), 0.5*(-2*s)*(t**2+t), 0.5*(2*s-1)*(1-t**2), (-2*s)*(1-t**2)],
                                        [0.25*(s**2-s)*(2*t-1), 0.25*(s**2+s)*(2*t-1), 0.25*(s**2+s)*(2*t+1), 0.25*(s**2-s)*(2*t+1),
                                        0.5*(1-s**2)*(2*t-1), 0.5*(s**2+s)*(-2*t), 0.5*(1-s**2)*(2*t+1), 0.5*(s**2-s)*(-2*t), (1-s**2)*(-2*t)]])
                shape[:, igaus] = buffershape
                deriv[:, :, igaus] = bufferderiv
        else:
            raise RuntimeError('Number of element nodes not supported.')
    else:
        raise RuntimeError('Number of dimensions not supported.')
    return shape, deriv


@njit
def gauss(ndim, ngaus, nenode):
    if ndim == 2:
        if nenode == 4:
            xi = np.array([[-0.57735026918963, 0.57735026918963, -0.57735026918963, 0.57735026918963],
                           [-0.57735026918963, -0.57735026918963, 0.57735026918963, 0.57735026918963]])
            weight = np.array([1.0, 1.0, 1.0, 1.0])
        elif nenode == 9:
            x = np.sqrt(3/5)
            xi = np.array([[-x, 0, x, -x, 0, x, -x, 0, x],
                           [-x, -x, -x, 0, 0, 0, x, x, x]])
            a, b = 5/9, 8/9
            weight = np.array([a*a, a*b, a*a, a*b, b*b, a*b, a*a, a*b, a*a])
        else:
            raise RuntimeError('Number of element nodes not supported.')
    else:
        raise RuntimeError('Number of dimensions not supported.')
    return xi, weight
