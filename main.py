import time
import numpy as np
import input, comp, output, classes
import scipy.sparse as spp
import pypardiso as pyp


def main():
    print('\n----- Start -----\n')
    sim_start = time.time()

    iadap, nrunge = 1, 17
    nstep, ntime = 1000, 1.0
    domain, refine = 1.0, 7

    meta = classes.meta(nstep, ntime, nrunge, iadap)
    meta, abut, bbut, cbut = input.butcher(meta)

    meta, coords, cells, mesh, nodes, borders, mdof, sdof, fdof = input.geometry(meta, domain, refine)
    xi, weight = comp.gauss(meta.ndim, meta.ngaus, meta.nenode)
    shape, deriv = comp.sfr(meta.ndim, meta.ngaus, meta.nenode, xi)
    x = input.initial(meta, coords, mdof, sdof)
    output.results(meta, x, mesh, cells, shape, nodes, 0)
    I, J, kval = input.triplet(meta, refine, mdof, sdof, cells)

    valM = comp.scaling(meta, kval, coords, cells, deriv, shape, weight)
    M = spp.csr_matrix((valM, (I, J)), shape=(meta.nnode*meta.ndof,meta.nnode*meta.ndof))[np.ix_(fdof, fdof)]
    times = np.zeros(1)

    result = np.zeros((1, meta.ndof*meta.nnode+1))
    result[0,:] = np.append(0,x)

    print('\n----- Begin time stepping algorithm -----\n')
    print('Begin step number: {:5.0f}'.format(meta.istep+1))
    while (meta.ntime - meta.time) > meta.tol:
        step_start = time.time()
        xn = np.array(x)
        krk = np.zeros((meta.ndof*meta.nnode,meta.nstage+meta.nstageadd))

        for istage in range(meta.nstage):
            iit = 0
            print('        ----- Newton Iterations ----- Stage:', istage+1,'-----------------------------------------------------')
            while True:
                try:
                    iter_start = time.time()
                    iit += 1

                    val, res = comp.rungekutta(meta, cells, x, xn, shape, deriv, coords, weight, kval, abut, krk, valM, istage)
                    jac = spp.csr_matrix((val, (I, J)), shape=(meta.nnode*meta.ndof,meta.nnode*meta.ndof))

                    res[mdof[0,:]] += res[sdof[0,:]]
                    res[mdof[1,:]] += res[sdof[1,:]]

                    error = np.linalg.norm(res[fdof])
                    if error <= meta.tol or iit >= meta.niter:
                        break

                    dk = np.zeros((meta.nnode*meta.ndof))
                    dk[fdof] = pyp.spsolve(jac[np.ix_(fdof, fdof)], -res[fdof])
                    dk[sdof[0,:]] = dk[mdof[0,:]]
                    dk[sdof[1,:]] = dk[mdof[1,:]]
                    krk[:,istage] += dk

                finally:
                    print('        Iteration count: {:5.0f}        Time: {:10.4f} seconds        Residual: {:10.4e}'.format(iit, time.time()-iter_start, error))

        dx = comp.post_newton(meta, krk, coords, cells, mdof, sdof, kval, x, xn, shape, deriv, weight, fdof, abut, bbut, M)
        if meta.iadap == 1:
            err = (dx[:,0] - dx[:,1])*meta.dtime
            norm_err = np.linalg.norm(err)
            meta.dtime = meta.dtime*min(max(0.9*(meta.etol/norm_err)**(1/(meta.norder+1)),0.2),2.0)
            meta.adapit += 1
        else:
            norm_err = 0.0

        if norm_err <= meta.etol:
            meta.time += meta.dtime
            meta.istep += 1
            x = xn + meta.dtime*dx[:,0]

            times = np.append(times, meta.time)

            result = np.vstack([result, np.append(meta.time,x)])
            meta.adapit = 0

            output.results(meta, x, mesh, cells, shape, nodes, times)

            print('        --------------------------------------------------------------------------------------------')
            print('End step number: {:5.0f}    Time: {:8.4f}s    Step compute time: {:8.4f}s   Total compute time: {:8.4f}s'.format(meta.istep, meta.time, time.time()-step_start, time.time()-sim_start))
            print('----------------------------------------------------------------------------------------------------')
            print('----------------------------------------------------------------------------------------------------')

    print('\n ----- Done -----')
    # output.convergence_orders(meta, result)


if __name__ == "__main__":
    main()
