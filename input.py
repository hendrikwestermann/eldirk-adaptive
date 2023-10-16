import numpy as np
from numba import njit, prange
import pyvista as pv
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import sobel
from os.path import exists
import pickle


def geometry(meta, domain, refine):

    eps = int(np.sqrt(4**(refine-1)))
    center = (domain/2, domain/2, 0)

    # create 2D square mesh, extract node coordinates, extract node connecivity
    mesh = pv.Plane(center=center,i_size=domain, j_size=domain,i_resolution=eps, j_resolution=eps)
    mesh.point_data.clear()
    mesh.flip_x(inplace=True)
    mesh.points_to_double()
    meta.nnode, meta.nelem, meta.nenode = mesh.n_points, mesh.n_cells, 4
    cells = np.delete(mesh.faces.reshape(-1, 5),0,1)
    coords = np.array(np.delete(mesh.points, -1, axis=1))
    coords = np.round(coords,8)
    mesh.points = np.concatenate((coords, np.zeros((meta.nnode,1))), axis=1)

    # domain borders [bot, right, top, left]
    ncount = int(np.sqrt(meta.nnode))
    borders = np.zeros((meta.ndim*2, ncount), dtype=np.int_)
    borders[0,:] = np.where(coords[:,1] == 0.0)[0]
    borders[1,:] = np.where(coords[:,0] == domain)[0]
    borders[2,:] = np.where(coords[:,1] == domain)[0]
    borders[3,:] = np.where(coords[:,0] == 0.0)[0]

    master = np.zeros((meta.ndim, ncount), dtype=np.int_)
    slave = np.zeros((meta.ndim, ncount), dtype=np.int_)
    master[0,:] = borders[3,:]
    master[1,:] = borders[0,:]
    slave[0,:] = borders[1,:]
    slave[1,:] = borders[2,:]

    mdof = np.zeros((meta.ndim, master.shape[1]*meta.ndof), dtype=np.int_)
    sdof = np.zeros((meta.ndim, master.shape[1]*meta.ndof), dtype=np.int_)
    for j in range(meta.ndim):
        for inode in range(master.shape[1]):
            for idof in range(meta.ndof):
                mdof[j, inode+master.shape[1]*idof] = master[j, inode]*meta.ndof+idof
                sdof[j, inode+master.shape[1]*idof] = slave[j, inode]*meta.ndof+idof

    nodes = np.ones((meta.nnode, meta.nenode), dtype=np.int_)*(-1)
    fdof = np.setdiff1d(np.arange(meta.ndof*meta.nnode), np.array(sdof))

    return meta, coords, cells, mesh, nodes, borders, mdof, sdof, fdof


def initial(meta, coords, mdof, sdof):
    x = np.zeros((meta.ndof*meta.nnode))

    iinput = 2
    if iinput == 1:  # bicrystal input new
        x0 = np.max(coords[:,0])/2.0
        y0 = np.max(coords[:,1])/2.0
        radius = x0/4.0
        for inode in range(meta.nnode):
            x[inode*meta.ndof+1] = 0.8
            rad = np.sqrt((x0 - coords[inode, 0])**2 + (y0 - coords[inode, 1])**2)
            if rad <= radius:
                x[inode*meta.ndof+1] = -0.8

        x = sobel_filter(meta, x)
        x = gauss_filter(meta, x, [0.3, 0.2])

        for inode in range(meta.nnode):
            if x[inode*meta.ndof] <= 0.9:
                x[inode*meta.ndof] = 0.0

        x = gauss_filter(meta, x, [0.2, 0.0])

        x[sdof[0,:]] = x[mdof[0,:]]
        x[sdof[1,:]] = x[mdof[1,:]]

    elif iinput == 2:  # bicrystal input predetermined
        x = pickle.load(open('input/grain.p', "rb"))

    return x


def triplet(meta, refine, mdof, sdof, cells):

    if exists('input/I_{0:1d}_{1:1d}.p'.format(meta.nnode, meta.ndof)):

        I = pickle.load(open("input/I_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "rb"))
        J = pickle.load(open("input/J_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "rb"))
        kval = pickle.load(open("input/kval_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "rb"))

    else:
        I,J,kval = tripletInit(meta, refine, mdof, sdof, cells)

        pickle.dump(I, open("input/I_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "wb"))
        pickle.dump(J, open("input/J_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "wb"))
        pickle.dump(kval, open("input/kval_{0:1d}_{1:1d}.p".format(meta.nnode, meta.ndof), "wb"))

    meta.nval = len(I)

    return I, J, kval


@njit(parallel=True)
def tripletInit(meta, refine, mdof, sdof, cells):
    I = np.zeros((meta.nenode*meta.ndof*meta.nenode*meta.ndof*meta.nelem), dtype=np.int_)
    J = np.zeros((meta.nenode*meta.ndof*meta.nenode*meta.ndof*meta.nelem), dtype=np.int_)

    for ielem in prange(meta.nelem):
        for a in range(meta.nenode):
            for b in range(meta.nenode):
                for i in range(meta.ndof):
                    for k in range(meta.ndof):
                        iter = k+meta.ndof*(i+meta.ndof*(b+meta.nenode*(a+meta.nenode*(ielem))))
                        I[iter] = meta.ndof*cells[ielem, a]+i
                        J[iter] = meta.ndof*cells[ielem, b]+k

    kval = np.zeros((1), dtype=np.int_)

    orig = meta.nenode*meta.ndof*meta.nenode*meta.ndof
    init = orig*65/16
    add = int(2**(refine-1)*init/1.625 + 0.625*init/1.625)
    I = np.zeros((meta.nenode*meta.ndof*meta.nenode*meta.ndof*meta.nelem + add), dtype=np.int_)
    J = np.zeros((meta.nenode*meta.ndof*meta.nenode*meta.ndof*meta.nelem + add), dtype=np.int_)
    kval = np.zeros((add), dtype=np.int_)

    for ielem in prange(meta.nelem):
        for a in range(meta.nenode):
            for b in range(meta.nenode):
                for i in range(meta.ndof):
                    for k in range(meta.ndof):
                        iter = k+meta.ndof*(i+meta.ndof*(b+meta.nenode*(a+meta.nenode*(ielem))))
                        I[iter] = meta.ndof*cells[ielem, a]+i
                        J[iter] = meta.ndof*cells[ielem, b]+k

    iter = meta.ndof**2*meta.nenode**2*meta.nelem
    kiter = 0
    pdof = sdof.shape[1]
    size = len(I)

    # slave right I
    for k in range(size):
        Ik = I[k]
        for idof in range(pdof):
            if sdof[0,idof] == Ik:
                kval[kiter] = k
                I[iter] = mdof[0,idof]
                J[iter] = J[k]
                iter += 1
                kiter += 1
                break

    # slave right J
    for k in range(size):
        Jk = J[k]
        for idof in range(pdof):
            if sdof[0,idof] == Jk:
                kval[kiter] = k
                I[iter] = I[k]
                J[iter] = mdof[0,idof]
                iter += 1
                kiter += 1
                break

    # slave top I
    for k in range(size):
        Ik = I[k]
        for idof in range(pdof):
            if sdof[1,idof] == Ik:
                kval[kiter] = k
                I[iter] = mdof[1,idof]
                J[iter] = J[k]
                iter += 1
                kiter += 1
                break

    # slave top J
    for k in range(size):
        Jk = J[k]
        for idof in range(pdof):
            if sdof[1,idof] == Jk:
                kval[kiter] = k
                I[iter] = I[k]
                J[iter] = mdof[1,idof]
                iter += 1
                kiter += 1
                break

    kval = kval[kval!=0]

    return I, J, kval


def gauss_filter(meta, x, sigma):
    len = int(np.sqrt(meta.nnode))
    for idof in range(meta.ndof):
        foo = np.zeros((meta.nnode))
        for inode in range(meta.nnode):
            foo[inode] = x[inode*meta.ndof+idof]

        gauss = gaussian_filter(np.reshape(foo, (len, len)), [len/60*sigma[idof], len/60*sigma[idof]])
        buffer = gauss.ravel()

        for inode in range(meta.nnode):
            x[inode*meta.ndof+idof] = buffer[inode]
    return x


def sobel_filter(meta, x):
    len = int(np.sqrt(meta.nnode))
    buffer = np.zeros((meta.nnode))
    for inode in range(meta.nnode):
        buffer[inode] = x[inode*meta.ndof+1]
    foo = np.reshape(buffer, (len, len))
    sobx = sobel(foo,axis=-1)
    soby = sobel(foo,axis=0)

    bn_imgx = np.zeros([sobx.shape[0],sobx.shape[1]])
    sbl_max = np.amax(abs(sobx))
    bn_imgx = np.abs(sobx) >= (sbl_max/200.0)

    bn_imgy = np.zeros([soby.shape[0],soby.shape[1]])
    sbl_max = np.amax(abs(soby))
    bn_imgy = np.abs(soby) >= (sbl_max/200.0)

    sob = bn_imgx+bn_imgy
    sob[0,:] = False
    sob[:,0] = False
    sob[-1,:] = False
    sob[:,-1] = False

    foo[sob==True] = 0.0
    foo[sob==False] = 1.0

    # fig = plt.figure()
    # ax1 = fig.add_subplot(321)
    # ax2 = fig.add_subplot(322)
    # ax3 = fig.add_subplot(323)
    # ax4 = fig.add_subplot(324)
    # ax5 = fig.add_subplot(325)
    # ax1.imshow(sobx)
    # ax2.imshow(soby)
    # ax3.imshow(bn_imgx)
    # ax4.imshow(bn_imgy)
    # ax5.imshow(foo)
    # plt.show()

    buffer = foo.ravel()
    for inode in range(meta.nnode):
        x[inode*meta.ndof] = buffer[inode]

    return x


def butcher(meta):
    if meta.nrunge==1:  # RK1 Euler implicit
        # meta.name = 'rk1_euler_imp'
        meta.name = 'euler_implicit'
        meta.nstage = 1
        meta.norder = 1
        a = np.zeros((meta.nstage,meta.nstage))
        a[0,0] = 1
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,0] = 1
        c = np.array([[1]])
    elif meta.nrunge==2:  # RK1 Euler explicit
        meta.name = 'rk1_euler_exp'
        meta.nstage = 1
        meta.norder = 1
        a = np.zeros((meta.nstage,meta.nstage))
        a[0,0] = 0
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,0] = 1
        c = np.array([[0]])
    elif meta.nrunge==3:  # RK2 Ellsiepen
        meta.name = 'rk2_ellsiepen'
        meta.nstage = 2
        meta.norder = 2
        alpha = 1-1/np.sqrt(2)
        a = np.array([[alpha,0.0],[1-alpha,alpha]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([1-alpha,alpha])
        c = np.array([alpha,1.0])
    elif meta.nrunge==4:  # RK2 Trapez
        meta.name = 'rk2_trapez'
        meta.nstage = 2
        meta.norder = 2
        a = np.array([[0.0,0.0],[0.5,0.5]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([0.5,0.5])
        c = np.array([0.0,1.0])
    elif meta.nrunge==5:  # RK2 Heun
        meta.name = 'rk2_heun'
        meta.nstage = 2
        meta.norder = 2
        a = np.array([[0.0,0.0],[1.0,0.0]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([0.5,0.5])
        c = np.array([0.0,1.0])
    elif meta.nrunge==6:  # RK 2Midpoint
        meta.name = 'rk2_midpoint'
        meta.nstage = 1
        meta.norder = 2
        a = np.zeros((meta.nstage,meta.nstage))
        a[0,0] = 0.5
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([1])
        c = np.array([[0.5]])
    elif meta.nrunge==7:  # RK4 Classic
        meta.name = 'rk4_classic'
        meta.nstage = 4
        meta.norder = 4
        a = np.array([[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.0],[0.0,0.5,0.0,0.0],[0.0,0.0,1.0,0.0]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([1/6, 1/3, 1/3, 1/6])
        c = np.array([[0.0, 0.5, 0.5, 1]])
    elif meta.nrunge==8:  # RK3 Classic
        meta.name = 'rk3_classic'
        meta.nstage = 3
        meta.norder = 3
        a = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[-1.0,2.0,0.0]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([1/6, 2/3, 1/6])
        c = np.array([[0.0, 0.5, 1]])
    elif meta.nrunge==13:  # RK3 Euler implicit
        meta.name = 'rk2_euler_imp'
        meta.nstage = 1
        meta.norder = 1
        meta.nstageadd = 1
        a = np.array([[1.0,0.0],[0.5,0.0]])
        b = np.array([[0.0,1.0],[1.0,0.0]])
        c = np.array([0.0,0.5])
    elif meta.nrunge==14:  # RK3 Trapez
        meta.name = 'rk3_trapez'
        meta.nstage = 2
        meta.norder = 2
        meta.nstageadd = 1
        a = np.array([[0.0,0.0,0.0],[0.5,0.5,0.0],[3/8,1/8,0.0]])
        b = np.array([[1/6,1/6,2/3],[0.5,0.5,0.0]])
        c = np.array([0.0,1.0,0.5])
    elif meta.nrunge==15:  # RK3 Midpoint
        meta.name = 'rk3_midpoint'
        meta.nstage = 1
        meta.norder = 2
        meta.nstageadd = 2
        a = np.array([[0.5,0.0,0.0],[1/2-1/(2*np.sqrt(3)),0.0,0.0],[1/2-1/(2*np.sqrt(3)),1/np.sqrt(3),0.0]])
        b = np.array([[0.0,1/2,1/2],[1.0,0.0,0.0]])
        c = np.array([[0.5,1/2-1/(2*np.sqrt(3)),1/2+1/(2*np.sqrt(3))]])
    elif meta.nrunge==16:  # RK3 Euler implicit
        meta.name = 'rk3_euler_imp'
        meta.nstage = 1
        meta.norder = 1
        meta.nstageadd = 2
        a = np.array([[1.0,0.0,0.0],[0.5,0.0,0.0],[-5/4,3/2,0.0]])
        b = np.array([[2/9,1/3,4/9],[1.0,0.0,0.0]])
        c = np.array([[1.0,0.5,1/4]])
    elif meta.nrunge==17:  # RK3 Ellsiepen
        meta.name = 'rk3_ellsiepen'
        meta.nstage = 2
        meta.norder = 2
        meta.nstageadd = 1
        x = 1-1/np.sqrt(2)
        a = np.array([[x,0.0,0.0],[1.0-x,x,0.0],[x-1.0,1.0-x,0.0]])
        b = np.array([[1/(6*(x-x**2)),(2-3*x)/(6*(1-x)),(4*x-3*x**2-1)/(6*(x-x**2))],[1.0-x,x,0.0]])
        c = np.array([[x,1.0,0]])
    elif meta.nrunge==18:  # $RK4 Norsett
        meta.name = 'rk4_norsett'
        meta.nstage = 3
        meta.norder = 4
        x = 1.06858
        a = np.array([[x,0.0,0.0],[0.5-x,x,0.0],[2*x,1-4*x, x]])
        b = np.zeros((meta.nstage,meta.nstage))
        b[0,:] = np.array([1/(6*(1-2*x)**2),(3*(1-2*x)**2-1)/(3*(1-2*x)**2),1/(6*(1-2*x)**2)])
        c = np.array([[x,1/2,1-x]])
    elif meta.nrunge==19:  # RK4 Trapez
        meta.name = 'rk4_trapez'
        meta.nstage = 3
        meta.nstage = 2
        meta.norder = 2
        meta.nstageadd = 2
        a = np.array([[0.0,0.0,0.0,0.0],[0.5,0.5,0.0,0.0],[3/8,1/8,0.0,0.0],[1/6,1/6,2/3,0.0]])
        b = np.array([[1/6,-1/3,2/3,1/2],[0.5,0.5,0.0,0.0]])
        c = np.array([0.0,1.0,0.5,1.0])
    elif meta.nrunge==20:  # RK4 Euler implicit
        meta.name = 'rk4_euler_imp'
        meta.nstage = 1
        meta.norder = 1
        meta.nstageadd = 3
        a = np.array([[1,0.0,0.0,0.0],[1/2,0,0.0,0.0],[-3/2,3/2,0.0,0.0],[1/2,1/6,1/3,0.0]])
        b = np.array([[-1/3,2/3,1/6,1/2],[1,0,0.0,0.0]])
        c = np.array([1,1/2,0,1])
    elif meta.nrunge==21:  # RK4 Ellsiepen
        meta.name = 'rk4_ellsiepen'
        meta.nstage = 1
        meta.nstage = 2
        meta.norder = 2
        meta.nstageadd = 2
        x = 1-1/np.sqrt(2)
        a = np.array([[x,0.0,0.0,0.0],[1.0-x,x,0.0,0.0],[(-22*x**2+16*x-3)/(8*(x-1)*(3*x-1)**2),(4*x-1)*(6*x**2-4*x+1)/(8*(x-1)*(3*x-1)**2),0.0,0.0],
                  [(-72*x**5+228*x**4-198*x**3+88*x**2-25*x+3)/(6*(x-1)*(4*x-1)*(6*x**2-4*x+1)),
                   (12*x**3-30*x**2+11*x-1)/(6*(x-1)*(4*x-1)), (2*(3*x-1)**2)/(3*(6*x**2-4*x+1)), 0]])
        b = np.array([[-1/(6*(x-1)*(6*x**2-4*x+1)), (-6*x**2+9*x-2)/(6*(x-1)*(4*x-1)), (2*(3*x-1)**3)/(72*x**3-66*x**2+24*x-3), 1/2],[1.0-x,x,0.0,0.0]])
        c = np.array([[x,1.0,(2*x-1)/(2*(3*x-1))]])
    else:
        raise RuntimeError('Runge-Kutta type does not exist')
    return meta,a,b,c
