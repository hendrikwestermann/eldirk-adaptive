from numba.experimental import jitclass
from numba import int32, float64
from numba.types import string

specmeta = [
    ('nelem',int32),
    ('nnode',int32),
    ('nenode',int32),
    ('ndim',int32),
    ('ndof',int32),
    ('ngaus',int32),
    ('nstep',int32),
    ('netotx',int32),
    ('ntime',float64),
    ('dtime',float64),
    ('time',float64),
    ('nstage',int32),
    ('nrunge',int32),
    ('ntotx',int32),
    ('norder',int32),
    ('tol',float64),
    ('etol',float64),
    ('niter',float64),
    ('istep',int32),
    ('nstageadd',int32),
    ('iadap',int32),
    ('adapit',int32),
    ('name',string),
    ('nval',int32),
]


@jitclass(specmeta)
class meta:
    def __init__(self, nstep, ntime, nrunge, iadap):
        self.nelem = 0
        self.nnode = 0
        self.nenode = 4
        self.ndim = 2

        self.ngaus = 4

        self.nstep = nstep
        self.ntime = ntime
        self.netotx = 0

        self.ndof = 2
        self.dtime = self.ntime/self.nstep
        self.nstage = 0
        self.nrunge = nrunge
        self.ntotx = self.nnode*self.ndof
        self.norder = 0
        self.tol = 1e-12
        self.etol = 1e-3
        self.niter = 10
        self.time = 0
        self.istep = 0
        self.nstageadd = 0
        self.iadap = iadap
        self.adapit = 0
        self.name = ' '
        self.nval = 0
