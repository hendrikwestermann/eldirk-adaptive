import pyvista as pv
import numpy as np
import pickle


def results(meta, x, mesh, cells, shape, nodes, times):

    for i in range(meta.ndof):
        mesh.point_data['{:1.0f}'.format(i)] = x[np.arange(meta.nnode)*meta.ndof+i]

    pv.save_meshio('output/results_' + str(meta.istep) + '.vtu', mesh)

    if meta.istep == 0:
        with open("output/time.pvd", "w") as pvd:
            pvd.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
            pvd.write('  <Collection>\n')

            pvd.write('<DataSet timestep="{:1f}" group="" part="0"\n'.format(0))
            pvd.write('       file="results_{:1d}.vtu"/>\n'.format(0))
            pvd.write('  </Collection>\n</VTKFile>')
    else:
        with open("output/time.pvd", "w") as pvd:
            pvd.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n')
            pvd.write('  <Collection>\n')

            for istep in range(len(times)):
                pvd.write('<DataSet timestep="{:1f}" group="" part="0"\n'.format(times[istep]))
                pvd.write('       file="results_{:1d}.vtu"/>\n'.format(istep))
            pvd.write('  </Collection>\n</VTKFile>')


def convergence_orders(meta, global_error):
    pickle.dump(global_error, open("orders/{0}_{1:1.8f}.p".format(meta.name, meta.dtime), "wb"))
