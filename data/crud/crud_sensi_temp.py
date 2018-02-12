"""!
\file crud_sensitivity_ex.py
\brief Simple mongoose sensitivity study ex
"""
from __future__ import print_function
import sys
sys.path.append('../.')
from pygoose import PyGoose
import numpy as np
from six import iteritems
from mpi4py import MPI
from tables import *


class CoolantTableFmt(IsDescription):
    """!
    @brief Pytables storage layout for coolant chem and kinetics
    """
    c_T = Float32Col()
    c_NiFe = Float32Col()
    c_Ni = Float32Col()
    c_Fe = Float32Col()
    c_B = Float32Col()
    c_Li = Float32Col()
    c_VH2 = Float32Col()
    # Diffustion constants
    D_Ni = Float32Col(dflt=0.719e-5)
    D_BOH3 = Float32Col(dflt=1.07e-5)
    D_Fe = Float32Col(dflt=0.712e-5)
    D_Li = Float32Col(dflt=1.03e-5)
    D_H2 = Float32Col(dflt=4.8e-5)
    # Chimney constants
    m_por = Float32Col(dflt=0.7)
    m_sol_dens = Float32Col(dflt=5.33)
    m_chimney_htc = Float32Col(dflt=6.7e2)  # W/cm^2-K
    m_chimney_dens = Float32Col(dflt=4.8e4)  # N / cm^2
    m_chimney_rad = Float32Col(dflt=4e-4)  # cm

class ThTableFmt(IsDescription):
    """!
    @brief Pytables storage layout for thermal hydraulic BCs
    """
    b_twall = Float32Col()
    b_tke = Float32Col()
    b_tcool = Float32Col()
    b_bhf = Float32Col()

class ResultTableFmt(IsDescription):
    """!
    @brief Pytables storage layout for crud results
    """
    r_cthick = Float32Col()
    r_cmass = Float32Col()
    r_bmass = Float32Col()
    r_th = Float32Col()


def tables_setup(h5file, dt):
    compression_filter = Filters(complevel=9, complib='zlib')
    h5_grp = h5file.create_group("/", 'results', 'Sweep resutls')
    result_table = h5file.create_table(h5_grp, 'result_' + str(int(dt)), ResultTableFmt,
                                       'Crud results', filters=compression_filter)
    th_table = h5file.create_table(h5_grp, 'th_' + str(int(dt)), ThTableFmt,
                                   'Surface Conditions', filters=compression_filter)
    coolant_table = h5file.create_table(h5_grp, 'cool_' + str(int(dt)), CoolantTableFmt,
                                        'Coolant and Kinetics', filters=compression_filter)
    return [result_table, th_table, coolant_table]

def append_to_h5(my_tables, c_matrix, b_matirx, r_matrix, dt):
    """!
    @brief Write results to pytables
    """
    result_row_handle = my_tables[0].row
    th_row_handle = my_tables[1].row
    cool_row_handle = my_tables[2].row
    for i in range(c_matrix.shape[0]):
        cool_row_handle['c_T'] = c_matrix[i, 0]
        cool_row_handle['c_NiFe'] = c_matrix[i, 1]
        cool_row_handle['c_Ni'] = c_matrix[i, 2]
        cool_row_handle['c_Fe'] = c_matrix[i, 3]
        cool_row_handle['c_B'] = c_matrix[i, 4]
        cool_row_handle['c_Li'] = c_matrix[i, 5]
        cool_row_handle['c_VH2'] = c_matrix[i, 6]
        #
        cool_row_handle['D_Ni'] = c_matrix[i, 7]
        cool_row_handle['D_BOH3'] = c_matrix[i, 8]
        cool_row_handle.append()
        #
        th_row_handle['b_twall'] = b_matirx[i, 0]
        th_row_handle['b_tke'] = b_matirx[i, 1]
        th_row_handle['b_tcool'] = b_matirx[i, 2]
        th_row_handle['b_bhf'] = b_matirx[i, 3]
        th_row_handle.append()
        #
        result_row_handle['r_cthick'] = r_matrix[i, 0]
        result_row_handle['r_cmass'] = r_matrix[i, 1]
        result_row_handle['r_bmass'] = r_matrix[i, 2]
        result_row_handle['r_th'] = r_matrix[i, 3]
        result_row_handle.append()
    # clean up
    for i in range(len(my_tables)):
        my_tables[i].flush()


def grid_samples(pdict):
    for key, val in iteritems(pdict):
        pdict[key]['samples'] = np.linspace(val['bounds'][0],
                val['bounds'][1], val['n'])


def gen_state_vectors(pdict, sampler_type='mc'):
    """!
    @brief Draw samples for each param in pdict.
    TODO: include a LHC sampler
    """
    grid_samples(pdict)
    list_of_params = [[] for i in range(len(pdict))]
    for val in pdict.values():
       list_of_params[val['id']] =  val['samples']
    # Grid samples into a np_ndarray
    grid = np.meshgrid(*list_of_params)
    if sampler_type is 'mc':
        # draw uncorrelated random samples
        for key, val in iteritems(pdict):
            # np.random.seed(123)
            mc_samples = np.random.uniform( \
                    val['bounds'][0], val['bounds'][1], len(grid[val['id']].flatten()))
            mc_samples.reshape(grid[val['id']].shape)
            grid[val['id']] = mc_samples
    return grid


def state_generator(bdict, cdict, sampler_type='mc'):
    c_states = gen_state_vectors(cdict, sampler_type)
    b_states = gen_state_vectors(bdict, sampler_type)
    c_flat_states = np.array([state.flatten() for state in c_states])
    b_flat_states = np.array([state.flatten() for state in b_states])
    return b_flat_states, c_flat_states

def pre_grow_crud(my_goose, build_in_time, twall_size, twall=625., tke=0.04, bhf=120.):
    """!
    @brief Builds in CRUD untill target thickness is reached.
    @return pygoose instance with built-in CRUD
    """
    # setup fixed arrays
    my_goose.set_boundary_data(Twall=np.ones(twall_size) * twall,
                               Tke=np.ones(twall_size) * tke,
                               Tcoolant=np.ones(twall_size) * 600.,
                               Bhf=np.ones(twall_size) * bhf)
    my_goose.step(build_in_time)
    my_goose.set_restart_point()
    import pdb; pdb.set_trace()
    """
    converged = False
    while not converged:
        my_goose.step(1.0)
        crud_results = my_goose.get_crud_pin_solution()
        cthick = crud_results[:, :, 0]
        if cthick.flatten()[0] >= target_cthick:
            converged = True
    """
    return my_goose

def mongoose_sweeper(bdict, cdict, my_goose, dt,
                     h5file_handle=None, kinetics_params={},
                     verbose=0, sampler_type='mc', build_in_time=None):
    """!
    @brief Generates results from parameter sweep
    @param bdict dictionary of TH boundary conditions
    @param cdict dictionary of coolant chemistry and kinetics parameters
    @param my_goose  PyGoose instance
    @param dt  time step size [days]
    @param h5file_handle pytables row handle (optional).  If supplied, param
        sweep results are dumped to hdf5
    @param kinetics_params dictionary of kinetics parameters
    @param sampler_type  str.  either 'mc' or 'factorial'
    @param verbose  Bool or int.  If true print case progress.
    @param target_cthick  float.  Crud to build-in before sweeping
    @yeild result matrix
    """
    if h5file_handle != None:
        my_tables = tables_setup(h5file_handle, dt)
    else:
        my_tables = None
    b_flat_states, c_flat_states = state_generator(bdict, cdict, sampler_type)
    for i in range(len(c_flat_states[0])):
        if verbose: print("Running Case %d of %d." % (i, len(c_flat_states[0])))
        # slice out unique coolant chemestry set
        cool_chem = c_flat_states[:, i]
        t_wall = b_flat_states[0, :]
        tke_wall = b_flat_states[1, :]
        t_cool = b_flat_states[2, :]
        bhf_wall = b_flat_states[3, :]
        # setup pin
        radius = 0.475e-2
        nth = 1
        my_goose.init_dongoose()
        my_goose.pin_setup(nth, np.ones(len(t_wall)) * 0.1, radius,
                           ipin=1, ir=1, jc=1, a=1)
        cool_chem_settings = {
            'T': cool_chem[cdict['T']['id']],
            'NiFe': cool_chem[cdict['NiFe']['id']],
            'Ni': cool_chem[cdict['Ni']['id']],
            'Fe': cool_chem[cdict['Fe']['id']],
            'B': cool_chem[cdict['B']['id']],
            'Li': cool_chem[cdict['Li']['id']],
            'VH2': cool_chem[cdict['VH2']['id']],
            }
        my_goose.set_coolant_data(**cool_chem_settings)
        if build_in_time:
            my_goose = pre_grow_crud(my_goose, build_in_time, np.size(t_wall))
        my_goose.set_boundary_data(Twall=t_wall, Tcoolant=t_cool, Bhf=bhf_wall, Tke=tke_wall)
        # set diffusion/kinetics for sensitivity study
        try:
            my_goose.clear_dongoose_options()
            for param_name, val in iteritems(cdict):
                my_goose.set_dongoose_option(param_name, cool_chem[val['id']])
        except KeyError:
            pass
        # force-set tunable diffusion/kinetics constants if specififed
        if kinetics_params != {}:
            my_goose.clear_dongoose_options()
            my_goose.set_dongoose_options(kinetics_params)
            cool_chem_settings.update(kinetics_params)
            my_goose.set_coolant_data(**cool_chem_settings)
        # step time
        # my_goose.set_dongoose_option('tstep_size', 0.1)
        # my_goose.set_dongoose_option('ODE_tstep_size', 0.1)
        my_goose.step(dt)
        # get crud solution
        crud_results = my_goose.get_crud_pin_solution()
        # clear pin
        my_goose.clear_dongoose()
        #
        # post process results
        cthick = crud_results[:, :, 0]
        cmass = crud_results[:, :, 1]
        bmass = crud_results[:, :, 2]
        rth = crud_results[:, :, 3]
        cool_conds_out = np.ones((len(t_wall), len(cool_chem)))
        cool_conds_out[:, :] *= np.array(cool_chem)
        bound_conds_out = np.array([t_wall, tke_wall, t_cool, bhf_wall]).T
        results_out = np.array([cthick[0], cmass[0], bmass[0], rth[0]]).T
        #
        if h5file_handle != None:
            append_to_h5(my_tables, cool_conds_out, bound_conds_out, results_out, dt)
        if h5file_handle != None:
            yield 0
        else:
            result_matrix = np.hstack((cool_conds_out, bound_conds_out, results_out))
            yield result_matrix


def run_sweep(h5file_handle, dt=50.):
    """!
    @brief Run all crud sims for dt
    @param h5file_handle  pytables file handle
    @param dt  float time step size [days]
    """
    my_goose = PyGoose(1)
    # TH boundary conds
    bdict = {'twall': {'bounds':[615., 625.], 'n':10, 'id': 0},
             'tke':   {'bounds':[0.08, 0.13], 'n':5, 'id': 1},
             'tcool': {'bounds':[565, 605.], 'n':5, 'id': 2},
             'bhf': {'bounds':[100, 150], 'n':3, 'id': 3},
            }
    # coolant chem & kinetics
    cdict = {'T': {'bounds':[550, 570.], 'n':2, 'id':0},
             'NiFe': {'bounds':[1.8, 2.5], 'n':2, 'id':1},
             'Ni': {'bounds':[0.15, 0.35], 'n':2, 'id':2},
             'Fe': {'bounds':[0.1, 0.1], 'n':1, 'id':3},
             'B': {'bounds':[800, 1200], 'n':2, 'id':4},
             'Li': {'bounds':[3.0, 3.0], 'n':1, 'id':5},
             'VH2': {'bounds':[30., 30.], 'n':1, 'id':6},
             #
             'D_Ni': {'bounds': [1e-6, 1e-4], 'n': 4, 'id':7},
             'D_BOH3': {'bounds': [1e-6, 1e-4], 'n': 4, 'id':8},
            }
    # sweep over all parameter values
    [case for case in mongoose_sweeper(bdict, cdict, my_goose, dt, h5file_handle, verbose=1)]


def run_temperature_sweep(h5file_handle, dt):
    my_goose = PyGoose(1)
    # TH boundary conds
    """
    bdict = {'twall': {'bounds':[610., 625.], 'n':100, 'id': 0},
             'tke':   {'bounds':[0.04, 0.04], 'n':1, 'id': 1},
             'tcool': {'bounds':[585, 585.], 'n':1, 'id': 2},
             'bhf': {'bounds':[100, 100], 'n':1, 'id': 3},
            }
    """
    """
    bdict = {'twall': {'bounds':[620., 620.], 'n':1, 'id': 0},
             'tke':   {'bounds':[0.001, 0.08], 'n':100, 'id': 1},
             'tcool': {'bounds':[585, 585.], 'n':1, 'id': 2},
             'bhf': {'bounds':[100, 100], 'n':1, 'id': 3},
            }
    """
    bdict = {'twall': {'bounds':[620., 620.], 'n':1, 'id': 0},
             'tke':   {'bounds':[0.05, 0.05], 'n':1, 'id': 1},
             'tcool': {'bounds':[585, 585.], 'n':1, 'id': 2},
             'bhf': {'bounds':[80, 120], 'n':100, 'id': 3},
            }
    # coolant chem & kinetics
    cdict = {'T': {'bounds':[585, 585.], 'n':1, 'id':0},
             'NiFe': {'bounds':[2.0, 2.0], 'n':1, 'id':1},
             'Ni': {'bounds':[0.15, 0.35], 'n':1, 'id':2},
             'Fe': {'bounds':[0.1, 0.1], 'n':1, 'id':3},
             'B': {'bounds':[1200, 1200], 'n':1, 'id':4},
             'Li': {'bounds':[3.0, 3.0], 'n':1, 'id':5},
             'VH2': {'bounds':[30., 30.], 'n':1, 'id':6},
             #
             'D_Ni': {'bounds': [1e-4, 1e-4], 'n': 1, 'id':7},
             'D_BOH3': {'bounds': [1e-4, 1e-4], 'n': 1, 'id':8},
            }
    # sweep over all parameter values
    [case for case in mongoose_sweeper(bdict, cdict, my_goose, dt, h5file_handle, verbose=1, sampler_type='factorial')]


def main():
    time_step_size_list = np.array([100., 200., 300., 400., 500.,])
    for i, dt in enumerate(time_step_size_list):
        h5f = open_file("pygoose_sweep_bhf" + str(dt) + ".h5", mode="w", title="Sweep")
        run_temperature_sweep(h5f, dt)
        h5f.close()


if __name__ == "__main__":
    """!
    @brief Generate boundary conds and coolant chem
    then run crud sims and plot/compute correlation coeffs
    """
    # time_step_days = 50.
    # # open hdf5 file for writing
    # h5f = open_file("pygoose_sweep.h5", mode="w", title="Sweep")
    # # run sims
    # run_sweep(h5f, time_step_days)
    # h5f.close()
    main()
