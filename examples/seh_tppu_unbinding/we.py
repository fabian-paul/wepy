"""Example of using wepy to simulate a protein-ligand complex for
unbinding. Uses OpenMM to run dynamics using CUDA on the specified
number of GPU workers. This is the "some assembly required" version of
running a wepy simulation, which showcases how wepy can be built from
a collection of parts. Most of those parts are provided by wepy but
can be built from scratch allowing for maximum customizability.

The inputs to this script can be generated by running the
'minimize.py' script, which generates an initial structure from the
crystal structure.

This script is divided into a few sections:

1. Import necessary components. You can use different components by
importing different parts in this section.

2. Specify all parameters, modify for your needs, each is explained in
comments.

3. Set the I/O paths

4. Define functions for preparing the system. You could also build
your own components for use in wepy, such as a new distance like in
the Lennard Jones example.

5. The "main" function. This is needed because of the use of
multiprocessing. See other examples that don't use multiprocessing
work mappers to see how to do this without a main function, which is a
little simpler although can only ever be in serial.

  a. Set up the OpenMMRunner. This creates a system from the force
    field parameter files and protein structure and topology
    files. Also creates various "-stats" like barostat and
    thermostat. Initializes the integrator.

  b. Initialize the distance metric. The distance metric requires the
    indices of the binding site and the ligand within the coordinates
    which are determined at runtime from the crystal structure.

  c. Initialize the WExplore resampler with the distance metric and
    parameters. These parameters are taken from the paper Lotz and
    Dickson. 2018. JACS 140 (2) pp. 618-628 (DOI: 10.1021/jacs.7b08572)

  d. Initialize the Boundary conditions, because this is a
    nonequilibrium simulation.

  e. Initialize the reporters. This includes the WepyHDF5 reporter
  which produces an HDF5 file with all the data from the run. Here you
  can customize a lot of features related to saving data efficiently
  using reduced representations of the positions and setting some data
  to only be saved at certain intervals (i.e. sparse fields).

  f. Initialize the work mapper by specifying the number of workers and
    the type of worker to be used.

  g. Create the walkers with the initial states for each with uniform
  weights.

  h. Initialize the simulation manager with all the components.

  i. Run the simulations using the simulation manager according to the
  comman line arguments for number of runs, number of cycles, and
  number of steps of integrations.

This produces an HDF5 file that can then be used for analysis.

This script also runs with the debugging prints turned on so a fairly
large amount of text will be written to STDOUT. This can be turned off
by setting the debug_prints option to main to False.

"""
import os
import os.path as osp
import pickle
import logging
import multiprocessing as mp

import numpy as np
import mdtraj as mdj

# OpenMM libraries for setting up simulation objects and loading
# the forcefields
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

## Wepy classes

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper

# the runner for running dynamics and making and it's particular
# state class
from wepy.runners.openmm import OpenMMRunner, OpenMMState, OpenMMGPUWorker, UNIT_NAMES
from wepy.walker import Walker

# classes for making the resampler
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.resampling.resamplers.wexplore import WExploreResampler

# A standard Boundary condition object for unbinding
from wepy.boundary_conditions.unbinding import UnbindingBC

# standard reporters
from wepy.reporter.hdf5 import WepyHDF5Reporter

# reporter that saves a pickle of the important objects which may be
# useful for doing analysis after the run
from wepy.reporter.setup import SetupReporter
from wepy.reporter.restart import RestartReporter

# a reporter to show a dashboard in plaintext of current summarized
# results of the simulation
from wepy.reporter.dashboard import WExploreDashboardReporter

## PARAMETERS

# OpenMM simulation parameters
# cubic simulation side length
CUBE_LENGTH = 8.2435*unit.nanometer
# angles of the cubic simulation box
CUBE_ANGLE = 90*unit.degree
# distance cutoff for non-bonded interactions
NONBONDED_CUTOFF = 1.0 * unit.nanometer

# Monte Carlo Barostat
# pressure to be maintained
PRESSURE = 1.0*unit.atmosphere
# temperature to be maintained
TEMPERATURE = 300.0*unit.kelvin
# frequency at which volume moves are attempted
VOLUME_MOVE_FREQ = 50

# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.

# CUDA is the best for NVIDIA GPUs
PLATFORM = 'CUDA'

# Langevin Integrator
FRICTION_COEFFICIENT = 1/unit.picosecond
# step size of time integrations
STEP_SIZE = 0.002*unit.picoseconds

# Distance metric parameters, these are not used in OpenMM and so
# don't need the units associated with them explicitly, so be careful!

# distance from the ligand in the crystal structure used to determine
# the binding site, used to align ligands in the Unbinding distance
# metric
BINDING_SITE_CUTOFF = 0.8 # in nanometers

# the residue id for the ligand so that it's indices can be determined
LIG_RESID = "2RV"

# Resampler parameters

# the maximum weight allowed for a walker
PMAX = 0.1
# the minimum weight allowed for a walker
PMIN = 1e-12

# the maximum number of regions allowed under each parent region
MAX_N_REGIONS = (10, 10, 10, 10)

# the maximum size of regions, new regions will be created if a walker
# is beyond this distance from each voronoi image unless there is an
# already maximal number of regions
MAX_REGION_SIZES = (1, 0.5, .35, .25) # nanometers

# boundary condition parameters

# maximum distance between between any atom of the ligand and any
# other atom of the protein, if the shortest such atom-atom distance
# is larger than this the ligand will be considered unbound and
# restarted in the initial state
CUTOFF_DISTANCE = 1.0 # nm

# reporting parameters

# these are the properties of the states (i.e. from OpenMM) which will
# be saved into the HDF5
SAVE_FIELDS = ('positions', 'box_vectors', 'velocities')
# these are the names of the units which will be stored with each
# field in the HDF5
UNITS = UNIT_NAMES
# this is the frequency to save the full system as an alternate
# representation, the main "positions" field will only have the atoms
# for the protein and ligand which will be determined at run time
ALL_ATOMS_SAVE_FREQ = 10
# we can specify some fields to be only saved at a given frequency in
# the simulation, so here each tuple has the field to be saved
# infrequently (sparsely) and the cycle frequency at which it will be
# saved
SPARSE_FIELDS = (('velocities', 10),
                )

## INPUTS/OUTPUTS

# the inputs directory
inputs_dir = osp.realpath('./inputs')
# the outputs path
outputs_dir = osp.realpath('./outputs')
# make the outputs dir if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)

# inputs filenames
json_top_filename = "sEH_TPPU_system.top.json"
omm_state_filename = "initial_openmm_state.pkl"
charmm_psf_filename = 'sEH_TPPU_system.psf'
charmm_param_files = ['all36_cgenff.rtf',
                      'all36_cgenff.prm',
                      'all36_prot.rtf',
                      'all36_prot.prm',
                      'tppu.str',
                      'toppar_water_ions.str']

starting_coords_pdb = 'sEH_TPPU_system.pdb'

# outputs
hdf5_filename = 'results.wepy.h5'
dashboard_filename = 'wepy.dash.org'
setup_state_filename = 'setup.pkl'
restart_state_filename = 'restart.pkl'

# normalize the input paths
json_top_path = osp.join(inputs_dir, json_top_filename)
omm_state_path = osp.join(inputs_dir, omm_state_filename)
charmm_psf_path = osp.join(inputs_dir, charmm_psf_filename)
charmm_param_paths = [osp.join(inputs_dir, filename) for filename
                      in charmm_param_files]

pdb_path = osp.join(inputs_dir, starting_coords_pdb)

# normalize the output paths
hdf5_path = osp.join(outputs_dir, hdf5_filename)
dashboard_path = osp.join(outputs_dir, dashboard_filename)
setup_state_path = osp.join(outputs_dir, setup_state_filename)
restart_state_path = osp.join(outputs_dir, restart_state_filename)

def ligand_idxs(mdtraj_topology, ligand_resid):
    return mdtraj_topology.select('resname "{}"'.format(ligand_resid))

def protein_idxs(mdtraj_topology):
    return np.array([atom.index for atom in mdtraj_topology.atoms if atom.residue.is_protein])


def binding_site_atoms(mdtraj_topology, ligand_resid, coords):

    # selecting ligand and protein binding site atom indices for
    # resampler and boundary conditions
    lig_idxs = ligand_idxs(mdtraj_topology, ligand_resid)
    prot_idxs = protein_idxs(mdtraj_topology)

    # select water atom indices
    water_atom_idxs = mdtraj_topology.select("water")
    #select protein and ligand atom indices
    protein_lig_idxs = [atom.index for atom in mdtraj_topology.atoms
                        if atom.index not in water_atom_idxs]

    # make a trajectory to compute the neighbors from
    traj = mdj.Trajectory([coords], mdtraj_topology)

    # selects protein atoms which have less than 8 A from ligand
    # atoms in the crystal structure
    neighbors_idxs = mdj.compute_neighbors(traj, BINDING_SITE_CUTOFF, lig_idxs)

    # selects protein atoms from neighbors list
    binding_selection_idxs = np.intersect1d(neighbors_idxs, prot_idxs)

    return binding_selection_idxs

def main(n_runs, n_cycles, steps, n_walkers, n_workers=1, debug_prints=False, seed=None):
    ## Load objects needed for various purposes

    # load a json string of the topology
    with open(json_top_path, mode='r') as rf:
        sEH_TPPU_system_top_json = rf.read()

    # an openmm.State object for setting the initial walkers up
    with open(omm_state_path, mode='rb') as rf:
        omm_state = pickle.load(rf)

    ## set up the OpenMM Runner

    # load the psf which is needed for making a system in OpenMM with
    # CHARMM force fields
    psf = omma.CharmmPsfFile(charmm_psf_path)

    # set the box size lengths and angles
    lengths = [CUBE_LENGTH for i in range(3)]
    angles = [CUBE_ANGLE for i in range(3)]
    psf.setBox(*lengths, *angles)

    # charmm forcefields parameters
    params = omma.CharmmParameterSet(*charmm_param_paths)

    # create a system using the topology method giving it a topology and
    # the method for calculation
    system = psf.createSystem(params,
                              nonbondedMethod=omma.CutoffPeriodic,
                              nonbondedCutoff=NONBONDED_CUTOFF,
                              constraints=omma.HBonds)

    # make this a constant temperature and pressure simulation at 1.0
    # atm, 300 K, with volume move attempts every 50 steps
    barostat = omm.MonteCarloBarostat(PRESSURE, TEMPERATURE, VOLUME_MOVE_FREQ)

    # add it as a "Force" to the system
    system.addForce(barostat)

    # make an integrator object that is constant temperature
    integrator = omm.LangevinIntegrator(TEMPERATURE,
                                        FRICTION_COEFFICIENT,
                                        STEP_SIZE)

    # set up the OpenMMRunner with the system
    runner = OpenMMRunner(system, psf.topology, integrator, platform=PLATFORM)


    # the initial state, which is used as reference for many things
    init_state = OpenMMState(omm_state)

    ## Make the distance Metric

    # load the crystal structure coordinates
    crystal_traj = mdj.load_pdb(pdb_path)

    # get the atoms in the binding site according to the crystal structure
    bs_idxs = binding_site_atoms(crystal_traj.top, LIG_RESID, crystal_traj.xyz[0])
    lig_idxs = ligand_idxs(crystal_traj.top, LIG_RESID)
    prot_idxs = protein_idxs(crystal_traj.top)

    # make the distance metric with the ligand and binding site
    # indices for selecting atoms for the image and for doing the
    # alignments to only the binding site. All images will be aligned
    # to the reference initial state
    unb_distance = UnbindingDistance(lig_idxs, bs_idxs, init_state)

    ## Make the resampler

    # make a Wexplore resampler with default parameters and our
    # distance metric
    resampler = WExploreResampler(distance=unb_distance,
                                   init_state=init_state,
                                   max_n_regions=MAX_N_REGIONS,
                                   max_region_sizes=MAX_REGION_SIZES,
                                   pmin=PMIN, pmax=PMAX)

    ## Make the Boundary Conditions

    # makes ref_traj and selects lingand_atom and protein atom  indices
    # instantiate a revo unbindingboudaryconditiobs
    ubc = UnbindingBC(cutoff_distance=CUTOFF_DISTANCE,
                      initial_state=init_state,
                      topology=crystal_traj.topology,
                      ligand_idxs=lig_idxs,
                      receptor_idxs=prot_idxs)


    ## make the reporters

    # WepyHDF5

    # make a dictionary of units for adding to the HDF5
    # open it in truncate mode first, then switch after first run
    hdf5_reporter = WepyHDF5Reporter(hdf5_path, mode='w',
                                     # the fields of the State that will be saved in the HDF5 file
                                     save_fields=SAVE_FIELDS,
                                     # the topology in a JSON format
                                     topology=sEH_TPPU_system_top_json,
                                     # the resampler and boundary
                                     # conditions for getting data
                                     # types and shapes for saving
                                     resampler=resampler,
                                     boundary_conditions=ubc,
                                     # the units to save the fields in
                                     units=dict(UNITS),
                                     # sparse (in time) fields
                                     sparse_fields=dict(SPARSE_FIELDS),
                                     # sparse atoms fields
                                     main_rep_idxs=np.concatenate((lig_idxs, prot_idxs)),
                                     all_atoms_rep_freq=ALL_ATOMS_SAVE_FREQ
    )

    dashboard_reporter = WExploreDashboardReporter(dashboard_path, mode='w',
                                                   step_time=STEP_SIZE.value_in_unit(unit.second),
                                                   max_n_regions=resampler.max_n_regions,
                                                   max_region_sizes=resampler.max_region_sizes,
                                                   bc_cutoff_distance=ubc.cutoff_distance)

    setup_reporter = SetupReporter(setup_state_path, mode='w')

    restart_reporter = RestartReporter(restart_state_path, mode='w')

    reporters = [hdf5_reporter, dashboard_reporter, setup_reporter, restart_reporter]

    ## The work mapper

    # we use a mapper that uses GPUs
    work_mapper = WorkerMapper(worker_type=OpenMMGPUWorker,
                               num_workers=n_workers)

    ## Combine all these parts and setup the simulation manager

    # set up parameters for running the simulation
    # initial weights
    init_weight = 1.0 / n_walkers

    # a list of the initial walkers
    init_walkers = [Walker(OpenMMState(omm_state), init_weight) for i in range(n_walkers)]

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=ubc,
                          work_mapper=work_mapper,
                          reporters=reporters)


    ### RUN the simulation
    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps, debug_prints=True)
        print("Finished run: {}".format(run_idx))


if __name__ == "__main__":
    import time
    import multiprocessing as mp
    import sys
    import logging

    # needs to call spawn for starting processes due to CUDA not
    # tolerating fork
    mp.set_start_method('spawn')
    mp.log_to_stderr(logging.DEBUG)

    if sys.argv[1] == "--help" or sys.argv[1] == '-h':
        print("arguments: n_runs, n_cycles, n_steps, n_walkers, n_workers")
    else:

        n_runs = int(sys.argv[1])
        n_cycles = int(sys.argv[2])
        n_steps = int(sys.argv[3])
        n_walkers = int(sys.argv[4])
        n_workers = int(sys.argv[5])

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))

        steps = [n_steps for i in range(n_cycles)]

        start = time.time()
        main(n_runs, n_cycles, steps, n_walkers, n_workers, debug_prints=True)
        end = time.time()

        print("time {}".format(end-start))
