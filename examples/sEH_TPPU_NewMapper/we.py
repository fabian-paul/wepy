import sys

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

import mdtraj as mdj

import scoop.futures


from wepy.sim_manager import Manager
from wepy.resampling.wexplore  import WExplore2Resampler
from wepy.openmm import OpenMMRunner, OpenMMWalker ,OpenMMRunnerParallel
from wepy.gpumapper import GpuMapper

def make_table(init_walkers,n_walkers,cycle):
    import numpy as np
    import networkx as nx

    from wepy.walker import Walker
    from wepy.resampling.clone_merge import clone_parent_table, clone_parent_panel

    from tree import monte_carlo_minimization, make_graph

    if cycle == -1:
        result_template_str = "|".join(["{:^10}" for i in range(n_walkers + 1)])

        # print the initial walkers
        print("The initial walkers:")
        
         # slots
        slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
        print(slot_str)
   
        walker_weight_str = result_template_str.format("weight",
                                                        *[str(walker.weight) for walker in init_walkers])
        print(walker_weight_str)
         
    else :
        print("---------------------------------------------------------------------------------------")
        print("cycle: {}".format(cycle))    

        # print results for this cycle 
        print("Net state of walkers after resampling:")
        print("--------------------------------------")
        # slots
        slot_str = result_template_str.format("slot", *[i for i in range(n_walkers)])
        print(slot_str)
        # weights
        walker_weight_str = result_template_str.format("weight",
            *[str(walker.weight) for walker in cycle_walkers])
        print(walker_weight_str)

 
    
    
if __name__ == "__main__":


    psf = omma.CharmmPsfFile('sEH_TPPU_system.psf')

    # load the coordinates
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')

    # to use charmm forcefields get your parameters
    params = omma.CharmmParameterSet('all36_cgenff.rtf',
                                         'all36_cgenff.prm',
                                         'all36_prot.rtf',
                                         'all36_prot.prm',
                                         'tppu.str',
                                         'toppar_water_ions.str')

    # set the box size lengths and angles
    psf.setBox(82.435, 82.435, 82.435, 90, 90, 90)

        # create a system using the topology method giving it a topology and
        # the method for calculation
    system = psf.createSystem(params,
                                  nonbondedMethod=omma.CutoffPeriodic,
                                   nonbondedCutoff=1.0 * unit.nanometer,
                                  constraints=omma.HBonds)
    topology = psf.topology
        
    print("\nminimizing\n")
    # set up for a short simulation to minimize and prepare
    # instantiate an integrator
    integrator = omm.LangevinIntegrator(300*unit.kelvin,
                                            1/unit.picosecond,
                                            0.002*unit.picoseconds)
    # instantiate a OpenCL platform, in other platforms we can not have multiple simulation context 
    platform = omm.Platform.getPlatformByName('OpenCL')

     # instantiate a simulation object
    simulation = omma.Simulation(psf.topology, system, integrator,platform)
    # initialize the positions
    simulation.context.setPositions(pdb.openmm_positions(frame=0))
    # minimize the energy
    simulation.minimizeEnergy()
    # run the simulation for a number of initial time steps
    simulation.step(1000)
    print("done minimizing\n")

        # get the initial state from the context
    minimized_state = simulation.context.getState(getPositions=True,
                                                  getVelocities=True,
                                                  getParameters=True)
 
    # set up parameters for running the simulation
    num_workers = 8
    num_walkers = 8
    # initial weights
    init_weight = 1.0 / num_walkers
     # make a generator for the in itial walkers
    init_walkers = [OpenMMWalker(minimized_state, init_weight) for i in range(num_walkers)]
    make_table(init_walkers, num_workers,-1)

    # set up the OpenMMRunner with your system
    runner = OpenMMRunner(system, topology)

    # set up the Resampler (with parameters if needed)

    resampler = WExplore2Resampler(reference_traj=pdb,seed=123, pmax=0.2)

    # instantiate a hpcc object
    gpumapper  = GpuMapper(num_workers)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          num_workers,
                          runner=runner,
                          resampler=resampler,
                          work_mapper=scoop.futures.map)
    n_steps = 1000
    n_cycles = 100

    # run a simulation with the manager for 50 cycles of length 1000 each
    steps = [ n_steps for i in range(n_cycles)]
    walker_records, resampling_records = sim_manager.run_simulation(n_cycles,
                                                                    steps,
                                                                    debug_prints=True)
