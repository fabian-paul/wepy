## Wepy classes

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper, Worker

# the runner for running dynamics and making and it's particular state class
from wepy.runners.westpa import WestpaRunner, WestpaWalkerState, WestpaReporter, PairDistance, walkers_from_disk
from wepy.walker import Walker

# classes for making the resampler
from wepy.resampling.distances.receptor import UnbindingDistance
from wepy.resampling.resamplers.revo import REVOResampler

from wepy.boundary_conditions.boundary import NoBC

# standard reporters
#from wepy.reporter.hdf5 import WepyHDF5Reporter


def main(n_walkers=36, n_workers=12, n_runs=1, n_cycles=20, n_steps=100, continue_sim=False):
    runner = WestpaRunner()

    init_state = WestpaWalkerState.from_bstate(struct_data_ref='$WEST_SIM_ROOT/bstates/0')

    work_mapper = WorkerMapper(worker_type=Worker, num_workers=n_workers)

    init_weight = 1.0 / n_walkers

    if continue_sim:
        init_walkers = [Walker(init_state, init_weight) for i in range(n_walkers)]
    else:
        init_walkers = walkers_from_disk(n_expected_walkers=n_walkers)

    unb_distance = PairDistance()

    resampler = REVOResampler(distance=unb_distance, init_state=init_state, dpower=4)

    reporter = WestpaReporter()

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=NoBC(),
                          work_mapper=work_mapper,
                          reporters=[reporter])

    steps = [n_steps for i in range(n_cycles)]

    ### RUN the simulation
    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.run_simulation(n_cycles, steps)#, debug_prints=True)
        print("Finished run: {}".format(run_idx))


if __name__=='__main__':
    import os
    if not 'WEST_SIM_ROOT' in os.environ:
        os.environ['WEST_SIM_ROOT'] = os.getcwd()
    main()