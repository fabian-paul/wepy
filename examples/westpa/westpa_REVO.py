## Wepy classes

# the simulation manager and work mapper for actually running the simulation
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper, Worker

# the runner for running dynamics and making and it's particular state class
from wepy.runners.westpa import WestpaRunner, WestpaWalkerState, PairDistance, WestpaUnbindingBC
from wepy.walker import Walker

# classes for making the resampler
from wepy.resampling.resamplers.revo import REVOResampler

from wepy.boundary_conditions.boundary import NoBC

# standard reporters
from wepy.reporter.reporter import WalkersPickleReporter

from copy import deepcopy


def main(n_walkers=36, n_workers=12, n_runs=1, n_cycles=20, n_steps=100, continue_sim=False):
    runner = WestpaRunner()

    init_state = WestpaWalkerState.from_bstate(struct_data_ref='$WEST_SIM_ROOT/bstates/0')

    work_mapper = WorkerMapper(worker_type=Worker, num_workers=n_workers)

    if continue_sim:
        # init_walkers = walkers_from_disk(n_expected_walkers=n_walkers)
        init_walkers, start_cycle = WalkersPickleReporter.load_most_recent_cycle(debug_prints=True)
    else:
        start_cycle = 0
        init_weight = 1.0 / n_walkers
        init_walkers = [Walker(deepcopy(init_state), init_weight) for i in range(n_walkers)]


    unb_distance = PairDistance()

    resampler = REVOResampler(distance=unb_distance, init_state=init_state)

    reporters = [WalkersPickleReporter(freq=10)]

    boundary_conditions = WestpaUnbindingBC(init_state, 1.0)

    # Instantiate a simulation manager
    sim_manager = Manager(init_walkers,
                          runner=runner,
                          resampler=resampler,
                          boundary_conditions=boundary_conditions,
                          work_mapper=work_mapper,
                          reporters=reporters)

    segment_lengths = [n_steps for i in range(start_cycle + n_cycles)]

    ### RUN the simulation
    for run_idx in range(n_runs):
        print("Starting run: {}".format(run_idx))
        sim_manager.continue_run_simulation(0, n_cycles, segment_lengths, num_workers=n_workers,
                                            start_cycle=start_cycle)#, debug_prints=True)
        print("Finished run: {}".format(run_idx))


if __name__=='__main__':
    import os
    import sys
    if 'WEST_SIM_ROOT' not in os.environ:
        os.environ['WEST_SIM_ROOT'] = os.getcwd()
    main(continue_sim=len(sys.argv) > 1 and sys.argv[1] == '--continue')
