import numpy as np
import subprocess
import os
import tempfile
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.reporter.reporter import Reporter
from wepy.resampling.distances.distance import Distance


def randint(dtype):
    return np.random.randint(np.iinfo(dtype).max, dtype=dtype)


class WestpaRunner(Runner):
    'WESTPA compatibility layer for wepy'

    # TODO: use westpa configuration file for the name of executables
    def __init__(self,
                 runseg='$WEST_SIM_ROOT/westpa_scripts/runseg.sh',
                 get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh',
                 post_iter='$WEST_SIM_ROOT/westpa_scripts/post_iter.sh'):
        self.runseg = runseg
        self.get_pcoord = get_pcoord
        self.post_iter = post_iter
        if 'WEST_SIM_ROOT' not in os.environ:
            raise RuntimeError('Environment variable WEST_SIM_ROOT not set.')
        self.west_sim_root = os.environ['WEST_SIM_ROOT']

    def random_numbers(self):
        rv = {}
        rv['WEST_RAND16'] = '%d' % randint(np.uint16)
        rv['WEST_RAND32'] = '%d' % randint(np.uint32)
        rv['WEST_RAND64'] = '%d' % randint(np.uint64)
        rv['WEST_RAND128'] = '%d' % (randint(np.uint64) + (np.iinfo(np.uint64).max + 1) * randint(np.uint64))
        rv['WEST_RANDFLOAT'] = '%f' % np.random.rand(1)
        return rv

    def run_segment(self, walker, segment_length, propagation_id, random_seeds, debug_prints=False):
        # segment_length is ignored for now
        parent_id = walker.state.id
        iteration = walker.state['iteration'] + 1  # TODO: increment here or later?
        root = self.west_sim_root

        env = dict(os.environ)
        if parent_id == -1:  # start new trajectory
            assert walker.state.struct_data_ref is not None
            env['WEST_PARENT_DATA_REF'] = walker.state.struct_data_ref
            env['WEST_CURRENT_SEG_INITPOINT_TYPE'] = 'SEG_INITPOINT_NEWTRAJ'
        else:  # continue trajectory
            env['WEST_PARENT_DATA_REF'] = '%s/traj_segs/%06d/%06d' % (root, iteration - 1, parent_id)
            env['WEST_CURRENT_SEG_INITPOINT_TYPE'] = 'SEG_INITPOINT_CONTINUES'

        env['WEST_CURRENT_ITER'] = '%d' % iteration
        env['WEST_CURRENT_SEG_ID'] = '%d' % propagation_id
        env['WEST_PARENT_ID'] = '%d' % parent_id
        env['WEST_CURRENT_SEG_DATA_REF'] = '%s/traj_segs/%06d/%06d' % (root, iteration, propagation_id)

        env.update(random_seeds)

        pcoor_file = tempfile.NamedTemporaryFile(mode='r')
        env['WEST_PCOORD_RETURN'] = pcoor_file.name
        env['WEST_COORD_RETURN'] = '/dev/null'
        env['SEG_DEBUG'] = ''

        erl = subprocess.call(os.path.expandvars(self.runseg), shell=True, env=env)
        if erl != 0:
            raise RuntimeError('segment propagation failed')

        # creates new_walker from new state and current weight
        pcoor = np.loadtxt(pcoor_file)
        pcoor_file.close()
        new_state = WestpaWalkerState(positions=np.atleast_2d(pcoor)[-1, :], iteration=iteration, parent_id=parent_id, id=propagation_id)
        if debug_prints:
            print('walker #', id, 'with parent', parent_id, 'has weight', walker.weight)
        new_walker = Walker(state=new_state, weight=walker.weight)

        return new_walker

    def run_post_iter(self, walkers):
        post_iter = os.path.expandvars(self.post_iter)
        if os.path.exists(post_iter):
            iteration = walkers[0].state['iteration']

            env = dict(os.environ)

            env['SEG_DEBUG'] = ''
            env['WEST_CURRENT_ITER'] = '%d' % iteration

            erl = subprocess.call(post_iter, shell=True, env=env)
            if erl != 0:
                raise RuntimeError('running post-iteration script failed')



class WestpaWalkerState(WalkerState):
    'WESTPA compatibility layer for wepy'

    def __init__(self, positions, iteration, id, parent_id=None, struct_data_ref=None):
        self.positions = positions
        self.iteration = iteration
        self.id = id
        self.parent_id = parent_id
        self.struct_data_ref = struct_data_ref
        if 'WEST_SIM_ROOT' not in os.environ:
            raise RuntimeError('Environment variable WEST_SIM_ROOT not set.')
        self.west_sim_root = os.environ['WEST_SIM_ROOT']
        self._data = self.__dict__

    @classmethod
    def from_bstate(cls,
                    struct_data_ref='$WEST_SIM_ROOT/bstates/0',
                    get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh'):
        struct_data_ref = os.path.expandvars(struct_data_ref)
        p_coord = cls.get_pcoords(struct_data_ref=struct_data_ref, get_pcoord=get_pcoord)
        return cls(positions=np.atleast_2d(p_coord)[-1, :], iteration=0, id=-1, struct_data_ref=struct_data_ref)

    @staticmethod
    def get_pcoords(struct_data_ref, get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh'):
        env = dict(os.environ)

        pcoor_file = tempfile.NamedTemporaryFile(mode='r')
        env['WEST_PCOORD_RETURN'] = pcoor_file.name
        env['WEST_STRUCT_DATA_REF'] = struct_data_ref
        env['SEG_DEBUG'] = ''

        erl = subprocess.call(os.path.expandvars(get_pcoord), shell=True, env=env)
        if erl != 0:
            raise RuntimeError('getting progress coordinates failed')

        pcoor = np.loadtxt(pcoor_file)
        pcoor_file.close()
        return np.atleast_2d(pcoor)[-1, :]

    @classmethod
    def from_file(cls, iteration, id, path='$WEST_SIM_ROOT/traj_segs/%06d/%06d',
                  get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh'):
        path = os.path.expandvars(path % (iteration, id))
        p_coord = cls.get_pcoords(struct_data_ref=path, get_pcoord=get_pcoord)
        return cls(positions=np.atleast_2d(p_coord)[-1, :], iteration=iteration, id=id)


def _get_dirs(folder):
    dirs = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_dir():
                dirs.append(entry.name)
    return dirs


def walkers_from_disk(n_expected_walkers=48, path='$WEST_SIM_ROOT/traj_segs/'):
    # find the last iteration that was completed; this is for emergency cases and other hacks
    path = os.path.expandvars(path)
    max_iteration = -1
    n_found_walkers = {}
    for topdir in [d for d in _get_dirs(path) if d.isdigit()]:
        iteration = int(topdir)
        subdirs = [d for d in _get_dirs(path + topdir) if d.isdigit()]
        if len(set([int(d) for d in subdirs]) & set(range(n_expected_walkers))) == n_expected_walkers:
            max_iteration = max(max_iteration, iteration)
            n_found_walkers[iteration] = len(set([int(d) for d in subdirs]))
    if max_iteration == -1:
        raise RuntimeError('no valid iteration found')
    weights = np.ones(n_expected_walkers, dtype=float) / n_expected_walkers
    assert n_found_walkers[max_iteration] == n_expected_walkers
    walkers = [Walker(WestpaWalkerState.from_file(iteration=max_iteration, id=i), weight=weights[i]) for i in range(n_found_walkers[max_iteration])]
    print('continuing at iteration', max_iteration, 'with', n_found_walkers[max_iteration], 'walkers and with discarded weights')
    return walkers, max_iteration


class WestpaReporter(Reporter):
    # minimal reporter

    def init(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def cleanup(self, *args, **kwargs):
        pass


class PairDistance(Distance):
    def image(self, state):
        return state['positions']

    def image_distance(self, image_a, image_b):
        dim = image_a.shape[0]
        dist = np.linalg.norm(image_a - image_b) / dim**0.5
        return dist
