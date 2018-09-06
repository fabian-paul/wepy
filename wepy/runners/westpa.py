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

    def run_segment(self, walker, segment_length, debug_prints=False):
        # segment_length is ignored for now
        parent_id = walker.state.parent_id
        id = walker.unique_running_index  # provided by the simulation managers (only the manager sees all walkers together)
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
        env['WEST_CURRENT_SEG_ID'] = '%d' % id
        env['WEST_PARENT_ID'] = '%d' % parent_id
        env['WEST_CURRENT_SEG_DATA_REF'] = '%s/traj_segs/%06d/%06d' % (root, iteration, id)

        env.update(walker.random_seeds)

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

        x = np.atleast_2d(pcoor)[-1, :]

        running_acv = walker.state.running_acv
        if running_acv is not None:
            running_acv.add(x)
        new_state = WestpaWalkerState(positions=x, iteration=iteration, parent_id=id, running_acv=running_acv)
        if debug_prints:
            print('walker #', id, 'with parent', parent_id, 'has weight', walker.weight)
        # Here we create a new walker with a new walker state.
        # If if the walker state contains history information.
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


class RunningAutoCovar(object):
    'Computes the mean, variance and time-lagged autocovariance of a time series'
    def __init__(self, n_lag=10, n_decay=None, min_frames=10):
        r"""Computes the mean, variance and time-lagged autocovariance of a time series

        :param n_lag: int
            lag time of
        :param n_decay: int or None
            parametrizes the speed with which past frames are forgotten
            If `n_decay` is None, use `2*n_lag`.
        :param min_frames: int
            only report statistical averages when at least n_lag + min_frames have been added
        """
        import collections
        self.n_lag = n_lag
        if n_decay is None:
            n_decay = 2*n_lag
        self.alpha = np.exp(-1.0/n_decay)  # alpha**n_decay = 1/e
        self.min_frames = min_frames
        self.deque = collections.deque(maxlen=n_lag)
        self.mean1 = None
        self.mean0 = None
        self.var0 = None
        self.var1 = None
        self.acv = None
        self.dim = None
        self.dtype = None
        self.n_frames_seen = 0

    def add(self, x1):
        r"""Add a frame to the estimation

        :param x1: np.ndarray(N)
            data point, 1-D
        """
        if self.dim is None:
            self.dim = x1.shape[0]
            self.dtype = x1.dtype

        self.deque.append(x1)

        # https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
        # https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis
        alpha = self.alpha
        if len(self.deque) >= self.n_lag:
            # update means and variances
            if self.mean1 is None:
                self.mean1 = x1
                self.var1 = np.zeros_like(x1)
            else:
                delta1 = x1 - self.mean1
                self.mean1 = self.mean1 + alpha*delta1
                self.var1 = (1. - alpha)*(self.var1 + alpha*delta1*delta1)

            x0 = self.deque[0]

            # update lagged variance
            if self.mean0 is None:
                self.mean0 = x0
                self.var0 = np.zeros_like(x0)
            else:
                delta0 = x0 - self.mean0
                self.mean0 = self.mean0 + alpha*delta0
                self.var0 = (1. - alpha)*(self.var0 + alpha*delta0*delta0)

            # update (time-lagged) auto-covariances
            if self.acv is None:
                self.acv = np.zeros_like(x0)
            else:
                self.acv = (1. - alpha)*(self.acv + alpha*delta0*delta1)

            self.n_frames_seen += 1

            #print('current acf', self.acf)


    @property
    def acf(self):
        r"""Compute the time-lagged autocorrlation. This is the autocorrelation function at lag time n_lag

        :return: np.ndarray(N)
            The time-lagged autocorrelation for each time series.
        """
        if self.n_frames_seen >= self.min_frames:
            return np.abs(self.acv) / (np.sqrt(self.var0)*np.sqrt(self.var1))
        else:
            return np.ones(self.dim, dtype=self.dtype)



class WestpaWalkerState(WalkerState):
    'WESTPA compatibility layer for wepy'

    def __init__(self, positions, iteration, parent_id=None, struct_data_ref=None, running_acv=None):
        self.positions = positions
        self.iteration = iteration
        self.parent_id = parent_id
        self.struct_data_ref = struct_data_ref
        self.running_acv = running_acv
        if 'WEST_SIM_ROOT' not in os.environ:
            raise RuntimeError('Environment variable WEST_SIM_ROOT not set.')
        self.west_sim_root = os.environ['WEST_SIM_ROOT']
        self._data = self.__dict__

    @property
    def coordinate_weights(self):
        if self.running_acv is None:
            return 1.0
        else:
            return self.running_acv.acf

    @classmethod
    def from_bstate(cls,
                    struct_data_ref='$WEST_SIM_ROOT/bstates/0',
                    get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh',
                    use_history=False):
        struct_data_ref = os.path.expandvars(struct_data_ref)
        p_coord = cls.get_pcoords(struct_data_ref=struct_data_ref, get_pcoord=get_pcoord)
        if use_history:
            running_acv = RunningAutoCovar()
        else:
            running_acv = None
        return cls(positions=np.atleast_2d(p_coord)[-1, :], iteration=0, parent_id=-1, struct_data_ref=struct_data_ref,
                   running_acv=running_acv)

    @classmethod
    def from_file(cls, iteration, id, path='$WEST_SIM_ROOT/traj_segs/%06d/%06d',
                  get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh',
                  use_history=False):
        path = os.path.expandvars(path % (iteration, id))
        p_coord = cls.get_pcoords(struct_data_ref=path, get_pcoord=get_pcoord)
        if use_history:
            running_acv = RunningAutoCovar()
        else:
            running_acv = None
        return cls(positions=np.atleast_2d(p_coord)[-1, :], iteration=iteration, parent_id=id, running_acv=running_acv)

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


def _get_dirs(folder):
    dirs = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_dir():
                dirs.append(entry.name)
    return dirs


def walkers_from_disk(n_expected_walkers=48, path='$WEST_SIM_ROOT/traj_segs/'):
    # find the last iteration that was completed
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

    weights = np.ones(n_expected_walkers, dtype=float) / n_expected_walkers  # TODO: recover the correct weights from restart file

    assert n_found_walkers[max_iteration] == n_expected_walkers
    walkers = [Walker(WestpaWalkerState.from_file(iteration=max_iteration, id=i), weight=weights[i]) for i in range(n_found_walkers[max_iteration])]

    print('continuing at iteration', max_iteration, 'with', n_found_walkers[max_iteration], 'walkers')

    return walkers


class WestpaReporter(Reporter):
    # TODO: implement westpa-compatible hdf writer

    def init(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def cleanup(self, *args, **kwargs):
        pass


class PairDistance(Distance):
    def image(self, state):
        return state['positions']

    def image_distance(self, image_a, image_b, coordinate_weights=None):
        dim = image_a.shape[0]
        if coordinate_weights is None:
            return np.linalg.norm(image_a - image_b) / dim**0.5
        else:
            return np.linalg.norm((image_a - image_b)*coordinate_weights) / (np.linalg.norm(coordinate_weights)*dim**0.5)

