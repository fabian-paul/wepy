import numpy as np
import subprocess
import os
import tempfile
import copy
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
        iteration = walker.state.iteration + 1
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
        x = np.atleast_2d(pcoor)[-1, :]

        running_acv = walker.state.running_acv
        if running_acv is not None:
            running_acv = running_acv + x
        new_state = WestpaWalkerState(positions=x, iteration=iteration, parent_id=parent_id,
                                      id=propagation_id, running_acv=running_acv)
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
    'Computes the mean, variance and time-lagged autocovariance of a time series. This object is immutable.'

    def __init__(self, n_lag=50, n_decay=None, min_frames=10,
                 mean0=None, mean1=None, var0=None, var1=None, acv=None,
                 dim=None, dtype=None, n_frames_seen=0, deque=None):
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
            self.n_decay = 2*n_lag
        else:
            self.n_decay = n_decay
        self.min_frames = min_frames
        self.mean1 = mean1
        self.mean0 = mean0
        self.var0 = var0
        self.var1 = var1
        self.acv = acv
        self.dim = dim
        self.dtype = dtype
        self.n_frames_seen = n_frames_seen
        if deque is None:
            self.deque = collections.deque(maxlen=n_lag)
        else:
            self.deque = deque

    @classmethod
    def new_with_prior(cls, x, sigma=1.0, n_lag=50, n_decay=None):
        s = sigma*np.ones_like(x)
        rac = cls(n_lag=n_lag, n_decay=n_decay, min_frames=0, mean0=x, mean1=x, var0=s,
                  var1=s, acv=s, dim=x.shape[-1], dtype=x.dtype)
        return rac

    def __add__(self, x1):
        r"""Add a frame to the estimation

        :param x1: np.ndarray(N)
            data point, 1-D
        """
        if self.dim is None:
            dim = x1.shape[0]
            dtype = x1.dtype
        else:
            dim = self.dim
            dtype = self.dtype

        deque = copy.copy(self.deque)
        deque.append(x1)

        # https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
        # https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis
        # https://vlab.stern.nyu.edu/doc/12?topic=mdls
        alpha = 1. - np.exp(-1./self.n_decay)  # (1 - alpha)**n_decay = 1/e
        if len(deque) >= self.n_lag:
            # update means and variances
            if self.mean1 is None:
                mean1 = x1
                mean1.flags.writeable = False  # debug
                var1 = np.zeros_like(x1)
                var1.flags.writeable = False  # debug
            else:
                delta1 = x1 - self.mean1
                mean1 = self.mean1 + alpha*delta1
                var1 = (1. - alpha)*(self.var1 + alpha*delta1*delta1)

            x0 = deque[0]

            # update lagged variance
            if self.mean0 is None:
                mean0 = x0
                mean0.flags.writeable = False  # debug
                var0 = np.zeros_like(x0)
                var0.flags.writeable = False  # debug
            else:
                delta0 = x0 - self.mean0
                mean0 = self.mean0 + alpha*delta0
                var0 = (1. - alpha)*(self.var0 + alpha*delta0*delta0)

            # update (time-lagged) auto-covariances
            if self.acv is None:
                acv = np.zeros_like(x0)
                acv.flags.writeable = False  # debug
            else:
                acv = (1. - alpha)*(self.acv + alpha*delta0*delta1)

            n_frames_seen = self.n_frames_seen + 1
        else:
            mean0 = self.mean0
            mean1 = self.mean1
            var0 = self.var0
            var1 = self.var1
            acv = self.acv
            n_frames_seen = 0

        # print('updated RunningAutoCovar')

        return self.__class__(n_lag=self.n_lag, n_decay=self.n_decay, min_frames=self.min_frames,
                              mean0=mean0, mean1=mean1, var0=var0, var1=var1, acv=acv,
                              dim=dim, dtype=dtype, n_frames_seen=n_frames_seen, deque=deque)

    @property
    def acf(self):
        r"""Compute the time-lagged autocorrlation. This is the autocorrelation function at lag time n_lag

        :return: np.ndarray(N)
            The time-lagged autocorrelation for each time series.
        """
        if self.n_frames_seen >= self.min_frames:
            return np.abs(self.acv) / (np.sqrt(self.var0*self.var1))
        else:
            return np.ones(self.dim, dtype=self.dtype)

    @staticmethod
    def mean_acf(racs):
        if racs[0].n_frames_seen < racs[0].min_frames:
            print('returning ones')
            return np.ones(racs[0].dim, dtype=racs[0].dtype)
        else:
            print('actually computing something, var0 =', racs[0].var0, 'from', len(racs[0].deque))
            acv = np.sum([r.acv for r in racs], axis=0)
            var0 = np.sum([r.var0 for r in racs], axis=0)
            var1 = np.sum([r.var1 for r in racs], axis=0)
            return np.abs(acv) / (np.sqrt(var0*var1))

    @staticmethod
    def mean_std(racs):
        if racs[0].n_frames_seen < racs[0].min_frames:
            print('returning ones')
            return np.ones(racs[0].dim, dtype=racs[0].dtype)
        else:
            print('actually computing something, var0 =', racs[0].var0, 'from', len(racs[0].deque))
            var0 = np.mean([r.var0 for r in racs], axis=0)
            return np.sqrt(var0)

class WestpaWalkerState(WalkerState):
    'WESTPA compatibility layer for wepy'

    def __init__(self, positions, iteration, id, parent_id=None, struct_data_ref=None, running_acv=None):
        self.positions = positions
        self.iteration = iteration
        self.id = id
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
            running_acv = RunningAutoCovar.new_with_prior(x=p_coord, sigma=0.3)
        else:
            running_acv = None
        return cls(positions=p_coord, iteration=0, id=-1, struct_data_ref=struct_data_ref,
                   running_acv=running_acv)

    @classmethod
    def from_file(cls, iteration, id, path='$WEST_SIM_ROOT/traj_segs/%06d/%06d',
                  get_pcoord='$WEST_SIM_ROOT/westpa_scripts/get_pcoord.sh',
                  use_history=False):
        path = os.path.expandvars(path % (iteration, id))
        p_coord = cls.get_pcoords(struct_data_ref=path, get_pcoord=get_pcoord)
        if use_history:
            running_acv = RunningAutoCovar.new_with_prior(x=p_coord, sigma=0.3)
        else:
            running_acv = None
        return cls(positions=p_coord, iteration=iteration, id=id, running_acv=running_acv)

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

    def ensemble_weight(self, states):
        if self.running_acv is not None:
            return 1.0/RunningAutoCovar.mean_std([s.running_acv for s in states])
        else:
            return np.ones_like(self.positions)


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
    walkers = [Walker(WestpaWalkerState.from_file(iteration=max_iteration, id=i), weight=weights[i]) for i in
               range(n_found_walkers[max_iteration])]
    print('continuing at iteration', max_iteration, 'with', n_found_walkers[max_iteration],
          'walkers and with discarded weights')
    return walkers, max_iteration


class WestpaReporter(Reporter):
    # minimal reporter

    def __init__(self, n_walkers, hdf5_fname='we.hdf5'):
        import h5py
        self.hdf5 = h5py.File(hdf5_fname, mode='a')
        if 'tree' not in self.hdf5:
            self.hdf5.create_dataset('tree', (128, n_walkers,), maxshape=(None, n_walkers), dtype='i8')
        self.tree = self.hdf5['tree']
        if 'weights' not in self.hdf5:
            self.hdf5.create_dataset('weights', (128, n_walkers,), maxshape=(None, n_walkers), dtype='float64')
        self.weights = self.hdf5['weights']
        self.n_walkers = n_walkers

    def report(self, cycle_idx, new_walkers, warp_data, bc_data, progress_data,
               resampling_data, resampler_data, *args, **kwargs):

        assert len(new_walkers) == self.n_walkers == len(kwargs['resampled_walkers'])
        walkers = kwargs['resampled_walkers']

        weights_size = self.weights.shape[0]
        if weights_size <= cycle_idx:
            self.weights.resize((weights_size + 128, self.weights.shape[1]))
        tree_size = self.tree.shape[0]
        if self.tree.shape[0] <= cycle_idx:
            self.tree.resize((tree_size + 128, self.tree.shape[1]))

        walker_weights = [w.weight for w in walkers]
        current_walker_ids = [w.state['id'] for w in walkers]
        parent_walker_ids = [w.state['parent_id'] for w in walkers]

        current_weigths = np.zeros(self.n_walkers, dtype='float64')
        current_weigths[current_walker_ids] = walker_weights
        self.weights[cycle_idx, :] = current_weigths

        current_level = np.zeros(self.n_walkers, dtype='i8')
        current_level[current_walker_ids] = parent_walker_ids
        self.tree[cycle_idx, :] = current_level

        self.hdf5.flush()

    def cleanup(self, *args, **kwargs):
        self.hdf5.close()


class PairDistance(Distance):
    def image(self, state):
        return state['positions']

    def image_distance(self, image_a, image_b, coordinate_weights=None):
        dim = image_a.shape[0]//3
        if coordinate_weights is None:
            return np.linalg.norm(image_a - image_b) / dim**0.5
        else:
            return np.linalg.norm((image_a - image_b)*coordinate_weights) / (np.linalg.norm(coordinate_weights)*dim**0.5)


class WestpaUnbindingBC(object):

    def __init__(self, initial_state, cutoff_distance=1.0):

        super().__init__()
        self.initial_state = initial_state
        self.cutoff_distance = cutoff_distance

    def _calc_min_distance(self, walker):
        dim = len(walker.state['positions'])//3
        return np.linalg.norm(walker.state['positions'])*(dim**-0.5)

    def progress(self, walker):
        # test to see if the ligand is unbound
        return self._calc_min_distance(walker) >= self.cutoff_distance

    def warp(self, walker):
        # set the initial state into a new walker object with the same weight
        return type(walker)(state=copy.deepcopy(self.initial_state), weight=walker.weight)

    def warp_walkers(self, walkers, cycle, debug_prints=False):
        new_walkers = []
        for walker in walkers:
            unbound = self.progress(walker)
            # if the walker is unbound we need to warp it
            if unbound:
                print('wrapping walker')
                new_walkers.append(self.warp(walker))
            # no warping so just return the original walker
            else:
                new_walkers.append(walker)

        return new_walkers, None, None, None


