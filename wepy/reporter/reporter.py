import os
import os.path as osp
import pickle

class Reporter(object):

    def __init__(self):
        pass

    def init(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def cleanup(self, *args, **kwargs):
        pass

class FileReporter(Reporter):

    def __init__(self, file_path, mode='x'):
        self.file_path = file_path
        self.mode = mode




class WalkersPickleReporter(Reporter):

    def __init__(self, save_dir='./', freq=100, num_backups=2):
        # the directory to save the pickles in
        self.save_dir = save_dir
        # the frequency of cycles to backup the walkers as a pickle
        self.backup_freq = freq
        # the number of sets of walker pickles to keep, this will keep
        # the last `num_backups`
        self.num_backups = num_backups

    def init(self, *args, **kwargs):
        # make sure the save_dir exists
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # delete backup pickles in the save_dir if they exist
        #else:
        #    for pkl_fname in os.listdir(self.save_dir):
        #        os.remove(osp.join(self.save_dir, pkl_fname))

    def report(self, cycle_idx, walkers,
               *args, **kwargs):

        # ignore all args and kwargs

        # total number of cycles completed
        print('current cycle is', cycle_idx)
        n_cycles = cycle_idx + 1
        # if the cycle is on the frequency backup walkers to a pickle
        if n_cycles % self.backup_freq == 0:

            pkl_name = "walkers_cycle_{}.pkl".format(cycle_idx)
            pkl_path = osp.join(self.save_dir, pkl_name)
            print('saving restart file with name', pkl_path)
            with open(pkl_path, 'wb') as wf:
                pickle.dump(walkers, wf)

            # remove old pickles if we have more than the `num_backups`
            if self.num_backups is not None:
                if (cycle_idx // self.backup_freq) >= self.num_backups:
                    old_idx = cycle_idx - self.num_backups * self.backup_freq
                    old_pkl_fname = "walkers_cycle_{}.pkl".format(old_idx)
                    print('deleting old restart file with name', old_pkl_fname)
                    try:
                        os.remove(osp.join(self.save_dir, old_pkl_fname))
                    except FileNotFoundError:
                        pass

    @staticmethod
    def get_most_recent_cycle(save_dir='./'):
        r'''Search for the restart file with the highest cycle in the directory `save_dir`

        :param
            save_dir: directory name
        :return: fname, cycle
            * fname : str
                file name of the restart file

            * cycle : int
                number of the cycle
        :raises
            Raises a `FileNotFoundError`, if not restart file was found.
        '''
        import re
        max_cycle = -1
        max_fname = None
        pattern = re.compile('walkers_cycle_([0-9]+).pkl')
        for fname in os.listdir(save_dir):
            match = pattern.match(fname)
            if match:
                cycle = match.group(1)
                if int(cycle) > max_cycle:
                    max_fname = fname
                    max_cycle = int(cycle)
        if max_cycle == -1:
            raise FileNotFoundError('No restart file was found.')
        return save_dir + os.sep + max_fname, max_cycle + 1

    @staticmethod
    def load_most_recent_cycle(save_dir='./', debug_prints=False):
        restart_fname, cycle = WalkersPickleReporter.get_most_recent_cycle(save_dir)
        import pickle

        if debug_prints:
            print('loading walker restart file', restart_fname)
        with open(restart_fname, 'rb') as f:
            init_walkers = pickle.load(f)
        return init_walkers, cycle


