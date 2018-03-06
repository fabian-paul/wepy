import numpy as np

from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd

from wepy.resampling.distances.distance import Distance

class UnbindingDistance(Distance):
    def __init__(self, ligand_idxs, binding_site_idxs):

        # the idxs of the ligand and binding site from the whole state
        self._lig_idxs = ligand_idxs
        self._bs_idxs = binding_site_idxs

        # number of atoms in each
        self._n_lig_atoms = len(self._lig_idxs)
        self._n_bs_atoms = len(self._bs_idxs)

        # the idxs used for the whole image
        self._image_idxs = np.concatenate( (self._lig_idxs, self._bs_idxs) )

        # the idxs of the ligand and binding site within the image
        self._image_lig_idxs = np.arange(self._n_lig_atoms)
        self._image_bs_idxs = np.arange(self._n_lig_atoms, self._n_lig_atoms + self._n_bs_atoms)

    def image(self, state):

        state_image = state['positions'][self._image_idxs]

        return state_image

    def image_distance(self, image_a, image_b):

        # first we superimpose the two structures according to the
        # binding site, and by translating and rotating only one of
        # the images, image_b by convention.
        sup_image_b = superimpose(image_a, image_b, idxs=self._image_bs_idxs)


        # then we calculate the rmsd of only the ligands between the
        # images
        lig_rmsd = calc_rmsd(image_a, sup_image_b, idxs=self._image_lig_idxs)

        return lig_rmsd
