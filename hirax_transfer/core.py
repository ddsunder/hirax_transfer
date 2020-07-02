from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import astropy.units as units
import astropy.coordinates as coords

from caput import config
from drift.core import visibility
from drift.core.telescope import SimplePolarisedTelescope
from drift.core.telescope import _remap_keyarray, _merge_keyarray

from .beams import fetch_beam, fetch_taper
from .layouts import fetch_layout

# Based on DishArray
class HIRAXSinglePointing(SimplePolarisedTelescope):

    freq_lower = config.Property(proptype=float, default=400.0)
    freq_upper = config.Property(proptype=float, default=800.0)
    num_freq = config.Property(proptype=int, default=10)

    tsys_flat = config.Property(proptype=float, default=50.0)
    ndays = config.Property(proptype=int, default=733)
    redundancy_boost = config.Property(proptype=float, default=1.0)

    accuracy_boost = config.Property(proptype=float, default=1.0)
    l_boost = config.Property(proptype=float, default=1.0)
    lmax = config.Property(proptype=int, default=200)
    mmax = config.Property(proptype=int, default=200)

    minlength = config.Property(proptype=float, default=0.0)
    maxlength = config.Property(proptype=float, default=1.e7)

    auto_correlations = config.Property(proptype=bool, default=False)

    local_origin = config.Property(proptype=bool, default=True)

    dish_width = config.Property(proptype=float, default=6.)

    # Fixed at lat=-30deg. To be set with site lat and lon params in future
    zenith = np.array([np.radians(120), 0.])

    def read_config(self, conf):
        beam_conf = conf.pop('hirax_beam')
        self.hirax_beam = fetch_beam(beam_conf)
        self.beam_taper = fetch_taper(beam_conf)
        layout_conf = conf.pop('hirax_layout')
        self.hirax_layout = fetch_layout(layout_conf)
        super(HIRAXSinglePointing, self).read_config(conf)

    # Give the widths in the U and V directions in metres (used for
    # calculating the maximum l and m). This assumes always compact spacing,
    # so should be made more general at some stage.
    @property
    def u_width(self):
        return self.dish_width

    @property
    def v_width(self):
        return self.dish_width

    def beam(self, feed, freq, ignore_taper=False):
        if self.polarisation[feed] == "X":
            return self.beamx(feed, freq, ignore_taper=ignore_taper)
        else:
            return self.beamy(feed, freq, ignore_taper=ignore_taper)

    def beamx(self, feed, freq, pointing=None, ignore_taper=False):
        if pointing is None:
            pointing = self.zenith
        beam = self.hirax_beam(self._angpos, pointing, self.wavelengths[freq], feed, 0)
        # Assume non-vector beam has perfect polarisation separation
        if beam.ndim < 2:
            beam = beam[:, np.newaxis] * np.array([0.0, 1.0])
        if (self.beam_taper is not None) and not ignore_taper:
            beam[:, 0] *= self.beam_taper(
                self._angpos, pointing, self.wavelengths[freq])
        return beam

    def beamy(self, feed, freq, pointing=None, ignore_taper=False):
        if pointing is None:
            pointing = self.zenith
        beam = self.hirax_beam(self._angpos, pointing, self.wavelengths[freq], feed, 1)
        # Assume non-vector beam has perfect polarisation separation
        if beam.ndim < 2:
            beam = beam[:, np.newaxis] * np.array([1.0, 0.0])
        if (self.beam_taper is not None) and not ignore_taper:
            beam[:, 1] *= self.beam_taper(
                self._angpos, pointing, self.wavelengths[freq])
        return beam

    @property
    def _single_feedpositions(self):
        return self.hirax_layout()

    # Overloading to change solid angle calc.
    def _beam_map_single(self, bl_index, f_index):

        p_stokes = [
            0.5 * np.array([[1.0, 0.0], [0.0, 1.0]]),
            0.5 * np.array([[1.0, 0.0], [0.0, -1.0]]),
            0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]),
            0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]]),
        ]

        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self.beam(feedi, f_index), self.beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        pow_stokes = [
            np.sum(beami * np.dot(beamj.conjugate(), polproj), axis=1) * self._horizon
            for polproj in p_stokes
        ]

        # Calculate the solid angle of each beam
        pxarea = 4 * np.pi / beami.shape[0]
        # Ensure sa calculated assuming nominal beam so main lobe is identical
        beami_for_sa = self.beam(feedi, f_index, ignore_taper=True)
        beamj_for_sa = self.beam(feedj, f_index, ignore_taper=True)

        om_i = np.sum(np.abs(beami_for_sa) ** 2 * self._horizon[:, np.newaxis]) * pxarea
        om_j = np.sum(np.abs(beamj_for_sa) ** 2 * self._horizon[:, np.newaxis]) * pxarea

        omega_A = (om_i * om_j) ** 0.5

        # Calculate the complex visibility transfer function
        cv_stokes = [p * (2 * fringe / omega_A) for p in pow_stokes]

        return cv_stokes


# @copy_reader_properties(HIRAXSinglePointing)
class HIRAXSurvey(HIRAXSinglePointing):

    pointing_start = config.Property(proptype=float, default=0)
    pointing_stop = config.Property(proptype=float, default=0)
    npointings = config.Property(proptype=int, default=1)

    @property
    def pointings(self):
        return np.linspace(self.pointing_start,
                           self.pointing_stop,
                           self.npointings)

    @property
    def pointing_feedmap(self):
        return np.repeat(np.arange(len(self.pointings)),
                         self.single_pointing_telescope.nfeed)

    def read_config(self, conf):
        nconf = conf.copy()
        nconf.update(conf['hirax_spec'])
        super(HIRAXSurvey, self).read_config(nconf)
        # Make sure this also gets sorted for no-config initialisation...
        self.single_pointing_telescope = HIRAXSinglePointing.from_config(conf['hirax_spec'])

    def _calculate_feedpairs(self):
        self.single_pointing_telescope.calculate_feedpairs()
        super(HIRAXSurvey, self).calculate_feedpairs()

    def _init_trans(self, nside):
        super(HIRAXSurvey, self)._init_trans(nside)
        self.single_pointing_telescope._init_trans(nside)

    @property
    def u_width(self):
        return self.single_pointing_telescope.u_width

    @property
    def v_width(self):
        return self.single_pointing_telescope.v_width

    @property
    def redundancy(self):
        return self.redundancy_boost*super(HIRAXSinglePointing, self).redundancy

    @property
    def _single_feedpositions(self):
        return np.tile(self.single_pointing_telescope._single_feedpositions,
                       (len(self.pointings), 1))

    def _unique_baselines(self):
        """
        Ensures baselines from different pointings are treated as unique
        """
        fmap, mask = self.single_pointing_telescope._unique_baselines()

        block_fmap = linalg.block_diag(*[fmap+i*self.single_pointing_telescope.nfeed for i, _ in enumerate(self.pointings)])
        block_mask = linalg.block_diag(*[mask for _ in self.pointings])

        return _remap_keyarray(block_fmap, block_mask), block_mask

    def _unique_beams(self):
        """
        Ensures beams from different pointings are treated as unique
        """
        bmap, mask = self.single_pointing_telescope._unique_beams()
        block_bmap = linalg.block_diag(*[bmap+i*self.single_pointing_telescope.nfeed for i, _ in enumerate(self.pointings)])
        block_mask = linalg.block_diag(*[mask for _ in self.pointings])

        return block_bmap, block_mask

    @property
    def beamclass(self):
        return np.tile(self.single_pointing_telescope.beamclass, len(self.pointings))

    def beamx(self, feed, freq, ignore_taper=False):
        ddec = np.radians(self.pointings[self.pointing_feedmap[feed]])
        pointing_vector = np.array([
            self.single_pointing_telescope.zenith[0] - ddec, # negative for healpix convention
            self.single_pointing_telescope.zenith[1]])
        return self.single_pointing_telescope.beamx(feed, freq, pointing=pointing_vector,
                                                    ignore_taper=ignore_taper)

    def beamy(self, feed, freq, ignore_taper=False):
        ddec = np.radians(self.pointings[self.pointing_feedmap[feed]])
        pointing_vector = np.array([
            self.single_pointing_telescope.zenith[0] - ddec, # negative for healpix convention
            self.single_pointing_telescope.zenith[1]])
        return self.single_pointing_telescope.beamy(feed, freq, pointing=pointing_vector,
                                                    ignore_taper=ignore_taper)

    """
    Visualisation helpers,
    need to add frequency index
    """

    def fish_ell_m(self, mi, fi=0, baselines=None, pointing_ind=None,
                   bt=None, pol_index=0, which_m='full'):

        bl_keep = np.ones(self.npairs, dtype=bool)

        if baselines is not None:
            to_keep = [(self.baselines[:, 0] == bl[0]) & (self.baselines[:, 1] == bl[1])
                       for bl in baselines]
            to_keep = np.any(to_keep, axis=0)
            bl_keep &= to_keep

        if pointing_ind is not None:
            mid_pointing = self.pointing_feedmap==pointing_ind
            mid_pointing_feedmap = mid_pointing[None, :] * mid_pointing[:, None]
            pointing_baselines = np.zeros(self.npairs, dtype=bool)
            pointing_baselines[np.unique(self.feedmap[mid_pointing_feedmap])] = True
            bl_keep &= pointing_baselines

        if bt is None:
            # This is super inefficient (recalculates for all m's for each call...)
            bi = np.arange(self.npairs)[bl_keep]
            transfer_mats = self.transfer_matrices(bi, fi)[0, ..., np.abs(mi)]
            B_posm = transfer_mats
            B_negm = (-1)**np.abs(mi)*transfer_mats[..., -np.abs(mi)].conj()
        else:
            # Rather feed in a BeamTransfer from a cached products run
            B = bt.beam_m(np.abs(mi), fi=fi)[:, :, pol_index, :].copy()
            B_posm = B[0, bl_keep, ...]
            B_negm = B[1, bl_keep, ...]

        B_posm = np.abs(B_posm)
        B_negm = np.abs(B_negm)
        Ninv = np.diag(1/self.noisepower(np.arange(self.npairs)[bl_keep], 0)) # In K

        if which_m.lower() == 'full':
            out = (B_posm.T.dot(Ninv)).dot(B_posm) + (B_negm.T.dot(Ninv)).dot(B_negm)
        elif which_m.lower() == 'negative':
            out = (B_negm.T.dot(Ninv)).dot(B_negm)
        elif which_m.lower() == 'positive':
            out = (B_posm.T.dot(Ninv)).dot(B_posm)

        return out / 1.e12 # Put in uK

    def sensitivity_alm_plot(self, fi=0, baselines=None, pointing_ind=None,
                             ax=None, bt=None,
                             positive_m_only=True, vmin=None, vmax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if positive_m_only:
            alm_sens = np.zeros((self.lmax+1, self.mmax))
        else:
            alm_sens = np.zeros((self.lmax+1, 2*self.mmax+1))
        mrange = range(0 if positive_m_only else -self.mmax, self.mmax)
        for i, mi in enumerate(mrange):
            alm_sens[:, i] = 1/np.diag(np.abs(
                self.fish_ell_m(
                    mi, fi, baselines=baselines,
                    pointing_ind=pointing_ind, bt=bt)))
        to_plot = alm_sens**0.5
        to_plot[to_plot == 0] = np.NaN
        to_plot = np.log10(to_plot)
        im = ax.imshow(to_plot, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       extent=(0 if positive_m_only else -self.mmax,
                               self.mmax, 0, self.lmax))
        ax.set_ylabel('$\ell$')
        ax.set_xlabel('$m$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r'$\log_{10}({\rm diag}(\mathcal{F}^{-1/2}) / {\rm \mu~K})$')

        #    ax.set_aspect(0.75)
        if pointing_ind is None:
            ax.set_title('All pointings')
        else:
            ax.set_title('Pointing {:.0f}$^{{\circ}}$'.format(self.pointings[pointing_ind]))

        return ax.figure

    def sensitivity_ell(self, fi=0, baselines=None, pointing_ind=None, bt=None):
        n = self.lmax + 1
        full_fish = np.zeros((n, n), dtype=np.complex128)
        for mi in range(self.mmax):
            fish = self.fish_ell_m(
                mi, fi=fi, baselines=baselines,
                pointing_ind=pointing_ind, bt=bt)
            full_fish += fish
        return 1/np.diag(np.abs(full_fish))

    def sensitivity_cl_plot(self, fi=0, pointing_ind=None, bt=None):

        fig, ax = plt.subplots()

        all_bl_sens = self.sensitivity_ell(fi=fi, pointing_ind=pointing_ind, bt=bt)
        ax.plot(all_bl_sens, label='All baselines')

        for bl in np.unique(self.baselines, axis=0):
            ls = '-' if bl[0] > bl[1] else '--'
            bl_sens = self.sensitivity_ell(
                fi=fi, baselines=[bl],
                pointing_ind=pointing_ind, bt=bt)
            lab = '{:.0f}m; {:.0f}m'.format(*bl)
            ax.plot(bl_sens, label=lab, ls=ls)
            ax.axvline(2*np.pi*np.sqrt(bl[0]**2 + bl[1]**2)/0.5, color='lightgray', ls='--')
            ax.set_yscale('log')
        ax.set_xlabel('$\ell$')
        ax.legend(ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_ylabel(r'$\sim$ sensitivity (diag($\mathcal{F}^{-1}$) ) [{\rm $\mu~K^2$}]')
        ax.set_ylim(1e-1, 1e4)
        if pointing_ind is None:
            ax.set_title('All pointings')
        else:
            ax.set_title('Pointing {:.0f}$^{{\circ}}$'.format(
                self.pointings[pointing_ind]))
