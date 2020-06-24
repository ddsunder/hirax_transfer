from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

# from scipy.special import jn
from scipy.special import j1 as bessel_j1
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import units as units
from astropy import coordinates as coords
import h5py
import healpy as hp

from caput import config

def fetch_taper(conf):
    if 'taper' in conf.keys():
        return BEAMTaper.from_config(conf['taper'])
    else:
        return None

def fetch_beam(conf):
    btype = conf['type']
    return BEAM_TYPES[btype].from_config(conf)

def hpang2lonlat(hpang):
    """Lon-lat in degrees from healpix coords"""
    theta, phi = hpang.T
    lat = np.degrees(np.pi/2 - theta)
    lon = np.degrees(2*np.pi - phi)
    return lon, lat

def separations(angpos, zenith):
    """
    Zenith-position separations in radians. This calculation could be cached
    """
    lon, lat = hpang2lonlat(angpos)
    pixel_coords = coords.SkyCoord(ra=lon, dec=lat, unit=units.degree)
    zlon, zlat = hpang2lonlat(zenith)
    zenith_coords = coords.SkyCoord(ra=zlon, dec=zlat, unit=units.degree)
    return zenith_coords.separation(pixel_coords).to('radian').value


class AiryBeam(config.Reader):

    diameter = config.Property(proptype=float, default=6.)
    sep_limit = config.Property(proptype=float, default=90.)

    def __call__(self, angpos, zenith, wavelength, feed, pol_index):

        seps = separations(angpos, zenith)
        x = np.pi*self.diameter/wavelength*np.sin(seps)
        out = (2*bessel_j1(x)/x)  # (Voltage Beam)
        out[np.degrees(seps) > self.sep_limit] = 0.
        return out

class GaussianBeam(config.Reader):

    diameter = config.Property(proptype=float, default=6)
    fwhm_factor = config.Property(proptype=float, default=1.0)
    sep_limit = config.Property(proptype=float, default=np.inf)

    def __call__(self, angpos, zenith, wavelength, feed, pol_index):

        fwhm = self.fwhm_factor*wavelength/self.diameter
        sigma = gaussian_fwhm_to_sigma*fwhm
        seps = separations(angpos, zenith)

        out = np.exp(-seps**2/2/sigma**2)**0.5  # (Voltage Beam)
        if np.isfinite(self.sep_limit):
            out[np.degrees(seps) > self.sep_limit] = 0.
        return out

class HEALPixBeam(config.Reader):

    """
    Much of this is hard-coded and would benefit from
    a class that handles all of this better...
    Assumes hdf5 file containts "beam" dataset with dims:
    pol_index (x/y feed), freq, healpix pixels, v/h
    Must setup so h5py will read as complex
    (usually 'r' and 'i' compound dset)
    Assumes the input beam is pointed at [phi=pi, theta=pi/2]
    Also an "index_map/freqs" with frequencies in MHz
    This can fail badly if the beamtransfer class ends up
    using a different nside...
    """

    filename = config.Property(proptype=str)

    def __call__(self, angpos, zenith, wavelength, feed, pol_index):

        freq = (wavelength*units.m).to('MHz', equivalencies=units.spectral()).value

        with h5py.File(self.filename, 'r') as fil:
            freq_ind = np.argmin(np.abs(fil['index_map/freqs'][()] - freq))
            beam = fil['beam'][pol_index, freq_ind, :][()]

        rot = [0, zenith[0]-np.pi/2]
        r = hp.Rotator(deg=False, rot=rot)
        return np.stack(
            [r.rotate_map_pixel(beam[..., 0]),
             r.rotate_map_pixel(beam[..., 1])
            ], axis=-1)

BEAM_TYPES = {
    'airy': AiryBeam,
    'gaussian': GaussianBeam,
    'healpix': HEALPixBeam,
}

class BEAMTaper(config.Reader):

    level = config.Property(proptype=float, default=0.) # in dB
    start = config.Property(proptype=float, default=3) # radius in FWHM units
    end = config.Property(proptype=float, default=5)  # radius in FWHM units
    diameter = config.Property(proptype=float, default=6)  # in m, for FWHM calculation
    # type = ## in future specify type of taper maybe

    def __call__(self, angpos, zenith, wavelength):

        seps = separations(angpos, zenith)
        FWHM = wavelength/self.diameter # in rad
        dec_lev = 10**(0.1*self.level)

        normed_seps = seps / FWHM

        out = np.ones_like(seps)
        out[normed_seps < self.start] = 1.
        out[normed_seps >= self.end] = dec_lev

        taper_inds = (normed_seps >= self.start) & (normed_seps < self.end)
        nseps = normed_seps[taper_inds]

        taper_range = (nseps - self.start)*np.pi/(self.end - self.start)
        tape_func = ((np.cos(taper_range) + 1)/2.)

        if self.level >= 0:
            out[taper_inds] = (1 - tape_func)*(dec_lev - 1) + 1
        else:
            out[taper_inds] = (tape_func)*(1 - dec_lev) + dec_lev

        return out
