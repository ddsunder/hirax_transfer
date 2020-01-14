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

class HEALpixBeam(config.Reader):

    filename = config.Property(proptype=str)
    # Assumes "beam" dataset with dims: pol, freq, pixels
    # and "index_map/freqs" with frequencies in MHz

    def __call__(self, angpos, zenith, wavelength, feed, pol_index):

        freq = (wavelength*units.m).to('MHz', equivalencies=units.spectral()).value

        with h5py.File(self.filename, 'r') as fil:
            freq_ind = np.argmin(np.abs(fil['index_map/freqs'] - freq))
            return fil['beam'][pol_index, freq_ind, :]

BEAM_TYPES = {
    'airy': AiryBeam,
    'gaussian': GaussianBeam,
    'healpix': HEALpixBeam,
}
