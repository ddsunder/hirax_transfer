#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:06:22 2020

@author: dalian
"""

import yaml
import hirax_transfer.core as hirax

f = open('config_gauss.yaml','r')
conf = yaml.load(f)
telescope = conf.pop('telescope')

hirax_survey = hirax.HIRAXSurvey()
hirax_survey.read_config(telescope)


# A lot of desired functionality already incorporated in driftscan and Devin's Hirax Transfer
# I have listed some examples

freqs = hirax_survey.frequencies        # frequency channels
bslns = hirax_survey.baselines          # unique baselines
redun = hirax_survey.redundancy         # corresponding redundancy of baselines
dish  = hirax_survey.dish_width         # dish diameter
nfreq = hirax_survey.num_freq           # number of frequency channels
delnu = hirax_survey.channel_width      # channel width
Sarea = hirax_survey.survey_area        # approximation of survey area in square degrees
FoV   = hirax_survey.FoV                # FoV for all frequency channels in str
Tsys  = hirax_survey.tsys_flat          # system temperature
lmax  = hirax_survey.lmax               # maximum ell that we set, not the instrument
mmax  = hirax_survey.mmax               # maximum m that we set, not the instrument

"""
TO DO LIST:
    Add baseline sensitivity
    Add in noise sensitivity
    return umin, umax
    Set ell range from umin and umax 
"""


# compute beam transfer matrix for a fixed frequency
"""
output beam transfer matrix has shape (nbaselines, npol, lmax, 2*lmax+1)
"""
bt_freq = hirax_survey.transfer_for_frequency(0)

# compute beam transfer matrix for a fixed baseline
"""
output beam transfer matrix has shape (nfreq, npol, lmax, 2*lmax+1)
"""
bt_freq = hirax_survey.transfer_for_baseline(0)