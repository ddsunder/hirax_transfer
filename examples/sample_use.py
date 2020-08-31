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

# to get frequency channels
freqs = hirax_survey.frequencies

# to get unique baselines
bslns = hirax_survey.baselines

# corresponding redundancy of baselines
redun = hirax_survey.redundancy

# compute beam transfer matrix for a fixed frequency
"""
output beam transfer matrix has shape (nbaselines, npol, lmax, 2*lmax+1)
"""
bt_freq = hirax_survey.transfer_for_frequency(0)

# compute beam transfer matrix for a fixed frequency
"""
output beam transfer matrix has shape (nfreq, npol, lmax, 2*lmax+1)
"""
bt_freq = hirax_survey.transfer_for_baseline(0)