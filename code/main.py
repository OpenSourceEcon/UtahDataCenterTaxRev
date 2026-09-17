"""
This file executes the main
"""

# Import packages
import numpy as np

"""
Set parameters and ranges

dc_peak_elec_capac: Data center peak eletrical capacity (megawatts, MW)
"""
dc_peak_elec_capac = 100
dc_peak_elec_capac_min = 0.9
dc_peak_elec_capac_max = 2_000

"""
Define functions
"""


def get_dc_fmv(elec_capac, plot=False):
    """
    Calculate data center fair market value (FMV). This estimated relationship
    between peak electrical capacity of a data center and its fair market value
    is described in the appendix of "Introducing DataCenterAtlas.org"
    (https://econosseur.rickecon.com/p/introducing-datacenteratlas). See Figure
    3 and Table 1 in that article and the equations that follow.

    Args:
        elec_capac (float): Peak electrical capacity of data center

    Returns:
        fmv (float): Fair market value of data center
    """
    if elec_capac >= 95:
        slope = 0.010018082083149
        intercept = -0.349602038703626
        fmv = slope * elec_capac + intercept
    else:
        exp_slope = 0.010396604326046
        exp_intercept = -1.02476498822734
        constant = -0.361475983740317
        fmv = np.exp(exp_slope * elec_capac + exp_intercept) + constant
    if plot:


    return fmv
