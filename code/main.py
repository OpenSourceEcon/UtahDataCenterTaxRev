"""
This file executes the main
"""

# Import packages
import numpy as np
import fig_MWtoFMV as fmv

"""
Set parameters and ranges

dc_peak_elec_capac: Data center peak eletrical capacity (megawatts, MW)
"""
dc_peak_elec_capac_default = 100
dc_peak_elec_capac_min = 0.9
dc_peak_elec_capac_max = 1_200

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
        slope = 0.01001808208314903
        intercept = -0.34960203870362627
        fmv = slope * elec_capac + intercept
    else:
        exp_slope = 0.010396604326045987
        exp_intercept = -1.0247649882273393
        constant = -0.36147598374031653
        fmv = np.exp(exp_slope * elec_capac + exp_intercept) + constant
    if plot:
        fmv.make_mw_fmv_plot(title=True)

    return fmv
