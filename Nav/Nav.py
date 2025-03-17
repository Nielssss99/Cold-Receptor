import json
from numba import njit
import numpy as np
import os

# Based on the code of the following paper:
# Balbi P, Massobrio P, Hellgren Kotaleski J. A single Markov-type kinetic model 
# accounting for the macroscopic currents of all human voltage-gated sodium channel isoforms. 
# PLoS Comput Biol. 2017 Sep 1;13(9):e1005737. doi: 10.1371/journal.pcbi.1005737. 
# PMID: 28863150; PMCID: PMC5599066.

# Get the path of the current file
file_path = os.path.dirname(os.path.abspath(__file__))

# Load Nav parameters --> Set as global variable to prevent reloading
Nav_params = np.empty((9, 3, 16), dtype = np.float64)
for inav in range(1, 10):    
    with open(file_path + f'\\Nav1{inav}.json') as json_file:
        # Load Nav parameters
        dictNav = json.load(json_file)

        # Convert to numpy arrays
        bs = np.array(
            [
                dictNav["C1C2b2"], 
                dictNav["C2C1b1"], dictNav["C2C1b2"], 
                dictNav["C2O1b2"],
                dictNav["O1C2b1"], dictNav["O1C2b2"], 
                dictNav["C2O2b2"], 
                dictNav["O2C2b1"], dictNav["O2C2b2"], 
                dictNav["O1I1b1"], dictNav["O1I1b2"], 
                dictNav["I1O1b1"],
                dictNav["I1C1b1"], 
                dictNav["C1I1b2"], 
                dictNav["I1I2b2"], 
                dictNav["I2I1b1"]
            ], dtype = np.float64
            )

        vvs = np.array(
            [
                dictNav["C1C2v2"], 
                dictNav["C2C1v1"], dictNav["C2C1v2"], 
                dictNav["C2O1v2"],
                dictNav["O1C2v1"], dictNav["O1C2v2"], 
                dictNav["C2O2v2"], 
                dictNav["O2C2v1"], dictNav["O2C2v2"], 
                dictNav["O1I1v1"], dictNav["O1I1v2"], 
                dictNav["I1O1v1"],
                dictNav["I1C1v1"], 
                dictNav["C1I1v2"], 
                dictNav["I1I2v2"], 
                dictNav["I2I1v1"]
            ], dtype = np.float64
            )

        ks = np.array(
            [
                dictNav["C1C2k2"],
                dictNav["C2C1k1"], dictNav["C2C1k2"], 
                dictNav["C2O1k2"],
                dictNav["O1C2k1"], dictNav["O1C2k2"], 
                dictNav["C2O2k2"],
                dictNav["O2C2k1"], dictNav["O2C2k2"], 
                dictNav["O1I1k1"], dictNav["O1I1k2"], 
                dictNav["I1O1k1"],
                dictNav["I1C1k1"],
                dictNav["C1I1k2"],
                dictNav["I1I2k2"],
                dictNav["I2I1k1"]
            ], dtype = np.float64
            )
        
        # Store Nav parameters
        Nav_params[inav - 1, 0, :] = bs
        Nav_params[inav - 1, 1, :] = vvs
        Nav_params[inav - 1, 2, :] = ks

@njit(fastmath = True)
def tempcorr(Temp: np.float64) -> np.float64:
    """Goal:
        Calculate the temperature correction factor.
    --------------------------------------------------------------------------
    Inputs:
        Temp: Temperature [°C]
    --------------------------------------------------------------------------
    Outputs:
        Q10: Q10 temperature coefficient."""
    return 3 ** ((Temp - 20) / 10)

@njit(fastmath = True)
def transition_matrix(v: np.float64, Q10: np.float64, inav: int) -> np.ndarray[np.float64]:
    """Goal:
        Create the transition matrix for the Nav channel.
    --------------------------------------------------------------------------
    Inputs:
        v: Membrane potential [mV]
        Q10: Q10 temperature coefficient.
        inav: Nav channel number.
    --------------------------------------------------------------------------
    Outputs:
        Q: Transition matrix."""
    # Unpack Nav parameters
    bs, vvs, ks = Nav_params[inav, 0, :], Nav_params[inav, 1, :], Nav_params[inav, 2, :]
    # Define the rates and sum some of them
    rates_raw = Q10 * bs / (1 + np.exp((v - vvs) / ks))
    rates = rates_raw[np.array([0, 1, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15])].copy()
    rates[np.array([1, 3, 5, 6])] += rates_raw[np.array([2, 5, 8, 10])]

    # Create the transition matrix
    Q = np.zeros((6, 6))
    Q[0, 0] = -(rates[0] + rates[9])
    Q[0, 1] = rates[1]
    Q[0, 4] = rates[8]
    Q[1, 0] = rates[0]
    Q[1, 1] = -(rates[1] + rates[2] + rates[4])
    Q[1, 2] = rates[3]
    Q[1, 3] = rates[5]
    Q[2, 1] = rates[2]
    Q[2, 2] = -(rates[3] + rates[6])
    Q[2, 4] = rates[7]
    Q[3, 1] = rates[4]
    Q[3, 3] = -rates[5]
    Q[4, 0] = rates[9]
    Q[4, 2] = rates[6]
    Q[4, 4] = -(rates[8] + rates[7] + rates[10])
    Q[4, 5] = rates[11]
    Q[5, 4] = rates[10]
    Q[5, 5] = -rates[11]

    return Q


@njit(fastmath = True)
def Nav_model(y: np.ndarray[np.float64], v: np.float64, Temp: np.float64, inav: int
        ) -> np.ndarray[np.float64]:
    """Goal:
        General model for Nav channels 1-9.
    --------------------------------------------------------------------------
    Inputs:
        y: State vector of the Nav channel.
        v: Membrane potential [mV]
        Temp: Temperature [°C]
        inav: Nav channel number + 1
    --------------------------------------------------------------------------
    Outputs:
        dydt: Derivative of the state vector."""
    Q10 = tempcorr(Temp)
    Q = transition_matrix(v, Q10, int(inav - 1)) * 1000 # Rates from 1/ms to 1/s
    return Q @ y


def Nav_steady(v: np.float64, Temp: np.float64, inav: int) -> np.ndarray[np.float64]:
    """Goal:
        Calculate the steady state of the Nav channel.
    --------------------------------------------------------------------------
    Inputs:
        v: Membrane potential [mV]
        Temp: Temperature [°C]
        inav: Nav channel number.
    --------------------------------------------------------------------------
    Outputs:
        y: Steady state of the Nav channel.
    """
    # Get the transition matrix
    Q10 = tempcorr(Temp)
    Q = transition_matrix(v, Q10, int(inav - 1)) * 1000 # Rates from 1/ms to 1/s
    # One formula is adapted as the sum of probabilities should be 1
    Q[-1, :] = 1
    # Solve the linear system
    b = np.array([0, 0, 0, 0, 0, 1])
    steady_state = np.linalg.solve(Q, b)
    return steady_state