import numpy as np
from numba import njit

@njit(fastmath=True)
def Kv3_2_steady_state(v: np.float64):
    """Goal:
        Steady State of Hodgkin-Huxley model of Kv3.2 channel
    ---------------------------------------------------------------------------
    Input:
        v: membrane voltage (mV)
    ---------------------------------------------------------------------------
    Output:
        mInf: steady state gating variable
    """
    mInf = 1 / (1 + np.exp((v - -0.373267) / -8.568187))
    return mInf

@njit(fastmath=True)
def Kv3_2(v: np.float64, Temp: np.float64, m: np.float64):
    """Goal:
        Hodgkin-Huxley model of Kv3.2 channel
        @ Ranjan, R., Schartner, M., Khanna, N., & Johnston, K. Channelpedia 
            https://channelpedia.epfl.ch/wikipages/12/, accessed on 2025 Mar 17
    ---------------------------------------------------------------------------
    Input:
        v: membrane voltage (mV)
        Temp: temperature (C)
        m: gating variable
    ---------------------------------------------------------------------------
    Output:
        dmdt: time derivative of gating variable
    """
    Q10 = 3**((Temp - 22) / 10)
    mInf = Kv3_2_steady_state(v)
    mTau = 3.241643 + (19.106496 / (1 + np.exp((v - 19.220623) / 4.451533)))
    mTau = mTau / 1000 # ms to s
    dmdt = Q10 * (mInf - m) / mTau
    return dmdt
