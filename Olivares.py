from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.integrate import odeint
from numba.typed import Dict
from numba.core import types

from Nav.Nav import Nav_steady, Nav_model
from Kv.Kv import Kv3_2_steady_state, Kv3_2

@njit(fastmath=True)
def kmmrt(T, label, Tref):
    if label == "Na":
        dCp = -2.70 * 1000 # --> Na (done)
        dH = 33.90 * 1000
        dS = -133.10481651
    elif label == "K":
        dCp = -1.70 * 1000 # --> K (done)
        dH = 31.80 * 1000
        dS = -140.03208024
    elif label == "Ca":
        dCp = -5.07 * 1000
        dH = 70.80 * 1000
        dS = -11.38289667
    T0 = Tref + 273.15
    kb = 1.38064852e-23
    h = 6.62607004e-34
    R = 8.31
    kmmr = (kb * T) / h * np.exp(
        -(dCp * (T - T0) + dH) / (R * T)
        +
        (dCp * np.log(T / T0) + dS) / R)
    return kmmr

@njit(fastmath=True)
def sigmoid(x: np.float64) -> np.float64:
    """Goal:
        Sigmoid Activation Function.
    --------------------------------------------------------------------
    Input:
        x: input value
    --------------------------------------------------------------------
    Output:
        Activation value [-]"""
    return 1 / (1 + np.exp(-x))

@njit(fastmath=True)
def get_inf(VT: np.ndarray[np.float64], l1: np.ndarray[np.float64], 
            b1: np.ndarray[np.float64], cs: tuple[np.float64], 
            current_val: np.float64, flag: str
            ) -> np.float64:
    """Goal:
        Get the derivative of the given state variable.
    --------------------------------------------------------------------
    Input:
        VT: Voltage [mV] and Temperature [K]
        l1: Weights for the first layer. Here, 1 / K.
             --> steepness of the sigmoid
        b1: Bias for the first layer --> shift of the sigmoid
        cs: Parameters for the time constant calculation
            - cs[0]: 1 / Time constant
            - cs[1]: Shift for the voltage-dependent activation
            - cs[2]: Slope for the voltage-dependent activation (here, 1 / K)
            - cs[3]: Bias for the time constant
            - cs[4]: Scaler for the time constant
        current_val: Current activation value
        flag: Type of state variable
    --------------------------------------------------------------------
    Output:
        Activation value [-]"""
    # Adjust input for steady state activation sigmoid
    xinf_md = ((VT + b1) @ l1).sum() # ---> For now we just take the sum!
    # Activation Function to get the steady state activation
    inf_md = sigmoid(xinf_md)

    # Get '1 / time constant'
    if flag in ["hNa", "mK"]:
        # @ Maksymchuk et al. (2023) mention that 'tau_hNa' is fitted to
        # the data of 'Wang (2013)'
        xcosh = (VT[0][0] + cs[1]) * (abs(cs[0]) * cs[2])
        arccosinus = np.cosh(xcosh)
        if arccosinus != np.inf:
            tau_1 = arccosinus / (cs[3] * arccosinus + cs[4])
        else:
            # If arccosinus is infinite, cs[4]'s contribution is negligible
            tau_1 = 1 / cs[3] 
    elif flag in ["mBK"]:
        xsig = cs[0] * (VT[0][0] + cs[1])
        divsig = cs[4] * sigmoid(xsig) + cs[3]
        if divsig != 0:
            tau_1 = 1 / divsig
            tau_1 = min(tau_1, 10**15)
        else:
            tau_1 = 10**15
    elif flag in ["mNa", "mCa", "hCa"]:
        tau_1 = cs[0]
    else:
        raise ValueError("Invalid flag for the state variable!")

    # ReLU --> 1 / timeconstant should always be either 0 or positive!
    tau_1 = tau_1 * (tau_1 > 0.0)

    # Calculate the derivative
    dmddt = (inf_md - current_val) * tau_1
    return dmddt


@njit(fastmath=True)
def hill_like(x: np.float64, K: np.float64, n: np.float64) -> np.float64:
    """Goal:
        Hill-like activation function.
    --------------------------------------------------------------------
    Input:
        x: input value
        K: Half-maximal concentration
        n: Hill coefficient
    --------------------------------------------------------------------
    Output:
        Activation value"""
    return 1 / (1 + (K / x)**n) # Higher if higher concentration of x

@njit(fastmath=True)
def forward(
    y: np.ndarray[np.float64], t: np.ndarray[np.float64], 
    dy: np.ndarray[np.float64], T: np.float64, vars: tuple[np.float64]
    ) -> np.ndarray[np.float64]:
    """Goal:
        Main function to solve the ODE system for a cold thermoreceptor.
    --------------------------------------------------------------------
    Input:
        y: state variables [Vm, ...]
        t: time [s]
        dy: derivative of the state variables
        T: temperature [K]
        vars: model parameters
    --------------------------------------------------------------------
    Output:
        dy: derivative of the state variables
    """
    # Combine Voltage and Temperature
    VT = np.array([[y[0], T]], dtype = np.float64) # Voltage and Temperature
    
    # ---- Get Equilibrium Potential for Calcium [mV]
    # ECa = vars["1000_R_Z_F"] * T  * np.log(vars["Cae"] / max(y[7], (1 * 10**(-15)))) # Nernst Potential for Calcium
    # ETRP = vars["kPCa"] * ECa + vars["kPNa"] * vars["ENa"] + vars["kPK"] * vars["EK"]

    rho = 1.3**((T - (273.15 + 30)) / 10) # Temperature dependence of the rate constants
    # phi = 3.0**((T - (273.15 + 30)) / 10) # Temperature dependence of the rate constants
    phi_Na = kmmrt(T, "Na", 30)
    phi_K = kmmrt(T, "K", 30)
    # phi_Ca = kmmrt(T, "Ca", 30)

    # ---- Currents & Membrane Potential
    I_Na = rho * vars["gNa"] * (y[3] + y[4]) * (y[0] - vars["ENa"]) # Voltage Gated Sodium Current
    
    I_K = rho * vars["gK"] * y[8]**2 * (y[0] - vars["EK"]) # Voltage Gated Potassium Current

    I_L = rho * vars["gL"] * (y[0] - vars["EL"]) # Leak Current
    # I_Ca = rho * vars["gCa"] * y[5] * y[6] * (y[0] - ECa) # Voltage Gated N-Type Calcium Current
    # I_SK = rho * vars["gSK"] * y[8] * (y[0] - vars["EK"]) # Small Conductance Calcium-Activated Potassium Current

    # fCaBk = hill_like(max(y[7], (1 * 10**(-15))), vars["CaBK"], vars["nBK"]) # Calcium dependent BK activation [-]
    # I_BK = rho * vars["gBK"] * fCaBk * y[4]**4 * (y[0] - vars["EK"]) # Big Conductance Calcium-Activated Potassium Current


    # dH, dS, z, F, R, ETRP = -167076, -560, 0.82, 96485, 8.3144, 0
    # am8 = 1 / (1 + np.exp((dH - T * dS - z * F * y[0] / 1000) / (R * T)))
    # I_TRP = vars["GleakTest"] * am8 * (y[0] - ETRP)
    # I_TRP = vars["GleakTest"] * y[9] * y[10] * (y[0] - ETRP) # TRP current
    # I_TRPCa = y[10] * vars["GleakTest"] * y[9] * vars["kPCa"] * (y[0] - ECa) # Ca dependent modulation of TRP current


    # # Kv1.1 @ Channelpedia
    # rhoKv1_1 = 1.3**((T - (273.15 + 24)) / 10)
    # phiKv1_1 = kmmrt(T, "K", 24)
    # I_Kv1_1 = rhoKv1_1 * vars["gKv1.1"] * y[11] * y[12]**2 * (y[0] - vars["EK"])

    # # Kv1.2 @ Channelpedia
    # rhoKv1_2 = 1.3**((T - (273.15 + 20)) / 10)
    # phiKv1_2 = kmmrt(T, "K", 20)
    # I_Kv1_2 = rhoKv1_2 * vars["gKv1.2"] * y[13] * y[14] * (y[0] - vars["EK"])

    # # HCN1 @ Channelpedia
    # rhoHCN1 = 1.3**((T - (273.15 + 0)) / 10)
    # phiHCN1 = kmmrt(T, "K", 0)
    # vars["EHCN1"] = -45
    # I_HCN1 = rhoHCN1 * vars["gHCN1"] * y[15] * (y[0] - vars["EHCN1"])

    # # HCN2 @ Channelpedia
    # rhoHCN2 = 1.3**((T - (273.15 + 0)) / 10)
    # phiHCN2 = kmmrt(T, "K", 0)
    # vars["EHCN2"] = -45
    # I_HCN2 = rhoHCN2 * vars["gHCN2"] * y[16] * (y[0] - vars["EHCN2"])

    # # Nav1.1 @ Zheng
    # rhoNav1_1 = 1.3**((T - (273.15 + 37)) / 10)
    # phiNav1_1 = kmmrt(T, "Na", 37)
    # I_Nav1_1 = rhoNav1_1 * vars["gNav1.1"] * y[17]**3 * y[18] * (y[0] - vars["ENa"])

    # Membrane Potential
    dy[0] = -(I_Na + I_K + I_L) / vars["Cap"]
    #dy[0] = -(I_Na + I_K + I_L + I_Ca + I_SK + I_BK + I_TRP) / vars["Cap"] #+ I_Kv1_1 + I_Kv1_2 + I_HCN1 + I_HCN2 + I_Nav1_1
    # ---- Sodium/Na+
    dy[1:7] = Nav_model(
        y[1:7], y[0], T - 273.15, 9
        )
    

    # ---- Potassium/K+ 
    dy[8] = Kv3_2(y[0], T - 273.15, y[8]) # Kv3.2

    # # ----- Big Potassium
    # dy[4] = phi_K * get_inf(
    #     VT, 
    #     np.array([[vars["1_KmBK"]], [0.0]]),
    #     np.array([[vars["vmBK"], 0.0]]),
    #     (
    #         vars["1_KmBK_tau"], vars["vmBK_tau"], np.nan, 
    #         vars["bias_tau_mBK"], vars["scale_tau_mBK"]
    #         ), 
    #     y[4], 
    #     flag = "mBK"
    #     )
    # # ---- Calcium/Ca2+
    # dy[5] = phi_Ca * get_inf(
    #     VT, 
    #     np.array([[vars["1_KmCa"]], [0.0]]),
    #     np.array([[vars["vmCa"], 0.0]]),
    #     (vars["tau_mCa"], np.nan, np.nan, np.nan, np.nan), 
    #     y[5], 
    #     flag = "mCa"
    #     )
    
    # dy[6] = phi * get_inf(
    #     VT, 
    #     np.array([[vars["1_KhCa"]], [0.0]]),
    #     np.array([[vars["vhCa"], 0.0]]),
    #     (vars["tau_hCa"], np.nan, np.nan, np.nan, np.nan), 
    #     y[6], 
    #     flag = "hCa"
    #     ) # --> 1/KhCa should be negative!
    # # ---- Calcium Concentration Inside the Cell
    # dy[7] = -(I_Ca + I_TRPCa) / vars["F_Z_Vol"] - vars["kCa"] * (y[7] - vars["Camin"])

    # # ---- Small Potassium Current
    # fSK = hill_like(max(y[7], (1 * 10**(-15))), vars["CaSK"], vars["nSK"])
    # dy[8] = phi_K * (fSK - y[8]) * vars["1_tau_mSK"] # Activation

    # # ---- TRP
    # Cain_sqr = y[7]**2
    # hTRP = 1 - Cain_sqr / (vars["Cah"]**2 + Cain_sqr)
    # dy[9] = (hTRP - y[9]) * vars["1_tau_hTRP"]

    # mTRP = vars["mTRP_C"] + vars["mTRP_B"] / (1 + np.exp(-vars["mTRP_A"] * (vars["mTRP_th"] - T))) # th and t are reversed! --> incorrect wih paper
    # dy[10] = (mTRP - y[10]) * vars["1_tau_mTRP"]


    # # Kv1.1
    # mInf_Kv1_1 = 1.0000 / (1 + np.exp((y[0] - -30.5000) / -11.3943))
    # mTau_Kv1_1 = (30.0000 / (1 + np.exp((y[0] - -76.5600) / 26.1479))) / 1000 # [s]
    # dy[11] = phiKv1_1 * (mInf_Kv1_1 - y[11]) / mTau_Kv1_1

    # hInf_Kv1_1 = 1.0000 / (1 + np.exp((y[0] - -30.0000) / 27.3943)) 
    # hTau_Kv1_1 = (15000.0000 / (1 + np.exp((y[0] - -160.5600) / -100.0000))) / 1000 # [s]
    # dy[12] = phiKv1_1 * (hInf_Kv1_1 - y[12]) / hTau_Kv1_1

    # # Kv1.2
    # mInf_Kv1_2 = 1.0000/(1+ np.exp((y[0] - -21.0000)/-11.3943))
    # mTau_Kv1_2 = (150.0000/(1+ np.exp((y[0] - -67.5600)/34.1479))) / 1000 # [s]
    # dy[13] = phiKv1_2 * (mInf_Kv1_2 - y[13]) / mTau_Kv1_2

    # hInf_Kv1_2 = 1.0000/(1+ np.exp((y[0] - -22.0000)/11.3943))
    # hTau_Kv1_2 = (15000.0000/(1+ np.exp((y[0] - -46.5600)/-44.1479))) / 1000 # [s]
    # dy[14] = phiKv1_2 * (hInf_Kv1_2 - y[14]) / hTau_Kv1_2

    # # HCN1
    # mInf_HCN = 1.0000/(1+np.exp((y[0]- -94)/8.1)) 
    # mTau_HCN = 30.0000 / 1000
    # dy[15] = phiHCN1 * (mInf_HCN - y[15]) / mTau_HCN

    # # HCN2
    # mInf_HCN2 = 1.0000/(1+np.exp((y[0]- -99)/6.2)) 
    # mTau_HCN2 = 184.0000 / 1000
    # dy[16] = phiHCN2 * (mInf_HCN2 - y[16]) / mTau_HCN2

    # # Nav1.1
    # vhminf = -35
    # kminf = 5.5
    # amtaul = 0.006
    # bmtaul = 0.08
    # cmtaul = -55
    # dmtaul = 12
    # brkvmtau = -50
    # amtaur = 0.015
    # bmtaur = 0.065
    # cmtaur = -10.8
    # dmtaur = 10
    # vhhinf = -40
    # khinf = 12
    # ahtaul = 1.98
    # bhtaul = 8.54
    # chtaul = -73.3
    # dhtaul = 4.7
    # brkvhtau = -55
    # ahtaur = 0.17
    # bhtaur = 10.82
    # chtaur = -39.1
    # dhtaur = 4.59
    # mInf_Nav1_1 = 1/(1+np.exp(-(y[0] - vhminf) / kminf))
    # if y[0] < brkvmtau:
    #      mtau_Nav1_1 = amtaul+bmtaul*(1/(1+np.exp(-(y[0]-cmtaul)/dmtaul)))
    # else:
    #      mtau_Nav1_1 = amtaur+bmtaur*(1/(1+np.exp((y[0]-cmtaur)/dmtaur)))
    # mtau_Nav1_1 = mtau_Nav1_1 / 1000
    # dy[17] = phiNav1_1 * (mInf_Nav1_1 - y[17]) / mtau_Nav1_1

    # hinf_Nav1_1 = 1/(1+np.exp((y[0]-vhhinf)/khinf))

    # if y[0] < brkvhtau:
    #     htau_Nav1_1 = ahtaul+bhtaul*(1/(1+np.exp(-(y[0]-chtaul)/dhtaul)))
    # else:
    #      htau_Nav1_1 = ahtaur+bhtaur*(1/(1+np.exp((y[0]-chtaur)/dhtaur)))
    # htau_Nav1_1 = htau_Nav1_1 / 1000
    # dy[18] = phiNav1_1 * (hinf_Nav1_1 - y[18]) / htau_Nav1_1
    
    return dy 


def solve_model(temps, vars):

    R =  8.31 * 10**(-9) # Gas Constant [J/(nmolK)]
    Z = 2 # Number of electrons involved [-]
    F = 96485.35 * 10**(-9) # Faraday Constant [C/nmol]

    # Map list to dictionary
    numba_dict = Dict.empty(
        key_type=types.unicode_type,  # String keys
        value_type=types.float64      # Float values
    )
    
    numba_dict["1_KmNa"] = vars[0]
    numba_dict["vmNa"] = vars[1]
    numba_dict["1_tau_mNa"] = vars[2]

    numba_dict["1_KhNa"] = vars[3]
    numba_dict["vhNa"] = vars[4]
    numba_dict["1_tau_KhNa"] = vars[5]
    numba_dict["bias_tau_hNa"] = vars[6]
    numba_dict["scale_tau_hNa"] = vars[7]

    numba_dict["1_KmK"] = vars[8]
    numba_dict["vmK"] = vars[9]
    numba_dict["1_tau_KmK"] = vars[10]
    numba_dict["bias_tau_mK"] = vars[11]
    numba_dict["scale_tau_mK"] = vars[12]

    numba_dict["CaBK"] = vars[13]
    numba_dict["nBK"] = vars[14]

    numba_dict["1_KmBK"] = vars[15]
    numba_dict["vmBK"] = vars[16]
    numba_dict["1_KmBK_tau"] = vars[17]
    numba_dict["vmBK_tau"] = vars[18]
    numba_dict["bias_tau_mBK"] = vars[19]
    numba_dict["scale_tau_mBK"] = vars[20]
    
    numba_dict["1_KmCa"] = vars[21]
    numba_dict["vmCa"] = vars[22]
    numba_dict["tau_mCa"] = vars[23]

    numba_dict["1_KhCa"] = vars[24]
    numba_dict["vhCa"] = vars[25]
    numba_dict["tau_hCa"] = vars[26]

    numba_dict["Vol"] = vars[27]
    numba_dict["Camin"] = vars[28]
    numba_dict["kCa"] = vars[29]

    numba_dict["CaSK"] = vars[30]
    numba_dict["nSK"] = vars[31]
    numba_dict["1_tau_mSK"] = vars[32]
    
    numba_dict["gNa"] = vars[33]
    numba_dict["gK"] = vars[34]
    numba_dict["gL"] = vars[35]
    numba_dict["gCa"] = vars[36]
    numba_dict["gSK"] = vars[37]
    numba_dict["gBK"] = vars[38]
    numba_dict["GleakTest"] = vars[39]

    numba_dict["PCa"] = vars[40]
    numba_dict["PK"] = vars[41]
    numba_dict["Cah"] = vars[42]
    numba_dict["1_tau_hTRP"] = vars[43]
    numba_dict["mTRP_C"] = vars[44]
    numba_dict["mTRP_B"] = vars[45]
    numba_dict["mTRP_A"] = vars[46]
    numba_dict["mTRP_th"] = vars[47]
    numba_dict["1_tau_mTRP"] = vars[48]


    numba_dict["gKv1.1"] = vars[49]
    numba_dict["gKv1.2"] = vars[50]
    numba_dict["gHCN1"] = vars[51]
    numba_dict["gHCN2"] = vars[52]
    numba_dict["gNav1.1"] = vars[53]


    numba_dict["Cae"] = 2.0 * 10**6 # External Ca2+ concentration [nM]
    numba_dict["1000_R_Z_F"] = 1000 * R / (Z * F) # Invariant term for ECa calculation
    numba_dict["F_Z_Vol"] = F * Z * numba_dict["Vol"] # Invariant term for dCa/dt calculation

    numba_dict["ENa"] = 61 / 1 * np.log10(440 / 50) # Equilibrium Potential Natrium [mV] @ page 41-42 of Purves et al.(2019)
    numba_dict["EK"] = 61 / 1 * np.log10(20 / 400) # Equilibrium Potential Kalium [mV] @ page 41-42 of Purves et al.(2019)
    numba_dict["EL"] = 61 / 1 * np.log10(20 / 400) # Equilibrium Potential Leak Current [mV] @ Equal to EK, similar to Maksymchuk et al. (2023)
    ECa = 61 / 2 * np.log10(1.5 / 0.0001) # [mV] @ page 41-42 of Purves et al.(2019)
    numba_dict["Cap"] = 0.01 # Membrane Capacitance [nF]

    numba_dict["PNa"] = -(numba_dict["PK"] * numba_dict["EK"] + numba_dict["PCa"] * ECa) / numba_dict["ENa"]
    numba_dict["kPCa"] = numba_dict["PCa"] / (numba_dict["PCa"] + numba_dict["PK"] + numba_dict["PNa"])
    numba_dict["kPNa"] = numba_dict["PNa"] / (numba_dict["PCa"] + numba_dict["PK"] + numba_dict["PNa"])
    numba_dict["kPK"] = numba_dict["PK"] / (numba_dict["PCa"] + numba_dict["PK"] + numba_dict["PNa"])

    
    
    threshold = -20
    Vr = -65


    freqs = []

    for temp in temps:
        y0_nav = Nav_steady(Vr, temp, 9) # Nav1.9
        y0_kv32 = Kv3_2_steady_state(Vr)
        # Initialize
        y0 = np.array(
            [
            deepcopy(Vr), 
            y0_nav[0], y0_nav[1], y0_nav[2], y0_nav[3],
            y0_nav[4], y0_nav[5],
            y0_kv32
            ], dtype = np.float64)	
        dy = np.zeros((8), dtype = np.float64)

        # Normalize temperature
        T = 273.15 + temp
        # print(T, threshold)
        # print(y0)

        # Initialize
        t = np.arange(0, 300, 0.001)
        t_eval = np.arange(0, 30, 0.001)
        try:
            sol, info = odeint(forward, y0, t, args = (dy,
                T, numba_dict
                ), h0 = 0.00001, hmax = 0.01, atol = 1.e-9, rtol = 1.e-8,
                    mxords = 15, mxordn = 15, full_output = True)
            
            # if info["message"] != "Integration successful.":
            #     print(info["message"])
            # sol, info = odeint(forward, sol[-1, :], t_eval, args = (dy,
            #     T, numba_dict
            #     ), h0 = 0.00001, hmax = 0.01, atol = 1.e-9, rtol = 1.e-8,
            #         mxords = 15, mxordn = 15, full_output = True)
            
            plt.plot(t, sol[:, 0])
            plt.show()
            # fig, ax = plt.subplots()
            # ax2 = ax.twinx()
            # ax2.plot(t, sol[:, 11:], label = ["mKv1.1", "hKv1.1", "mKv1.2", "hKv1.2", "mHCN1", "mHCN2", "mNav1.1", "hNav1.1"])
            # ax.plot(t, sol[:, 0], "r", label = "Vm")
            # plt.legend()
            # plt.show()
            
            freqs.append((np.diff(sol[:, 0] > threshold) == 1).sum() / 300)
        except ZeroDivisionError:
            freqs.append(np.inf)
            raise ZeroDivisionError
        except ValueError as e:
            freqs.append(np.inf)
            print("Value Error")
            raise e
        except IndexError:
            freqs.append(np.inf)
            print("Index Error")
        
        # Delete the initial states --> apparently, this is necessary
        del y0
        del dy
    return freqs
        


#https://www.pnas.org/doi/10.1073/pnas.96.21.11825