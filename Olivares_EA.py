import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import Repair
from pymoo.termination import get_termination
from Olivares import solve_model
from pymoo.core.problem import StarmapParallelization
import sqlite3
import multiprocessing
from pymoo.core.callback import Callback
import os


# ---- DATABASE SETUP ----
def create_database(DB_NAME):
    """Initialize the database and create the results table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                Generation INTEGER,
                solution TEXT,
                frequency TEXT,
                fitness REAL)''')
    conn.commit()
    conn.close()

def save_to_database(generation, solution, frequencies, fitness):
    """Save a solution and its fitness value to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO results (generation, solution, frequency, fitness) VALUES (?, ?, ?, ?)", 
              (generation, str(solution.tolist()), str(frequencies), fitness))
    conn.commit()
    conn.close()


class MyRepair(Repair):

    def _do(self, problem, X, **kwargs):
        # # Thresholds for sigmoids: Set order of thresholds
        # map_v = {"vmNa": 1, "vhNa": 4} # str to idx --> low to high

        # #map_v = {"vmK": 9, "vmNa": 1, "vhNa": 4}
        # idx_v = list(map_v.values())
        # X[:, idx_v] = np.sort(X[:, idx_v], axis = 1)

        # map_v = {"vmK": 9, "vhNa": 4} # str to idx --> low to high
        # idx_v = list(map_v.values())
        # X[:, idx_v] = np.sort(X[:, idx_v], axis = 1)
        

        # # Slope of sigmoids
        # map_K = {"1_kmK": 8, "1_kmNa": 0, "1_khNa": 3} # str to idx --> low to high
        # idx_K = list(map_K.values())
        # X[:, idx_K] = abs(np.sort(X[:, idx_K], axis = 1)) * np.sign(X[:, idx_K])

        # # 1 / K
        # map_K = {"1_kmNa"}


        return X
    

np.random.seed(0)
limits = [0.5, 1.5]
class MyProblem(Problem): # --> not elementwise problem and then multiprocess! --> combine results later
    
    def __init__(self, pool = None):
        super().__init__(n_var=54, n_obj=1, n_constr=0, 
                         xl = [
                             # [0, 1, 2]
                             0.01, -80, 1, # Activation mNa; 1 / KmNa (0.29), vmNa (24.7), 1 / tauNa (10000)
                             # [3, 4, 5, 6, 7]
                             limits[1] * -0.24, limits[0] * 41.2, limits[0] * 0.33, # Activation hNa; 1 / KhNaF (-0.24), vhNaF (41.2), cs[2] (0.33)
                             limits[0] * (7.5 * 10**(-4)), limits[0] * (4.5 * 10**(-3)), # cs[3] (7.5e-4), cs[4] (4.5e-3)
                             # [8, 9, 10, 11, 12]
                             limits[0] * 0.14, limits[0] * 12, limits[0] * 0.5, # Activation mK; 1 / KmK (0.14), vmK (12), cs[2] (0.5)
                             limits[0] * (7.5 * 10**(-4)), limits[0] * (5.0 * 10**(-3)), # cs[3] (7.5e-4), cs[4] (5.0e-3)
                             # [13, 14]
                             limits[0] * 1700, limits[0] * 3, # Activation mBK; CaBk (1700), nBK (3)
                             # [15, 16, 17, 18, 19, 20]
                             limits[0] * 0.033, limits[0] * 28.3, limits[0] * 0.044, # Activation mBK; 1 / KmBK (0.033), vmBK (28.3), 1 / tauBK (0.044)
                             limits[0] * 46, limits[0] * 0.1806, limits[1] * -0.1502, # vmBK_tau (46), bias_tau_mBK (0.1806), scale_tau_mBK (-0.1502)
                             # [21, 22, 23]
                             limits[0] * 0.15, limits[0] * 23, limits[0] * 285.7, # Activation mCa; 1 / KmCa (0.15), vmCa (23), 1 / tmCa (285.7)
                             # [24, 25, 26]
                             limits[1] * -0.083, limits[0] * 59, limits[0] * 10.5, # Activation hCa; 1 / KhCa (0.083), vhCa (59), 1 / thCa (10.5)
                             # [27, 28, 29]
                             limits[0] * 0.2, limits[0] * 50, limits[0] * 403, # Ca: Vol (0.2), Camin (50), kCa (403)
                             # [30, 31, 32]
                             limits[0] * 800, limits[0] * 3, limits[0] * 25, # SK: CaSK (800), nSK (3), 1 / tauSK (25)
                             # [33, 34, 35, 36, 37, 38, 39]
                             limits[0] * 80, limits[0] * 140, limits[0] * 0.25, # gNa (80), gK (140), gL (0.25)
                             limits[0] * 3.5, limits[0] * 0.31, limits[0] * 6, # gCa (3.5), gSK (0.31), gBK (6)
                             0, # Gleaktest

                             # [40, 41]	
                             0, 0, # pCa (0.4), pK (1)
                             # [42, 43]
                             limits[0] * 700, limits[0] * (1/10), # Cah (700), 1_tau_hTRP (0.1)
                             # [44, 45, 46, 47, 48]
                             limits[0] * 0, limits[0] * 1, 0.01, # mTRP_C (0), mTRP_B (1), mTRP_A (1)
                             273.15 + 10, limits[0] * (1 / 0.002), # mTRP_th (290.15), 1 / tau_TRP (1 / 0.002)
                             
                             # [49, 50]
                             0, 0, 0, 0, 0 # Kv1.1
                             ], 
                        xu = [
                            # [0, 1, 2]
                            1, 80, 10000, 
                            # [3, 4, 5, 6, 7]
                            limits[0] * -0.24, limits[1] * 41.2, limits[1] * 0.33,
                            limits[1] * (7.5 * 10**(-4)), limits[1] * (4.5 * 10**(-3)),
                            # [8, 9, 10, 11, 12]
                            limits[1] * 0.14, limits[1] * 12, limits[1] * 0.5, # Activation mK; 1 / KmK (0.14), vmK (12), cs[2] (0.5)
                            limits[1] * (7.5 * 10**(-4)), limits[1] * (5.0 * 10**(-3)), # cs[3] (7.5e-4), cs[4] (5.0e-3)
                            # [13, 14]
                            limits[1] * 1700, limits[1] * 3,
                            # [15, 16, 17, 18, 19, 20]
                            limits[1] * 0.033, limits[1] * 28.3, limits[1] * 0.044,
                            limits[1] * 46, limits[1] * 0.1806, limits[0] * -0.1502,
                            # [21, 22, 23]
                            limits[1] * 0.15, limits[1] * 23, limits[1] * 285.7,
                            # [24, 25, 26]
                            limits[0] * -0.083, limits[1] * 59, limits[1] * 10.5,
                            # [27, 28, 29]
                            limits[1] * 0.2, limits[1] * 50, limits[1] * 403,
                            # [30, 31, 32]
                            limits[1] * 800, limits[1] * 3, limits[1] * 25,
                            # [33, 34, 35, 36, 37, 38, 39]
                            limits[1] * 80, limits[1] * 140, limits[1] * 0.25, 
                            limits[1] * 3.5, limits[1] * 0.31, limits[1] * 6,
                            200,
                            # [40, 41]
                            1, 1,
                            # [42, 43]
                            limits[1] * 700, limits[1] * (1/10),
                            # [44, 45, 46, 47, 48]
                            limits[1] * 0, limits[1] * 1, 1,
                            273.15 + 30, limits[1] * (1 / 0.002),

                            # [49, 50]
                            200, 200, 200, 200, 200
                            ]
                            )
        self.pool = pool
    def _evaluate(self, x, out, *args, **kwargs):
        # Conditions
        conditions = [30]#[20, 25, 30, 35, 40]
        # Create Pool
        # Prepare arguments as (x, condition) pairs for each combination of x and condition
        args = [([condition], x[i_x, :]) for i_x in range(x.shape[0]) for condition in conditions]

        # Evaluate all combinations in parallel using starmap
        frqs = self.pool.starmap(solve_model, args)

        # Map back
        frqs = np.array(frqs).reshape(x.shape[0], len(conditions))
        out["frq"] = list(frqs)
        fitness_value = ((frqs - np.array([11]))**2).sum(axis = 1) #[0.2, 8.9, 11.0, 8.0, 0.1]
        out["F"] = list(fitness_value)
        

class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.generation = 0
    
    def notify(self, algorithm):
        self.generation += 1
        pop = algorithm.pop
        solutions = pop.get("X")
        frqs = pop.get("frq")
        fitness_values = pop.get("F")
        
        # Save results to the database
        for sol, frq, fit in zip(solutions, frqs, fitness_values):
            save_to_database(self.generation, sol, frq, fit[0])


# def evaluate_parallel(solutions):
#     """Evaluate solutions in parallel."""
#     with multiprocessing.Pool() as pool:
#         return pool.map(MyProblem, solutions)


if __name__ == "__main__":
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = pool_size)


    DB_NAME = "results.db"

    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"The database {DB_NAME} has been deleted.")
    else:
        print(f"The database {DB_NAME} does not exist.")

    create_database(DB_NAME)
    
    callback = MyCallback()

    # Set up algorithm
    algorithm = GA(pop_size=40,
                sampling=FloatRandomSampling(),
                repair = MyRepair(),
                eliminate_duplicates=False)

    # Run optimization
    res = minimize(MyProblem(pool=pool),
                algorithm,
                seed=0,
                verbose=True,
                termination=('n_gen', 20),
                callback = callback,
    )


    pool.close()
    pool.join()