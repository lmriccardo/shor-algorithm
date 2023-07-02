import math

from quantum.circuit.gates import mod_exp
from quantum.circuit.circuits import phase_estimation
from quantum.util.util import continued_fraction_expansion, euclidean_gcd, is_power
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from typing import List
from random import randint
from copy import deepcopy


def shor_factoring(N: int) -> List[int]:
    """ Algorithm for factoring that uses the Shor's algorithm ... """
    # Define the backend to simulate the circuit and retrieve the output
    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1000)
    
    def _inner_shor(a: int, N: int) -> List[float]:
        """ Inner loop for the Shor's algorithm """
        # First we need to build the circuit and add measurements
        # Create the modular exponentiation circuit with given a and N
        N_bits = N.bit_length()
        mod_exp_aN = mod_exp(a, N)
        shor_circuit = phase_estimation(2 * N_bits, 2 * N_bits + 2, mod_exp_aN, measurements=True)
        result_counts = quantum_instance.execute(shor_circuit).get_counts()
        
        # Initialize the factor list
        current_factors = []
        
        # Now for each phase we need to extract the period applying
        # the continued fraction expansion.
        for measurement in result_counts.keys():
            # We obtain possible even denominators and we try to get
            # factors of N computing the GCD(a^(denominator / 2) + 1, N)
            # and GDC(a^(denominator / 2) - 1, N). If there is at least
            # one non-trivial factor then we returns it.
            for denominator in continued_fraction_expansion(measurement, N):
                # If happened any error during continued fraction expansion
                # just break the current loop and start with a new measurement
                if denominator is None:
                    break
                    
                exponential = math.pow(a, denominator / 2)
                
                # Check if the exponential is not too big
                if exponential > 1000000000:
                    break
                    
                exp_gcd_p1 = int(exponential + 1)
                exp_gcd_m1 = int(exponential - 1)
                factor_plus, _, _ = euclidean_gcd(exp_gcd_p1, N)
                factor_minus, _, _ = euclidean_gcd(exp_gcd_m1, N)
                
                # Check if at least one of the two factors is non-trivial
                if factor_plus not in current_factors + [1, N]: current_factors.append(factor_plus)
                if factor_minus not in current_factors + [1, N]: current_factors.append(factor_minus)
                
        return current_factors
    
    print(f"Input number to be factorized: {N}")
    print("------------------------------------")
    
    factors = [] # Initialize the list that will contains all the factors
    
    # First step check if the given N is even. If it is the case
    # Then output 2 and restart with N' = N/2
    if N % 2 == 0:
        print(f"{N} is even, factor 2 will be added to factors... Restarting with {N}/2")
        factors.append(2)
        factors += shor_factoring(N // 2)
    
    # Otherwise, we search for each possible a >= 1 and b >= 2
    # such that N = a^b. If we found such an a then we will
    # output in the list many a larger is b.
    result, a, b = is_power(N, return_decomposition=True)
    if result:
        print(f"Input {N} is a power number {a}^{b}")
        return factors + [a] * b
    
    print(f"Input {N} is not a power of some number")
    print("Starting Shor Algorithm ...")
    print("============================================")
    
    # Save the current factors that have been found
    previous_factors = deepcopy(factors)
    
    # Otherwise we select a x at random between 1 and N and compute
    # the Euclidean algorithm to find the GCD between x and N.
    # Let g be GCD, if g > 1 then we output x and restart with N/g
    # Otherwise just apply the Shor algorithm. If also applying the Algorithm
    # no factor has been found, then restart with a new random x.
    controlled = {x : False for x in range(2, N)}
    while not all(controlled.values()):
        random_x = randint(2, N - 1)
        if controlled[random_x] is True:
            continue
            
        print(f"Selected Random number between 1 and {N}: {random_x}")
        controlled[random_x] = True
        
        # Compute the GCD between x and N
        gcd_x_N, _, _ = euclidean_gcd(random_x, N)
        
        # If it is non-trivial, meaning different from 1, we have found
        # a factor and we restart with N / g
        if gcd_x_N > 1 and gcd_x_N != N:
            print(f"g = GDC({random_x}, N) = {gcd_x_N} > 1 --> factor {gcd_x_N} added ... ", end="")
            print(f"Restarting with {N}/{gcd_x_N} ...")
            print("============================================\n")
            factors.append(gcd_x_N)
            factors += shor_factoring(N // gcd_x_N)
            break
            
        print(f"g = GDC({random_x}, N) = 1 --> Starting Quantum Computation ...")
            
        # Otherwise we apply the Shor's Algorithm
        try:
            shor_factors = _inner_shor(random_x, N)
            if len(shor_factors) == 0:
                print("No new factors have been found. Trying with another random value ...")
                continue

            factors += shor_factors
            break
        except ValueError as ve:
            print(ve)
            continue
            
        print("No new factors have been found. Trying with another random value ...")
            
    # In the case no new factors have been inserted in the factors list
    # This means that the current N must be a prime number. At this
    # point we need to put it into the factor list.
    if len(previous_factors) == len(factors):
        factors.append(N)
    
    return factors