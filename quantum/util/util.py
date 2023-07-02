import numpy as np
import math
import fractions

from qiskit import QuantumCircuit
from typing import Tuple, Generator


def apply_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """ Apply measurements to the input quantum circuit """
    n_qubits = circuit.num_qubits
    
    # Create the circuit with only measurements
    meas = QuantumCircuit(n_qubits,n_qubits)
    meas.barrier(range(n_qubits))
    meas.measure(range(n_qubits), range(n_qubits))
    qc = meas.compose(circuit, range(n_qubits), front=True)
    
    return qc


def encode(number: int, req_bits: int | None=None) -> QuantumCircuit:
    """ Encode a general n bit number in the computational basis
    and returns the circuit representing that number """
    # First encode in binary and then compute the number of bits
    # The number of qubits is equal to the number of bits
    n_bits = number.bit_length()
    
    # If req_bits is not None then we have to extend the number
    # of bits to represent the number to the required number of bits
    bin_repr = bin(number)[2:]
    if req_bits is not None and n_bits < req_bits: 
        n_bits = req_bits
        bin_repr = bin_repr.rjust(req_bits)
    
    # Create the circuit and apply the X-gate whenever there is a 1
    e_circuit = QuantumCircuit(n_bits)
    for n_bit in range(n_bits):
        if bin_repr[n_bit] == '1':
            e_circuit.x(n_bits - 1 - n_bit)
    
    return e_circuit


def get_angles(a: int | float, n: int) -> np.ndarray:
    """
    Compute the angles that will be used in the Phi Adder circuit to perform the addition 
    in the Fourier Basis. Given a number A, let its binary representation be A1A2...Am.
    When we extend its representation to N bits, we obtain something like: 
    00...0A1A2...Am where we will have a prefix of N - M zeros. In composing the vector
    of angles, whenever we see a 0 bit in the binary representation the corresponding
    angle, wrt to the other operand, will be 0 as well, otherwise it will be pi / 2^j 
    for some j carefully selected. 
    
    If we look at the Phi Adder presented in https://arxiv.org/pdf/quant-ph/0205095.pdf
    (Circuit For Shor's Algorithm using 2n + 3 qubits), Figure 2, for example taking the
    qubit phi_L(b) we see it will apply the controlled phase shift for a0, a1, ...,
    aL, but since it is controlled the P-gate will be applied only if aJ is |1>. At the
    end, we have N sequential P-gate applied to the phase of phi_L(b), meaning that
    its phase will shift by the angle 
    
    $$\frac{\pi}{2^k} = \pi \cdot \sum_{j \in [0, L]\;|\; a[j]=|1\rangle}2^{j - 1}$$
    
    :param a: An input number
    :param n: The total number of bits to extend the representation of a
    :return: A vector of angles associated with the binary representation of a
    """
    # First we need to extend the binary representation of the input
    # number a with the correct number of bit given as input.
    s = bin(int(a))[2:].zfill(n + 1)
    
    # Initialize the vector with all the angles
    angles = np.zeros([n + 1])
    
    # Fill the angles vector with the correct angles. In this implementation
    # we are first iterating along each 
    for i in range(0, n + 1):
        for j in range(i, n + 1):
            if s[j] == '1':
                angles[n - i] += math.pow(2, -(j - i))
                
        angles[n - i] *= np.pi
        
    return angles[::-1]


def euclidean_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """ Compute the GCD using the Euclidean Algorithm """
    # In the most trivial case, i.e., when a == 0 then we need
    # to return b, then 0 and 1
    if a == 0: return b, 0, 1

    # Otherwise we recursively apply the algorithm
    g, y, x = euclidean_gcd(b % a, a)
    return g, x - (b // a) * y, y


def modular_inverse(a: int, N: int) -> int:
    """ Return the Modular inverse of a mod N according
    according to the Euclidean algorithm """
    g, x, _ = euclidean_gcd(a, N)
    
    # If g != 1 then the modular inverse does not exists
    if g != 1:
        raise ValueError(f"Modular inverse of {a} mod {N} does not exists")
    
    return x % N


def is_power(number: int, return_decomposition: bool=False) -> bool | Tuple[bool, int, int]:
    """ Check if a number is a perfect power in O(n^3) time with n = ceil(log N) """
    # We need a >= 1 and b >= 2, hence we start with b = 2
    b_value = 2
    while (2 ** b_value) <= number:
        a_value = 1
        c_value = number
        while (c_value - a_value) >= 2:
            base = int((a_value + c_value) / 2)
            
            power_value = int((base ** b_value))
            if not (base ** b_value) < number + 1:
                power_value = int(number + 1)
            
            # Check if the power_value is exactly the input number
            if power_value == number:
                if return_decomposition:
                    return True, base, b_value
                
                return True
            
            # If it is not then we continue to search
            if power_value < number:
                a_value = int(base)
                continue
            
            c_value = int(base)
        
        # Increase the value b
        b_value = b_value + 1
    
    # If nothing has been found, return False
    if return_decomposition:
        return False, number, 1
    
    return False


def continued_fraction_expansion(measurement: str, N: int) -> Generator[float, None, None]:
    """ Compute the continue fraction expansion iteratively
    and every time it yields a denominator value. """
    # First convert to decimal value the measurement
    measurement_dec = int(measurement, 2)
    fail_reason = None
    
    # If the value is negative, there are no continued fractions
    if measurement_dec <= 0: 
        return
        
    # Compute the value for r and s/r = period
    r_nbit = len(measurement)
    r = math.pow(2, r_nbit)
    meas_over_r = measurement_dec / r
    
    # Cycle in which each iteration corresponds to putting one more term in the
    # calculation of the Continued Fraction of M/r
    # Initialize the first values according to the CF rule
    counter = 0
    b_list = [math.floor(meas_over_r)]
    t_list = [meas_over_r - b_list[0]]
    while counter < N and fail_reason is None:
        # From the second iteration we compute the new terms
        if counter > 0:
            b_list.append(math.floor(1 / t_list[counter - 1]))
            t_list.append((1 / t_list[counter - 1]) - b_list[counter])
        
        # Compute the denominator
        # Compute the continued fraction with the given expansion
        x_over_t = 0
        for i in reversed(range(len(b_list) - 1)):
            x_over_t = 1 / (b_list[i + 1] + x_over_t)
        
        x_over_t = x_over_t + b_list[0]
        
        # Get the denominator
        frac = fractions.Fraction(x_over_t).limit_denominator()
        denominator = frac.denominator
        counter = counter + 1
        
        # If the denominator is odd we have to go on because we cannot use it
        if denominator % 2 == 1 or denominator >= 1000:
            continue
            
        # Check if the number has already been found
        if t_list[counter - 1] == 0:
            return
        
        yield denominator