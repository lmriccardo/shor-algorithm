import numpy as np
import math

from numpy import pi
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from quantum.util.util import modular_inverse, get_angles


def qft(n_qubits: Optional[int], inverse: bool=False, swaps: bool=False) -> QuantumCircuit | None:
    """ Construct the (Inverse) Quantum Fourier Transform """
    # If an invalid number of qubits is given then return None
    if n_qubits is None or n_qubits == 0: return

    # Otherwise just build the circuit
    # First define registers for qubits and classical bits
    qreg_q = QuantumRegister(n_qubits, 'q')

    # Create the circuit and apply the quantum gates
    # Computational Complexity: O(N^2)
    label = "gQFT" if not inverse else "gIQFT"
    circuit = QuantumCircuit(qreg_q, name=label)
    
    # Qiskit's least significant bit has always the lowest index (0)
    # thus the circuit will be mirrored through the horizontal.
    # Hence, first we need to apply Hadamard on the qubits 2 and the
    # controlled-P gates with control qubits 1 and then 0, and so on
    # for all the other qubits.
    for nq in reversed(range(n_qubits)):
        circuit.h(qreg_q[nq])

        # If it is the last qubit we only apply Hadamard
        if nq == 0: break

        # Otherwise we need to apply controlled-P gates
        for qi in range(0, nq):
            global_phase = pi / 2**(nq - qi)
            circuit.cp(global_phase, qreg_q[qi], qreg_q[nq])
        
    # Apply the SWAPS gates
    if swaps:
        for nq in range(n_qubits):
            # There are no more ways to do swap exit
            if nq >= n_qubits - 1 - nq: break

            # Otherwise apply the gate
            circuit.swap(qreg_q[nq], qreg_q[n_qubits - 1 - nq])
            
    # If inverse is True, then invert the circuit
    if inverse: circuit = circuit.inverse()
    
    # Create the corresponding gate
    wrapped = circuit.to_gate()
    circuit = QuantumCircuit(n_qubits)
    circuit.compose(wrapped, range(n_qubits), inplace=True, front=True)
    
    return circuit


def phi_adder(n_qubits: int, angles: np.ndarray | ParameterVector, inverse: bool=False) -> QuantumCircuit:
    """ Compose the circuit for the Adder taking as input
    two registers where in the second register is contained
    the quantum fourier transformation of the second operand.
    """
    # Define the quantum register that will contains the sum and also
    # identifies the quantum fourier transformed input B.
    b_reg = QuantumRegister(n_qubits, name="phi(b)")
    
    # Initialize the quantum circuit
    phi_circuit = QuantumCircuit(b_reg, name="φADD")
    
    # Add all the phase shift gates according to the circuit
    for bi in range(n_qubits):
        phi_circuit.p(angles[bi], b_reg[bi])
    
    suffix = ""
    if inverse:
        phi_circuit = phi_circuit.inverse()
        suffix = "_dg"
    
    phi_gate = phi_circuit.to_gate()
    q_circuit = QuantumCircuit(n_qubits, name="φADD" + suffix)
    q_circuit.append(phi_gate, range(n_qubits))
    return q_circuit


def phi_add_mod(n_qubits: int, a_angles: np.ndarray | ParameterVector, 
                N_angles: np.ndarray | ParameterVector, inverse: bool=False
) -> QuantumCircuit:
    """
    The Quantum Doubly Controlled Modular Adder φADD(a)MOD(N) from the paper
    "Circuit For Shor's Algorithm using 2n + 3 qubits" (Stephane Beauregard)
    [https://arxiv.org/pdf/quant-ph/0205095.pdf].
    
    It takes as input |φ(b)> and returns |φ((b + a) mod N)> in O(n) time.
    
    Uses a total of n + 4 qubits, 2 controller qubits, 1 ancillary and the
    input |φ(b)> that is a (n + 1)-qubits statevector. We use n + 1 qubits
    for |φ(b)> to prevent overflows before applying the QFT.
    
    :param n_qubits: The total number of qubits, i.e., n + 3
    :param a_angles: The angles used for the φADD(a) gate
    :param N_angles: The angles used for the φADD(N) gate
    :return: φADD(a)MOD(N) circuit
    """
    # In the input number of qubits there are also those relative to the
    # two controller qubits and the auxiliary bit |0> in the circuit
    # To be consistent with the structure of the paper we consider at index
    # 0 and 1 the two controlled bit c1 and c2, and at index -1 the 0 qubit.
    qubits = list(range(n_qubits))
    c1, c2 = qubits[:2]
    aux    = qubits[-1]
    pb_reg = qubits[2:-1]
    
    # Let's initialize the circuit first
    phi_add_mod_circuit = QuantumCircuit(n_qubits, name="φADD(a)MOD(N)")
    
    # Then let's define all the sub-circuit that we need. In order:
    # 1. Phi adder with input a
    # 2. Inverse phi adder with input N
    # 3. Inverse Quantum Fourier Transform
    # 4. Quantum Fourier Transform
    # 5. Phi adder with input N
    # 6. Inverse Phi Adder with input a
    phi_add_a = phi_adder(len(pb_reg), a_angles)
    iphi_add_N = phi_adder(len(pb_reg), N_angles, inverse=True)
    iqft_c = qft(len(pb_reg), inverse=True, swaps=False)
    qft_c = qft(len(pb_reg), inverse=False, swaps=False)
    phi_add_N = phi_adder(len(pb_reg), N_angles)
    iphi_add_a = phi_adder(len(pb_reg), a_angles, inverse=True)
    
    # First we have to apply the double-controlled φADD(a) and the Inv_φADD(N)
    phi_add_mod_circuit.append(phi_add_a.control(2), [c1, c2] + pb_reg)
    phi_add_mod_circuit.append(iphi_add_N, pb_reg)
    
    # Then the block: IQFT, cNOT, QFT
    phi_add_mod_circuit.append(iqft_c, pb_reg)
    phi_add_mod_circuit.cx(pb_reg[-1], aux)
    phi_add_mod_circuit.append(qft_c, pb_reg)
    
    # Then we apply the controlled φADD(N) and the CCInv_φADD(a)
    phi_add_mod_circuit.append(phi_add_N.control(1), [aux] + pb_reg)
    phi_add_mod_circuit.append(iphi_add_a.control(2), [c1, c2] + pb_reg)
    
    # Then the block: IQFT, X, CNOT, X, QFT
    phi_add_mod_circuit.append(iqft_c, pb_reg)
    phi_add_mod_circuit.x(pb_reg[-1])
    phi_add_mod_circuit.cx(pb_reg[-1], aux)
    phi_add_mod_circuit.x(pb_reg[-1])
    phi_add_mod_circuit.append(qft_c, pb_reg)
    
    # Finally the double-controlled φADD(a) gate
    phi_add_mod_circuit.append(phi_add_a.control(2), [c1, c2] + pb_reg)
    
    if inverse:
        return phi_add_mod_circuit.inverse()
    
    return phi_add_mod_circuit


def controlled_mul_mod(n_qubits: int, a: int, N: int, 
                       N_angles: np.ndarray | ParameterVector, inverse: bool=False
) -> QuantumCircuit:
    """ ... """
    # First let's divide the register that we need
    N_bits = math.ceil(math.log(N, 2))
    qubits = range(n_qubits)
    c_ctrl = qubits[0]
    x_ctrl = qubits[1:N_bits + 1]
    b_reg  = qubits[N_bits + 1:]
    z_ctrl = b_reg[-1]
    
    # Let's initialize the circuit
    cmul_mod_circuit = QuantumCircuit(n_qubits, name="CMULT(a)MOD(N)")
    
    # Let's define the angles parameter for a and the φADD(a)MOD(N) circuit
    a_angles = ParameterVector("angles", length=N_bits + 1)
    dc_phi_add_mod = phi_add_mod(N_bits + 4, a_angles, N_angles)
    
    idc_phi_add_mod = dc_phi_add_mod.inverse()
    
    # First apply the QFT on the register containing |b>
    qft_c = qft(N_bits + 1, inverse=False, swaps=False)
    cmul_mod_circuit.append(qft_c, b_reg[:N_bits + 1])
    
    # Now we need to apply the doubly-controlled Modular Adder with correct powers 
    # of 2 times a mod N. However, we need to take care of which value of a and which 
    # φADD(a)MOD(N) we are gonna use. If we are computing the classical controlled
    # multiplier modulo N then we have to use the input a and the φADD(a)MOD(N).
    # On the other hand, if we are computing the inverse of CMULT(a)MOD(N) then
    # we have to take the modular inverse of a modulo N and the inverted φADD(a)MOD(N).
    used_a = a if not inverse else modular_inverse(a, N)
    used_x_ctrl = x_ctrl if not inverse else reversed(x_ctrl)
    used_dc_phi_add_mod = dc_phi_add_mod if not inverse else idc_phi_add_mod
    for x_ctrl_i in used_x_ctrl:
        a_exp = (2 ** x_ctrl_i) * used_a % N
        a_angles_values = get_angles(a_exp, N_bits)
        assigned_dc_phi_add_mod = used_dc_phi_add_mod.assign_parameters(
            {a_angles: a_angles_values})
        cmul_mod_circuit.append(assigned_dc_phi_add_mod, 
                                [c_ctrl, x_ctrl_i] + list(b_reg[:N_bits + 1]) + [z_ctrl])
    
    # Finally we have to compute the IQFT
    iqft_c = qft(N_bits + 1, inverse=True, swaps=False)
    cmul_mod_circuit.append(iqft_c, b_reg[:N_bits + 1])
    
    if inverse:
        cmul_mod_circuit.name = f"{cmul_mod_circuit.name}_dg"
    
    return cmul_mod_circuit


def controlled_Ua(n_qubits: int, a: int, N: int, 
                  N_angles: np.ndarray | ParameterVector, inverse: bool=False
) -> QuantumCircuit:
    """ Compose the controlled-U_a gate """
    # Define all the registers
    N_bits = math.ceil(math.log(N, 2))
    qubits = range(n_qubits)
    c_ctrl = qubits[0]
    x_ctrl = qubits[1:N_bits + 1]
    b_reg  = qubits[N_bits + 1:]
    
    # Initialize the Quantum Circuit and define the additional ones
    ua_circuit = QuantumCircuit(n_qubits, name=f"c-U{a}")
    cmult_mod_circuit = controlled_mul_mod(n_qubits, a, N, N_angles, inverse=False)
    icmult_mod_circuit = controlled_mul_mod(n_qubits, a, N, N_angles, inverse=True)
    
    # First apply the controlled multiplier modulo N gate
    ua_circuit.append(cmult_mod_circuit.to_instruction(), qubits)
    
    # Then we apply the controlled-SWAP
    for qubit in range(N_bits):
        ua_circuit.cswap(c_ctrl, x_ctrl[qubit], b_reg[qubit])
        
    # Finally we applu the inverse-controlled multiplier mod N gate
    ua_circuit.append(icmult_mod_circuit.to_instruction(), qubits)
    
    if inverse:
        ua_circuit.name = f"{ua_circuit.name}_dg"
        return ua_circuit.inverse()
    
    return ua_circuit


def mod_exp(a: int, N: int) -> QuantumCircuit:
    """ Compose the circuit for the modular exponentiation """
    # Define all the variables that we requires to build the circuit
    N_bits = math.ceil(math.log(N, 2))
    n_qubits_ua = 2 * N_bits + 2
    n_qubits_ctrl = 2 * N_bits
    N_angles = get_angles(N, N_bits)
    
    # Define the quantum registers
    ctrl_reg = QuantumRegister(n_qubits_ctrl, name="ctrl")
    x_reg = QuantumRegister(n_qubits_ua, name="x")
    
    # Initialize the circuit
    mod_exp_circuit = QuantumCircuit(ctrl_reg, x_reg, name=f"{a}^j mod {N}")
    
    # Apply the U_a gate with a to the power of power of 2
    for ctrl_i, c_reg in enumerate(ctrl_reg):
        a_exp = int(math.pow(a, math.pow(2, ctrl_i)))
        controlled_ua_circuit = controlled_Ua(n_qubits_ua + 1, a_exp, N, N_angles, inverse=False)
        mod_exp_circuit.append(controlled_ua_circuit.to_instruction(), [c_reg, *x_reg])
    
    return mod_exp_circuit