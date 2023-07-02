from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from .gates import qft


def phase_estimation(
    n_qubits_first: int, n_qubits_second: int, unitary: QuantumCircuit,
    measurements: bool=False
) -> QuantumCircuit:
    """
    Construct the circuit for phase estimation. The circuit uses
    2 register: in the first register the input state is a bunch
    of |0> qubits, in the second one a non-trivial eigenvector.
    Then a number of Hadamard gates are applied to each qubits
    of the first registers and a Bunch of Controlled-U^2^j unitary
    operators are applied to the second register and controlled by
    each qubit of the first.
    
    Given a unitary operator U with eigenvector |u> and corresponding
    eigenvalue e^(2*π*i*φ), the goal of the phase estimation is to
    estimate the value of the global phase φ.
    
    :param n_qubits_first: the number of qubits in the first register
    :param n_qubits_second: the number of qubits in the second register
    :return: The corresponding quantum circuit
    """
    # Define the two initial quantum register
    qreg1_q = QuantumRegister(n_qubits_first, "reg1")
    qreg2_q = QuantumRegister(n_qubits_second, "reg2")
    
    # Instantiate the quantum circuit
    q_circuit = QuantumCircuit(qreg1_q, qreg2_q)
    
    # First apply Hadamard gate to all the qubits of the first register
    q_circuit.h(qreg1_q)
    q_circuit.x(qreg2_q[-1])
    q_circuit.barrier(qreg1_q, qreg2_q)
    
    # Then apply the controlled-U gate on the second register
    # The idea of the controlled-U gate is that, it leaves the
    # second register in the same state-vector while applying
    # the operator on each input of the first register
    q_circuit.append(unitary, qreg1_q[:] + qreg2_q[:])
        
    # Finally apply the Inverse Quantum Fourier Transform
    # However we do not want to apply the swap gates at the end
    q_circuit.compose(
        qft(n_qubits_first, inverse=True).reverse_bits(), 
        qreg1_q[:], inplace=True)
    
    if measurements:
        # Define the classical register and add the measurements
        cls_reg = ClassicalRegister(n_qubits_first, name="c")
        q_circuit.add_register(cls_reg)
        q_circuit.measure(qreg1_q, cls_reg)
    
    return q_circuit