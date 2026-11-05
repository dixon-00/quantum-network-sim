"""
Quantum part of the simulation, written in PennyLane.

We use the "default.mixed" device, which tracks density matrices, so noise
(lost coherence, random errors) can be modelled exactly instead of by
averaging many noisy runs.

Building blocks
---------------
prepare_pair      : an entangled pair |Phi+> = (|00> + |11>)/sqrt(2), degraded to a
                    Werner state with weight w (w = 1 perfect, w = 0 pure noise)
prepare_bell_diag : any mixture of the four Bell states (needed for purification)
dephase           : a quantum memory losing coherence while it waits
entanglement_swap : Bell measurement at a relay + classically controlled corrections

Metrics (computed from PennyLane measurements)
-------
fidelity with |Phi+>, the CHSH value S, QKD error rates (QBER) and key rate.
"""

import numpy as np
import pennylane as qml

DEV = "default.mixed"

# Pauli matrices and Bell states (used for checks and for reading out states)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)

BELL = {
    "phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
    "phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
    "psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
    "psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
}


# ---------------------------------------------------------------------------
# Quantum functions (composable pieces of circuits)
# ---------------------------------------------------------------------------
def prepare_pair(w, a, b):
    """Make a Bell pair on wires a, b, then send b through a depolarising channel.

    Depolarising one half of |Phi+> with probability p gives a Werner state
        rho = w |Phi+><Phi+| + (1 - w) I/4,   with   w = 1 - 4p/3.
    So for a target weight w we use p = 3(1 - w)/4.
    """
    qml.Hadamard(wires=a)
    qml.CNOT(wires=[a, b])
    p = 3 * (1 - w) / 4
    if p > 0:
        qml.DepolarizingChannel(p, wires=b)


def prepare_bell_diag(coeffs, a, b):
    """Prepare sum_k c_k |B_k><B_k| for coeffs = (phi+, psi-, psi+, phi-).

    Trick: start from |Phi+> and apply a random Pauli to qubit b:
    I keeps Phi+, Z gives Phi-, X gives Psi+, Y gives Psi- (up to a global phase).
    """
    c_phip, c_psim, c_psip, c_phim = np.clip(coeffs, 0, None)
    qml.Hadamard(wires=a)
    qml.CNOT(wires=[a, b])
    kraus = [np.sqrt(c_phip) * I2, np.sqrt(c_phim) * Z,
             np.sqrt(c_psip) * X, np.sqrt(c_psim) * Y]
    qml.QubitChannel(kraus, wires=b)


def dephase(wait_time, T2, wires):
    """Memory decoherence: off-diagonal terms shrink by exp(-t/T2) on each qubit.

    PennyLane's PhaseDamping(gamma) multiplies coherences by sqrt(1 - gamma),
    so gamma = 1 - exp(-2 t / T2).
    """
    if wait_time <= 0 or np.isinf(T2):
        return
    gamma = 1 - np.exp(-2 * wait_time / T2)
    for wire in wires:
        qml.PhaseDamping(gamma, wires=wire)


def entanglement_swap(m1, m2, target):
    """Bell-state measurement on (m1, m2) and corrections on `target`.

    This is teleportation of qubit m1's entanglement onto `target`: the relay
    measures, sends two classical bits, and the far node applies X and/or Z.
    """
    qml.CNOT(wires=[m1, m2])
    qml.Hadamard(wires=m1)
    r1 = qml.measure(m1)
    r2 = qml.measure(m2)
    qml.cond(r2, qml.PauliX)(wires=target)   # classical bit 2 -> X correction
    qml.cond(r1, qml.PauliZ)(wires=target)   # classical bit 1 -> Z correction


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def correlator_obs(a, b):
    """The three two-qubit correlations that fully characterise a Bell-diagonal state."""
    return [qml.X(a) @ qml.X(b), qml.Y(a) @ qml.Y(b), qml.Z(a) @ qml.Z(b)]


def chsh_obs(a, b):
    """Observables for the CHSH test with the optimal settings for |Phi+>.

    Alice measures Z or X; Bob measures (Z+X)/sqrt2 or (Z-X)/sqrt2.
    S = <A0 B0> + <A0 B1> + <A1 B0> - <A1 B1>.
    Classical (local hidden variable) theories give |S| <= 2; quantum allows up to 2*sqrt2.
    """
    B0 = (qml.Z(b) + qml.X(b)) / np.sqrt(2)
    B1 = (qml.Z(b) - qml.X(b)) / np.sqrt(2)
    return [qml.Z(a) @ B0, qml.Z(a) @ B1, qml.X(a) @ B0, qml.X(a) @ B1]


def chsh_from_terms(t):
    return t[0] + t[1] + t[2] - t[3]


# ---------------------------------------------------------------------------
# Ready-made circuits
# ---------------------------------------------------------------------------
def make_link_circuit(measure="dm", shots=None):
    """Single link A-B. Returns a function f(w, wait=0, T2=inf)."""
    dev = qml.device(DEV, wires=2)

    @qml.qnode(dev, shots=shots)
    def circuit(w, wait=0.0, T2=np.inf):
        prepare_pair(w, 0, 1)
        dephase(wait, T2, [0, 1])
        return _measure(measure, 0, 1)

    return circuit


def make_swap_circuit(measure="dm", shots=None):
    """Two links A-M and M-B joined by swapping at relay M.

    Wires: 0 = A, 1 and 2 = relay M, 3 = B.
    Returns f(w1, w2, wait=0, T2=inf, w_bsm=1): w_bsm < 1 adds noise to the relay's
    Bell measurement (imperfect two-qubit gate).
    """
    dev = qml.device(DEV)

    @qml.qnode(dev, shots=shots)
    def circuit(w1, w2, wait=0.0, T2=np.inf, w_bsm=1.0):
        prepare_pair(w1, 0, 1)
        prepare_pair(w2, 2, 3)
        # A and B store their qubits while the relay result travels to them
        dephase(wait, T2, [0, 3])
        if w_bsm < 1:
            qml.DepolarizingChannel(3 * (1 - w_bsm) / 4, wires=1)
        entanglement_swap(1, 2, 3)
        return _measure(measure, 0, 3)

    return circuit


def make_chain_circuit(n_links, measure="dm", shots=None):
    """A repeater chain of n_links identical links, swapped at every relay.

    Wires come in pairs (2k, 2k+1) for link k. The end-to-end pair sits on
    wire 0 (first node) and wire 2*n_links-1 (last node).
    """
    dev = qml.device(DEV)
    end = 2 * n_links - 1

    @qml.qnode(dev, shots=shots)
    def circuit(w):
        for k in range(n_links):
            prepare_pair(w, 2 * k, 2 * k + 1)
        # swap sequentially along the chain: each swap pushes the entanglement right
        for k in range(1, n_links):
            entanglement_swap(2 * k - 1, 2 * k, 2 * k + 1)
        return _measure(measure, 0, end)

    return circuit


def _measure(kind, a, b):
    if kind == "dm":
        return qml.density_matrix(wires=[a, b])
    if kind == "corr":
        return [qml.expval(o) for o in correlator_obs(a, b)]
    if kind == "chsh":
        return [qml.expval(o) for o in chsh_obs(a, b)]
    if kind == "samples_zz":
        return qml.sample(wires=[a, b])
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Purification (DEJMPS protocol, Deutsch et al. 1996)
# ---------------------------------------------------------------------------
_dev_pur = qml.device(DEV, wires=4)


@qml.qnode(_dev_pur)
def _purify_circuit(coeffs1, coeffs2):
    # pair 1 = (0, 1), pair 2 = (2, 3). Alice holds 0 and 2, Bob holds 1 and 3.
    prepare_bell_diag(coeffs1, 0, 1)
    prepare_bell_diag(coeffs2, 2, 3)
    # local rotations (they make the protocol work on non-Werner states)
    qml.RX(np.pi / 2, wires=0)
    qml.RX(np.pi / 2, wires=2)
    qml.RX(-np.pi / 2, wires=1)
    qml.RX(-np.pi / 2, wires=3)
    # bilateral CNOT: each side CNOTs its pair-1 qubit onto its pair-2 qubit
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 3])
    return qml.density_matrix(wires=[0, 1, 2, 3])


def purify(coeffs1, coeffs2=None):
    """One round of DEJMPS. Alice and Bob measure pair 2 in Z and keep pair 1 only
    if their results agree. Returns (new Bell coefficients, success probability)."""
    if coeffs2 is None:
        coeffs2 = coeffs1
    rho = np.asarray(_purify_circuit(np.asarray(coeffs1), np.asarray(coeffs2)))
    rho = rho.reshape([2] * 8)
    # keep outcomes (wire2, wire3) = (0,0) or (1,1); trace out wires 2 and 3
    kept = sum(rho[:, :, s, s, :, :, s, s] for s in (0, 1)).reshape(4, 4)
    p_success = np.real(np.trace(kept))
    return bell_coefficients(kept / p_success), p_success


def dejmps_theory(c):
    """Textbook DEJMPS map for coefficients (phi+, psi-, psi+, phi-)."""
    A, B, C, D = c
    N = (A + B) ** 2 + (C + D) ** 2
    return np.array([A**2 + B**2, 2 * C * D, C**2 + D**2, 2 * A * B]) / N, N


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bell_coefficients(rho):
    """Weights of (phi+, psi-, psi+, phi-) in a two-qubit density matrix."""
    rho = np.asarray(rho)
    return np.real([BELL[k].conj() @ rho @ BELL[k] for k in ("phi+", "psi-", "psi+", "phi-")])


def fidelity(rho):
    """Fidelity with the target |Phi+>: F = <Phi+| rho |Phi+>."""
    return float(np.real(BELL["phi+"] @ np.asarray(rho) @ BELL["phi+"]))


def fidelity_from_corr(xx, yy, zz):
    """Same fidelity, from measured correlations: F = (1 + <XX> - <YY> + <ZZ>)/4."""
    return (1 + xx - yy + zz) / 4


def werner_fidelity(w):
    return (1 + 3 * np.asarray(w)) / 4


def h2(p):
    """Binary entropy in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def qber(xx, zz):
    """Error rates of entanglement-based QKD (BBM92) in the Z and X bases."""
    return (1 - zz) / 2, (1 - xx) / 2


def bbm92_key_fraction(q_z, q_x, f_ec=1.1):
    """Asymptotic secret key per sifted coincidence (Shor-Preskill style bound).

    1 - f_ec*h(Q_Z) - h(Q_X): error correction leaks f_ec*h(Q_Z), privacy
    amplification removes h(Q_X). Negative means no secure key.
    """
    return np.maximum(0.0, 1 - f_ec * h2(q_z) - h2(q_x))


def key_rate(coincidence_rate, q_z, q_x, f_ec=1.1):
    """Secret bits per second. Half of coincidences survive basis sifting."""
    return 0.5 * coincidence_rate * bbm92_key_fraction(q_z, q_x, f_ec)


# ---------------------------------------------------------------------------
# Teleportation
# ---------------------------------------------------------------------------
_dev_tel = qml.device(DEV, wires=3)


@qml.qnode(_dev_tel)
def teleport(theta, phi, w, wait=0.0, T2=np.inf):
    """Teleport RZ(phi)RY(theta)|0> from wire 0 to wire 2 over a Werner pair (1, 2)."""
    qml.RY(theta, wires=0)
    qml.RZ(phi, wires=0)
    prepare_pair(w, 1, 2)
    dephase(wait, T2, [1, 2])
    entanglement_swap(0, 1, 2)
    return qml.density_matrix(wires=2)


def teleport_fidelity(theta, phi, w, **kw):
    psi = np.array([np.cos(theta / 2) * np.exp(-1j * phi / 2),
                    np.sin(theta / 2) * np.exp(1j * phi / 2)])
    rho = np.asarray(teleport(theta, phi, w, **kw))
    return float(np.real(psi.conj() @ rho @ psi))


# ---------------------------------------------------------------------------
# BB84 (prepare-and-measure) with an intercept-resend eavesdropper
# ---------------------------------------------------------------------------
def bb84(n_rounds, eve_fraction=0.0, channel_p=0.0, seed=0):
    """Simulate BB84 round by round, grouped by settings so PennyLane can sample in bulk.

    Alice sends a random bit in a random basis (Z or X). With probability
    `eve_fraction` Eve measures it in a random basis and resends what she saw.
    The channel depolarises with probability `channel_p`. Bob measures in a random basis.
    Returns sifted QBER and the sifted key length.
    """
    rng = np.random.default_rng(seed)
    a_bit = rng.integers(0, 2, n_rounds)
    a_basis = rng.integers(0, 2, n_rounds)
    b_basis = rng.integers(0, 2, n_rounds)
    eve_on = rng.random(n_rounds) < eve_fraction
    e_basis = rng.integers(0, 2, n_rounds)

    b_bit = np.empty(n_rounds, int)
    configs = np.stack([a_bit, a_basis, b_basis, eve_on, e_basis * eve_on], axis=1)
    for cfg in np.unique(configs, axis=0):
        idx = np.where((configs == cfg).all(axis=1))[0]
        b_bit[idx] = _bb84_circuit(*cfg, channel_p=channel_p, shots=len(idx), seed=rng)

    sifted = a_basis == b_basis
    errors = (a_bit != b_bit) & sifted
    return dict(qber=errors.sum() / sifted.sum(), sifted=int(sifted.sum()))


def _bb84_circuit(a_bit, a_basis, b_basis, eve_on, e_basis, channel_p, shots, seed):
    dev = qml.device(DEV, seed=int(seed.integers(1 << 31)))

    @qml.qnode(dev, shots=int(shots))
    def circuit():
        if a_bit:
            qml.PauliX(wires=0)
        if a_basis:
            qml.Hadamard(wires=0)
        if eve_on:   # Eve measures in her basis; the measurement collapses the photon
            if e_basis:
                qml.Hadamard(wires=0)
            qml.measure(0)
            if e_basis:
                qml.Hadamard(wires=0)
        if channel_p > 0:
            qml.DepolarizingChannel(channel_p, wires=0)
        if b_basis:
            qml.Hadamard(wires=0)
        return qml.sample(wires=0)

    return np.asarray(circuit()).reshape(-1)
