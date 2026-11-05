"""Simulation vs textbook formulas. Run with:  python -m pytest -q tests"""
import numpy as np
from qnet import quantum as Q, physics as P


def test_werner_fidelity():
    link = Q.make_link_circuit("dm")
    for w in [1, 0.9, 0.5, 0.2]:
        assert abs(Q.fidelity(link(w)) - Q.werner_fidelity(w)) < 1e-9


def test_corr_matches_dm():
    link_c = Q.make_link_circuit("corr")
    xx, yy, zz = link_c(0.8)
    assert abs(Q.fidelity_from_corr(xx, yy, zz) - Q.werner_fidelity(0.8)) < 1e-9


def test_chsh_werner():
    chsh = Q.make_link_circuit("chsh")
    for w in [1, 0.8, 0.6]:
        S = Q.chsh_from_terms(chsh(w))
        assert abs(S - 2 * np.sqrt(2) * w) < 1e-9


def test_swap_multiplies_weights():
    sw = Q.make_swap_circuit("dm")
    for w1, w2 in [(1, 1), (0.9, 0.8), (0.7, 0.95)]:
        F = Q.fidelity(sw(w1, w2))
        assert abs(F - Q.werner_fidelity(w1 * w2)) < 1e-9


def test_chain():
    for n in [2, 3]:
        ch = Q.make_chain_circuit(n, "dm")
        assert abs(Q.fidelity(ch(0.9)) - Q.werner_fidelity(0.9 ** n)) < 1e-9


def test_dephasing():
    link = Q.make_link_circuit("corr")
    w, t, T2 = 0.9, 1e-3, 5e-3
    xx, yy, zz = link(w, wait=t, T2=T2)
    lam = np.exp(-2 * t / T2)
    assert abs(zz - w) < 1e-9 and abs(xx - w * lam) < 1e-9


def test_purification_matches_dejmps():
    c = np.array([0.7, 0.1, 0.1, 0.1])
    for _ in range(3):
        sim, p = Q.purify(c)
        th, N = Q.dejmps_theory(c)
        assert np.allclose(sim, th, atol=1e-9) and abs(p - N) < 1e-9
        c = sim


def test_teleport():
    w = 0.8
    Fs = [Q.teleport_fidelity(t, f, w) for t, f in [(0.3, 0.1), (1.2, 2.0), (2.5, -1)]]
    F_pair = Q.werner_fidelity(w)
    assert np.allclose(Fs, (2 * F_pair + 1) / 3, atol=1e-9)


def test_bb84_eve():
    r = Q.bb84(4000, eve_fraction=1.0, seed=3)
    assert 0.21 < r["qber"] < 0.29
    r = Q.bb84(4000, eve_fraction=0.0, seed=3)
    assert r["qber"] == 0


def test_timestamps_match_formula():
    L, win = 50, 200e-12
    mc = P.simulate_timestamps(L, 0.2, win, sync_jitter=30e-12)
    an = P.link_counts(L, win, sync_jitter=30e-12)
    assert abs(mc["true"] / an["true"] - 1) < 0.05
