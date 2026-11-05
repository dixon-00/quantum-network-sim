# Entanglement distribution over a small quantum network (PennyLane)
Dickson Marfo Fosu, KNUST (QCAT Lab)

A stage-by-stage simulation of how two nodes come to share *usable* entanglement over fibre,
from a single link to a relay with entanglement swapping, quantum memories, clock synchronisation,
purification, and finally QKD, Bell tests and teleportation.

The quantum states are built and measured in **PennyLane** (`default.mixed`, density matrices).
The classical physics (fibre loss, detectors, time-stamps, signalling delay) is ordinary Python.
Every quantum result is checked against a textbook formula (`tests/test_theory.py`).

> Parameter values are illustrative, typical-of-the-literature numbers. They are not the parameters
> of any specific CERN or Geneva Quantum Network link.

## Stages

| Notebook | Question | Main result (with the default parameters) |
|---|---|---|
| `01_single_link` | How do loss and noise affect one link? | Loss cuts the *rate* exponentially; quality stays flat until noise catches up, then drops off a cliff (InGaAs detectors: ~235 km; SNSPDs: beyond 300 km). |
| `02_entanglement_swapping` | Can a relay join two links? | Swapping with mid-circuit measurement and feed-forward gives w = w₁·w₂ (matches theory to 10⁻¹⁴). Three 93% links fall below the QKD threshold. Without memories, a relay does not improve rate. |
| `03_memory_and_classical_delay` | What does waiting for classical messages cost? | Memory dephasing only causes X-basis errors. With retries, T₂ = 10 ms keeps QKD working to ~36 km and 100 ms to ~94 km. |
| `04_timing_synchronisation` | How precise must clock sync be? | A Monte Carlo time-tagger agrees with the analytic model. With brightness and window optimised, at 25 km: 10 ps (White Rabbit-like) ≈ 1.8 Mbit/s, 1 ns ≈ 1.8× lower, 10 ns ≈ 16× lower. Tight timing also tolerates background from a co-propagating classical channel. |
| `05_purification` | Can two noisy pairs make one better pair? | The DEJMPS circuit reproduces the textbook map exactly. Two rounds rescue the 3-hop chain (F 0.81 → 0.95) at ~7 raw pairs per final pair. |
| `06_using_the_link` | What is a pair good for? | CHSH, BBM92 QBER, teleportation (2F+1)/3 and BB84 with intercept-resend (QBER = fraction/4), all with finite shots and matching theory. |

Figures for each stage are saved in `figures/`.

**Overall message.** Whether two nodes can share usable entanglement depends as much on the *classical* side
(timing precision, offset calibration, signalling delay, retry and purification policy) as on the quantum hardware.
The natural next step is to study that classical control plane at network scale, for example in ns-3 with Q2NS.

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 10 checks against theory, ~2 s
```

Then open the notebooks in VS Code (or Jupyter) from this folder and run all cells. Each notebook takes a few seconds.
The notebooks import the shared code from `qnet/`, so open them from the project folder.

## Code layout

```
qnet/physics.py    fibre loss, detectors, coincidence window, accidentals, Monte Carlo time-tagger
qnet/quantum.py    PennyLane circuits: noisy Bell pairs, swapping, memory dephasing, DEJMPS,
                   CHSH, BBM92, teleportation, BB84; plus fidelity/QBER/key-rate metrics
qnet/plotstyle.py  shared plot style
tests/             simulation-vs-theory checks
```

## Model summary

* **Link state:** Werner state with weight w = w_source × true/(true + accidental).
* **Accidentals:** singles_A × singles_B × window; true pairs captured by a Gaussian of width
  √(2·detector jitter² + sync jitter²), shifted by any uncorrected clock offset.
* **Memory:** phase damping with coherence factor e^(−t/T₂) per qubit; wait = (A–B distance)/c per attempt,
  divided by the half-link success probability when retries are included (an average, not a full distribution).
* **Key rate:** asymptotic BBM92, ½ × coincidences × [1 − 1.1·h(Q_Z) − h(Q_X)]. No finite-key effects.
* **Not modelled:** multi-photon emission beyond its contribution to accidentals, detector dead time
  (brightness is capped at 2×10⁷ pairs/s instead), polarisation drift, and finite-key security.

## References

* Deutsch et al., "Quantum privacy amplification and the security of quantum cryptography over noisy channels", PRL 77, 2818 (1996), the DEJMPS protocol.
* Bennett, Brassard and Mermin, "Quantum cryptography without Bell's theorem", PRL 68, 557 (1992), BBM92.
* Briegel, Dür, Cirac and Zoller, "Quantum repeaters", PRL 81, 5932 (1998).
* CERN, "White Rabbit meets quantum entanglement" (2025), https://home.cern/news/news/computing/white-rabbit-meets-quantum-entanglement
