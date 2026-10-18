# Experiment Checklist: When Real Organoid Arrives

This checklist must be executed in order. Do NOT skip ahead.
Each gate must pass before proceeding to the next section.

---

## A. Hardware Acceptance (before any biology)

### A1. Intan RHS controller power-on
- [ ] Controller recognized by USB (version 3.2+ for USB-C)
- [ ] RHX software starts, selects correct controller
- [ ] Sample rate set to 30 kS/s, stimulation step size selected

### A2. TCP connection
- [ ] Run `intan_interface/demo_validation.py` — all commands return valid responses
- [ ] `Get RunMode` returns `Stop`
- [ ] `Set RunMode Run` / `Stop` cycle works
- [ ] `Get SampleRateHertz` returns `30000`

### A3. Stimulation parameter verification with oscilloscope
- [ ] Connect oscilloscope probe to a stimulation output channel ACROSS A KNOWN DUMMY LOAD (e.g., 10–100 kΩ resistor)
- [ ] Send `protocol_to_commands()` output for a test StimulusProtocol via TCP
- [ ] Trigger stimulation
- [ ] **Verify on oscilloscope**: biphasic CURRENT pulse, cathodic-first, default 10 µA × R_load voltage drop, ~500 µs per phase
- [ ] Repeat for 3 different channels — all show correct waveform
- [ ] Verify inter-pulse timing matches protocol within ±1 ms
- [ ] **Note**: V = I × R_load only applies to a pure-resistance dummy load. Real electrodes/tissue have complex impedance (capacitive + resistive + nonlinear), so voltage waveform shape may not directly mirror current.

**GATE A: Do not proceed unless oscilloscope confirms correct waveform shape and timing.**

---

## B. Biological Viability (before any experiments)

### B1. Organoid health check
- [ ] Culture age ≥ 10 DIV (days in vitro) — Wang et al. criterion
- [ ] Visual inspection under microscope: cell adhesion, no contamination
- [ ] Supply voltage (VDD) on all headstages: 3.2–3.6 V (green indicator in RHX)

### B2. Spontaneous activity baseline
- [ ] Record 5 minutes of spontaneous activity (no stimulation)
- [ ] At least 30% of recording channels show detectable spikes
  (threshold: 5.5× SD of background noise — Wang et al. criterion)
- [ ] Observe synchronized burst activity at short intervals
- [ ] Save this recording as the pre-training baseline

### B3. Evoked response confirmation
- [ ] Deliver a single biphasic test pulse (10 µA, 500 µs/phase, cathodic-first) on one channel
- [ ] Confirm evoked firing response within 200 ms post-stimulus
- [ ] Repeat 10 times — responses should be visually consistent
- [ ] Record firing rate: should be significantly higher than spontaneous

**GATE B: Do not proceed unless spontaneous and evoked activity are confirmed.**

---

## C. Stimulation Protocol Validation (before classification experiments)

### C1. Single-class stimulation
- [ ] Pick one MNIST digit, encode with `delayed_phase` encoder
- [ ] Deliver encoded StimulusProtocol 10 times, 15 s inter-trial interval
- [ ] Record all 10 evoked responses
- [ ] Compute per-trial firing counts — coefficient of variation < 0.5
  (if CV > 0.5: stimulation is too variable, check parameters)

### C2. Multi-class stimulus separability
- [ ] Encode 3 different MNIST digits
- [ ] Deliver each 10 times (30 total stimulations, randomized order)
- [ ] Extract time-bin features from evoked responses
- [ ] Plot PCA of all 30 feature vectors, colored by class
- [ ] **Visual check**: are the 3 classes forming distinct clusters?
- [ ] If NOT separable: stop. The encoding or organoid is not working.

### C3. Sustained viability
- [ ] After 30 stimulations, record 5 min spontaneous activity again
- [ ] Compare with B2 baseline: firing rate should not drop by >50%
  (if it drops: overstimulation — increase inter-trial interval)

**GATE C: Do not proceed unless evoked responses are repeatable and visually class-separable.**

---

## D. Classification Experiment (the actual experiment)

### D1. Full protocol (Wang et al. design)
- [ ] Select N classes (start with 3, scale to 10 only if 3 works)
- [ ] 50 repetitions per class, 15 s inter-trial interval
- [ ] Record all evoked responses
- [ ] Save pre-training and post-training spontaneous activity (5 min each)

### D2. Decode and report
- [ ] Extract features (time-bin, 200 ms window)
- [ ] Run 5-fold cross-validation with: LR, SVM, SNN decoder
- [ ] Report mean ± SD accuracy for each
- [ ] Report confusion matrices
- [ ] Run paired t-test between decoder pairs
- [ ] **Do NOT cherry-pick folds or retune decoders**

### D3. Honest assessment
- [ ] Is accuracy above chance (1/N_classes) with p < 0.05?
- [ ] If not: report as negative result. Do not adjust data or methods.
- [ ] If yes: report the delta above chance, the decoder, the feature type.
- [ ] Explicitly state: this was the first run, or Nth attempt.

---

## E. What to Record in Lab Notebook (non-negotiable)

For every experiment session:
- [ ] Date, organoid ID, culture age (DIV)
- [ ] Intan controller serial, sample rate, stim step size
- [ ] Which encoder was used, with what parameters
- [ ] Number of classes, repetitions per class, inter-trial interval
- [ ] Any deviations from this checklist (and why)
- [ ] Raw data file paths
- [ ] Any observed anomalies (noise, artifacts, electrode failures)
