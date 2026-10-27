# Demo bench commands

Run from the repository root. The user's Windows baseline passed all 201 existing tests.

## 1. Stimulation commands and response reading together (no lab hardware)

```powershell
python -m pytest -v tests/test_rhx_session.py::test_prepared_profile_executes_all_pulses_and_checks_cancellation tests/test_openloop_live_transport.py::test_physical_path_keeps_collecting_through_blocking_stimulation_schedule tests/test_openloop_live_transport.py::test_frozen_inference_reaches_arm_adapter_with_real_backend_socket_fixture tests/test_openloop_live_transport.py::test_missing_stim_marker_fails_session_instead_of_decoding_zeros --basetemp .pytest_joint_01
```

Expect 4 passed. These exercise command scheduling, simultaneous 5001/5002 parsing, trial windows,
frozen inference with a simulated arm, and refusal when a pulse marker is missing. The fixture uses
an in-memory command controller and real local socketpairs. It does not connect to the open RHX app
or prove physical current. RHX Demo spontaneous spikes are not evidence of a response to stimulation.
The pytest directory is disposable and may be cleared on reuse; keep experimental data elsewhere.

## 2. Number console

No hardware:

```powershell
python virtual_arm_test.py --interactive
```

Physical arm (replace COM4 with the actual STM32 port):

```powershell
python virtual_arm_test.py --interactive --live --port COM4 --home
```

Enter one number and press Enter; wait for completion before the next command.
1 left, 2 right, 3 close gripper, 4 open gripper, 5 up, 6 down, 7 forward, 8 back,
9/10 rotate wrist, 11 center wrist. 0 stops and locks; h explicitly homes again; q exits and stops.
Default translation step is 5 mm; use --step-mm 1 for a 1 mm command step. This is a commanded
step, not measured positioning accuracy. Every session homes once at startup in simulation;
physical startup homing requires --home. A serial fault can prevent software stop delivery.

This is a new console over the existing unchanged driver, not a byte-identical recovery of the
old local ZIP's script. Use the displayed mapping; it is separate from classification action maps.
Real position and grasp success still need observation. This console does not consume RHX data.
