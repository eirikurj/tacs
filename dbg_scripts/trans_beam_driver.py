"""Replay test_trans_beam xpt FD check with the integrator Newton log on the perturbed solve."""

import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, "tests/integration_tests")
import test_trans_beam as t  # noqa: E402

case = t.ProblemTest("test_total_xpt_sensitivities")
case.setUp()
f0 = case.run_solve()
case.run_sensitivities(includeXptSens=True)
for prob in case.tacs_probs:
    prob.setOption("printLevel", 2 if prob.name == "sinusoid" else 0)
    prob.integrator.setPrintLevel(2 if prob.name == "sinusoid" else 0)
if MPI.COMM_WORLD.rank == 0:
    print("#### PERTURBED SOLVE", flush=True)
f1 = case.run_solve(xpts=case.perturb_tacs_vec(case.xpts0, case.xpts_pert))
if MPI.COMM_WORLD.rank == 0:
    for k in sorted(f0):
        print(f"FUNC {k:22s} f0={f0[k]!r:24s} f1={f1[k]!r:24s} fd={(f1[k] - f0[k]) / case.dh!r}")
    print("xpts finite:", np.all(np.isfinite(case.xpts0)), flush=True)
