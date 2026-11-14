"""Replay test_trans_beam xpt FD check with the integrator Newton log on the perturbed solve."""

import os
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, "tests/integration_tests")
import test_trans_beam as t  # noqa: E402

_tol0 = os.environ.get("TB_TOL0")
if _tol0:
    _orig_setup = t.ProblemTest.setup_tacs_problems

    def _setup(self, comm):
        """Apply TB_TOL0 to the Newton tolerances before the first solve."""
        probs, fea = _orig_setup(self, comm)
        for prob in probs:
            prob.setOption("L2Convergence", float(_tol0))
            prob.setOption("L2ConvergenceRel", float(_tol0))
        return probs, fea

    t.ProblemTest.setup_tacs_problems = _setup
case = t.ProblemTest("test_total_xpt_sensitivities")
case.setUp()
f0 = case.run_solve()
case.run_sensitivities(includeXptSens=True)
tol = os.environ.get("TB_TOL")
if tol:
    for prob in case.tacs_probs:
        prob.integrator.setAbsTol(float(tol))
        prob.integrator.setRelTol(float(tol))
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
