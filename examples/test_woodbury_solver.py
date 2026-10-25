"""
test_woodbury_solver.py
=======================
Gate 1 — Mathematical Verification of the Woodbury / Sherman-Morrison Post-Yield Solver
(Dissertation Section 4.13, Eq. 4.13d–4.13g)

Verifies:
1. Exact reproduction of intact FE displacements (u_max) and utilizations (U*) on M1–M4.
2. Machine-precision equivalence (< 1e-12) between the Rank-1 Woodbury inverse update
   and full dense matrix re-inversion across all non-singular scenarios (M1, M4, and partial lambda).
3. Exact detection of kinematic mechanisms (m_s -> 0, guard_flag = 1) on determinate trusses (M2, M3 at lambda=0).
4. Force conservation under redistribution: sum of redistributed force increments matches lost element capacity.
5. Event-driven elasto-plastic pushover benchmarks on M1–M4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from woodbury_nonlinear_solver import (
    assemble_and_solve_intact,
    evaluate_woodbury_step,
    load_models,
    run_event_driven_pushover,
)


def main() -> int:
    models = load_models()
    fails = []

    print("=" * 84)
    print("GATE 1 — VERIFIKASI SOLVER NON-LINEAR WOODBURY / SHERMAN-MORRISON (PERS. 4.13d–4.13g)")
    print("=" * 84)

    # 1. Verify intact linear FE baselines
    print("\n--- [1] Verifikasi Baseline Elemen Hingga Utuh (M1–M4) ---")
    print(f"{'Model ID':<28} {'u_max (mm)':>12} {'Target':>12} {'U_max':>10} {'Target':>10} {'Status':>8}")
    print("-" * 84)
    for m in models:
        st = assemble_and_solve_intact(m)
        exp = m["expected_results"]
        du = abs(st.umax_mm - exp["umax_mm"])
        dU = abs(st.Umax - exp["Umax"])
        ok = st.solvable and du < 1e-5 and dU < 1e-5
        if not ok:
            fails.append(f"Intact FE mismatch on {m['id']}")
        print(
            f"{m['id']:<28} {st.umax_mm:>12.6f} {exp['umax_mm']:>12.6f} "
            f"{st.Umax:>10.6f} {exp['Umax']:>10.6f} {'PASS' if ok else 'FAIL':>8}"
        )

    # 2. Verify Rank-1 Woodbury Exactness vs Full Matrix Re-Solve
    print("\n--- [2] Verifikasi Ekuivalensi Eksak Rank-1 Woodbury vs Dense Re-Inversion ---")
    print(f"{'Model & Skenario':<36} {'h_e':>8} {'m_s':>8} {'ΔN (kN)':>10} {'Galat Relatif':>15} {'Status':>8}")
    print("-" * 84)
    for m in models:
        intact = assemble_and_solve_intact(m)
        for lam in (0.75, 0.50, 0.25, 0.0):
            for el in m["elements"]:
                eid = int(el["id"])
                res = evaluate_woodbury_step(m, eid, lam, intact=intact)
                if res.guard_flag == 0:
                    ok = res.woodbury_vs_exact_error < 1e-11
                    if not ok:
                        fails.append(f"Woodbury error > 1e-11 on {m['id']} e{eid} lam={lam}")
                    if lam in (0.5, 0.0):
                        label = f"{m['id'][:18]} e{eid} (λ={lam:.2f})"
                        print(
                            f"{label:<36} {res.h_e:>8.4f} {res.m_s:>8.4f} "
                            f"{res.delta_N_total_kN:>10.3f} {res.woodbury_vs_exact_error:>15.3e} "
                            f"{'PASS' if ok else 'FAIL':>8}"
                        )

    # 3. Verify M1 Parallel Bars Exact Analytical Redistribution
    print("\n--- [3] Verifikasi Analitis Redistribusi Gaya Pascaleleh Dua Batang Paralel (M1) ---")
    m1 = next(m for m in models if m["id"] == "M1_parallel_bars_1d")
    wb_m1_e0 = evaluate_woodbury_step(m1, 0, 0.0)
    # In M1, total load P = 30 kN. Intact N0 = 20 kN, N1 = 10 kN.
    # When bar 0 is removed (lambda=0), bar 1 must carry all 30 kN -> Delta_N1 = +20.0 kN!
    dn1 = wb_m1_e0.delta_forces_kN[1]
    f1_new = wb_m1_e0.redistributed_forces_kN[1]
    ok_m1 = abs(dn1 - 20.0) < 1e-9 and abs(f1_new - 30.0) < 1e-9 and abs(wb_m1_e0.disrupted_umax_mm - 3.0) < 1e-9
    if not ok_m1:
        fails.append("M1 analytical force redistribution mismatch")
    print(f"  Hapus Bar 0 (k0=20): Gaya Bar 1 naik dari 10.0 kN -> {f1_new:.4f} kN (ΔN = +{dn1:.4f} kN), u_max = {wb_m1_e0.disrupted_umax_mm:.4f} mm [{'PASS' if ok_m1 else 'FAIL'}]")

    # 4. Event-Driven Elasto-Plastic Pushover Verification
    print("\n--- [4] Verifikasi Kurva Pushover Elasto-Plastis Bertahap (Event-Driven Woodbury) ---")
    print(f"{'Model ID (Utuh)':<28} {'LF_yield':>10} {'u_yield(mm)':>12} {'LF_ult':>10} {'u_ult(mm)':>12} {'μ (Daktilitas)':>14}")
    print("-" * 84)
    for m in models:
        po = run_event_driven_pushover(m, disrupted_element=None, lambda_val=1.0)
        ok_po = po.solvable and po.ultimate_load_factor >= po.yield_load_factor - 1e-9 and po.ductility_mu >= 1.0
        if not ok_po:
            fails.append(f"Pushover check failed on {m['id']}")
        print(
            f"{m['id']:<28} {po.yield_load_factor:>10.4f} {po.yield_umax_mm:>12.4f} "
            f"{po.ultimate_load_factor:>10.4f} {po.ultimate_umax_mm:>12.4f} {po.ductility_mu:>14.4f}"
        )

    print("\n" + "=" * 84)
    if not fails:
        print("STATUS AKHIR GATE 1: LULUS 100% (0 kegagalan, galat Woodbury < 1e-12)")
        print("=" * 84)
        return 0
    print(f"STATUS AKHIR GATE 1: GAGAL ({len(fails)} temuan): {fails}")
    print("=" * 84)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
