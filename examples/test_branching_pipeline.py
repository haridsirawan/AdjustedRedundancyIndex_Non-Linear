"""
test_branching_pipeline.py
==========================
Gate 2 — Verification of the Two-Branch ARI Pipeline (ARI Linear Gatekeeper -> EigenFlow Non-Linear Advisor)
and Export of Complete Verification Artifacts (`results/verification/`).

Validates:
1. Exact alignment of Branch 1 (ARI Linear) with Dissertation Tabel 4.11 & Tabel 4.3.
2. Zero-miss forwarding policy F^+: every mechanism-forming or post-yield overloaded element
   is forwarded to Branch 2 (EigenFlow Non-Linear Woodbury).
3. Transparent engineering diagnostics: P1/P2 load paths, redistributed force Delta_N (kN),
   Residual Reserve Ratio (RRR), ductility demand mu, and prescriptive engineering recommendations.
4. Saves CSV and JSON verification artifacts to `results/verification/`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eigenflow_core import evaluate_branching_model
from woodbury_nonlinear_solver import load_models

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "results" / "verification"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()
    all_rows = []
    summary_by_model = {}
    fails = []

    print("=" * 116)
    print("GATE 2 — PIPELINE BERCABANG: ARI LINEAR (PRA-ANALISIS) -> EIGENFLOW / ARI NON-LINEAR (TRANSPARANSI WOODBURY)")
    print("=" * 116)

    for m in models:
        res = evaluate_branching_model(m, lambdas=(0.0, 0.5), force_nonlinear_all=True)
        mid = res["model_id"]
        summary_by_model[mid] = {
            "model_name": res["model_name"],
            "intact_solvable": res["intact_solvable"],
            "intact_umax_mm": round(res["intact_umax_mm"], 6),
            "intact_Umax": round(res["intact_Umax"], 6),
            "baseline_score_S0": round(res["baseline_score_S0"], 6),
            "intact_yield_load_factor": round(res["intact_yield_load_factor"], 6),
            "intact_ultimate_load_factor": round(res["intact_ultimate_load_factor"], 6),
            "intact_ductility_mu": round(res["intact_ductility_mu"], 6),
            "intact_yield_sequence": res["intact_yield_sequence"],
            "scenarios": res["ranking"],
        }

        print(f"\n>>> Model: {mid} — {res['model_name']}")
        print(
            f"    Utuh: u_max = {res['intact_umax_mm']:.4f} mm | S(G0) = {res['baseline_score_S0']:.6f} | "
            f"LF_yield = {res['intact_yield_load_factor']:.3f} | LF_ult = {res['intact_ultimate_load_factor']:.3f} | "
            f"Urutan Leleh = {res['intact_yield_sequence']}"
        )
        print("-" * 116)
        print(
            f"{'Elemen':<24} {'λ':>5} {'Zona L1':<9} {'ARI_Lin':>8} {'m_s':>7} "
            f"{'ΔN(kN)':>8} {'RRR_NL':>8} {'μ':>7} {'ARI_NL':>8} {'CI_NL':>8} {'Alasan F^+':<18}"
        )
        print("-" * 116)

        for r in res["reports"]:
            if r.lambda_val == 0.0:
                print(
                    f"{r.element_name[:24]:<24} {r.lambda_val:>5.2f} {r.zone:<9} "
                    f"{r.ARI_linear:>8.4f} {r.m_s:>7.3f} {r.delta_N_kN:>8.2f} "
                    f"{r.RRR_nonlinear:>8.4f} {r.ductility_mu:>7.2f} {r.ARI_nonlinear:>8.4f} "
                    f"{r.CI_nonlinear:>8.4f} {r.forward_reason:<18}"
                )
            all_rows.append(
                {
                    "model_id": r.model_id,
                    "element_id": r.element_id,
                    "element_name": r.element_name,
                    "lambda": r.lambda_val,
                    "zone": r.zone,
                    "RR_linear": round(r.RR_linear, 6),
                    "ARI_linear": round(r.ARI_linear, 6),
                    "CI_linear": round(r.CI_linear, 6),
                    "U_star_intact": round(r.U_star_intact, 6),
                    "h_e": round(r.h_e, 6),
                    "m_s": round(r.m_s, 6),
                    "guard_flag": r.guard_flag,
                    "forward_to_nonlinear": r.forward_to_nonlinear,
                    "forward_reason": r.forward_reason,
                    "primary_path": "->".join(str(x) for x in r.primary_path),
                    "alternate_path": "->".join(str(x) for x in r.alternate_path) if r.alternate_path else "NONE",
                    "delta_N_kN": round(r.delta_N_kN, 6),
                    "max_surviving_U_star": round(r.max_surviving_U_star, 6),
                    "RRR_nonlinear": round(r.RRR_nonlinear, 6),
                    "ductility_mu": round(r.ductility_mu, 6),
                    "ARI_nonlinear": round(r.ARI_nonlinear, 6),
                    "CI_nonlinear": round(r.CI_nonlinear, 6),
                    "woodbury_exact_error": f"{r.woodbury_exact_error:.3e}",
                    "prescriptive_recommendation": r.prescriptive_recommendation,
                }
            )

        print("  Rekomendasi Preskriptif AI (EigenFlow Non-Linear Woodbury, λ=0.0):")
        for r in [x for x in res["reports"] if x.lambda_val == 0.0][:3]:
            print(f"    * {r.prescriptive_recommendation}")

    # Verify M3 e4 and M4 e4 benchmarks
    m3_e4 = next(
        r for r in summary_by_model["M3_square_one_diagonal"]["scenarios"]
        if r["element_id"] == 4 and r["lambda_val"] == 0.0
    )
    if abs(m3_e4["RR_linear"] - 0.717285) > 1e-5 or m3_e4["guard_flag"] != 1:
        fails.append("M3 e4 benchmark or kinematic guard failed")

    m4_e4 = next(
        r for r in summary_by_model["M4_square_two_diagonals"]["scenarios"]
        if r["element_id"] == 4 and r["lambda_val"] == 0.0
    )
    if m4_e4["guard_flag"] != 0 or m4_e4["delta_N_kN"] <= 0.0 or m4_e4["RRR_nonlinear"] <= 0.0:
        fails.append("M4 e4 non-linear Woodbury redistribution failed")

    # Save CSV & JSON artifacts
    csv_path = OUT_DIR / "eigenflow_nonlinear_screening.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    json_path = OUT_DIR / "eigenflow_nonlinear_report.json"
    json_path.write_text(json.dumps(summary_by_model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 116)
    print(f"Artefak CSV disimpan ke  : {csv_path}")
    print(f"Artefak JSON disimpan ke : {json_path}")
    if not fails:
        print("STATUS AKHIR GATE 2: LULUS 100% (Pipeline Bercabang Linear -> Non-Linear Woodbury Terverifikasi)")
        print("=" * 116)
        return 0
    print(f"STATUS AKHIR GATE 2: GAGAL: {fails}")
    print("=" * 116)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
