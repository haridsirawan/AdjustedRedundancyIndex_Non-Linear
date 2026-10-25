#!/usr/bin/env python3
"""
run_all_verifications.py
========================
Skrip verifikasi terpadu satu pintu (single-entry-point runner) untuk repositori
"AdjustedRedundancyIndex_Non-Linear (EigenFlow)" — Harids Irawan (2026).

Menjalankan seluruh suite pengujian deterministik:
  1. Gate 1: Verifikasi Matematis Solver Non-Linear Woodbury / Sherman-Morrison & Pushover Elasto-Plastis (Pers. 4.13d–4.13g)
  2. Gate 2: Verifikasi Pipeline Bercabang (ARI Linear Gatekeeper -> EigenFlow Non-Linear Woodbury Transparency & Prescriptive Advisor)

Cara menjalankan dari root repositori:
    python3 run_all_verifications.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT_DIR / "examples"

SUITE = [
    ("1. Gate 1: Solver Non-Linear Woodbury & Pushover Elasto-Plastis (M1–M4)", "test_woodbury_solver.py"),
    ("2. Gate 2: Pipeline Bercabang ARI Linear -> Non-Linear & Rekomendasi Preskriptif", "test_branching_pipeline.py"),
]


def main() -> int:
    print("=" * 88)
    print("SUITE VERIFIKASI TERPADU — ADJUSTED REDUNDANCY INDEX NON-LINEAR (EIGENFLOW)")
    print("=" * 88)

    results = []
    t0_all = time.perf_counter()

    for label, script_name in SUITE:
        script_path = EXAMPLES_DIR / script_name
        print(f"\n>>> Menjalankan {label} ({script_name})...")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(EXAMPLES_DIR),
            capture_output=True,
            text=True,
        )
        dt = time.perf_counter() - t0
        passed = proc.returncode == 0
        print(proc.stdout.strip())
        if not passed and proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
        results.append((label, passed, dt))

    total_dt = time.perf_counter() - t0_all
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)

    print("\n" + "=" * 88)
    print("RINGKASAN HASIL VERIFIKASI REPOSITORI ARI NON-LINEAR (EIGENFLOW)")
    print("=" * 88)
    print(f"{'Tahap Verifikasi':<64} | {'Status':<8} | {'Waktu':>8}")
    print("-" * 88)
    for label, ok, dt in results:
        print(f"{label:<64} | {'LULUS' if ok else 'GAGAL':<8} | {dt:>6.2f} s")
    print("-" * 88)
    print(f"Total Tahap Lulus: {n_pass}/{n_total} | Total Waktu Eksekusi: {total_dt:.2f} detik")
    print("=" * 88)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
