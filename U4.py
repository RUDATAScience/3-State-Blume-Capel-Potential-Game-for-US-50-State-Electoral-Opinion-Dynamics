#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binder Cumulant U_4 Phase Diagram in (lambda, Delta_NG) Plane & 50-State Master Scaling Curve
==============================================================================================
- Horizontal Axis: Signal-to-Noise Ratio lambda_i = (mu_i / sigma_i)^2
- Vertical Axis: Non-Gaussian Fourth-Cumulant Residual Delta_NG,i = 4*(mu_i/sigma_i)*gamma_3,i + kappa_4,i
- Phase Boundary: U_4 = 0  <==>  Delta_NG = 2 * lambda^2
- Comparison between Cell 2 (Single Gamma Shock: NC, GA, AZ < 0) and Cell 3 (3-Issue Superposition: MN, NH, ME, NM > 0)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_or_reconstruct_ensemble_data():
    """
    Cell 2 (`us50_1M_10M_posterior_calibration_summary.csv`) と
    Cell 3 (`us50_post_midterm_ensemble_summary.csv`) を読み込み、
    信号対雑音比 lambda_i と非ガウス残差 Delta_NG,i を厳密計算する。
    """
    csv_cell2 = "us50_1M_10M_posterior_calibration_summary.csv"
    csv_cell3 = "us50_post_midterm_ensemble_summary.csv"

    if os.path.exists(csv_cell2) and os.path.exists(csv_cell3):
        df_c2 = pd.read_csv(csv_cell2)
        df_c3 = pd.read_csv(csv_cell3)

        # Cell 3: sim_mean_m, sim_std_m, binder_cumulant_U4 から厳密計算
        df_c3["lambda_snr"] = (df_c3["sim_mean_m"] / df_c3["sim_std_m"]) ** 2
        df_c3["two_lambda_sq"] = 2.0 * (df_c3["lambda_snr"] ** 2)
        df_c3["Delta_NG"] = (
            df_c3["two_lambda_sq"]
            - 3.0 * ((1.0 + df_c3["lambda_snr"]) ** 2) * df_c3["binder_cumulant_U4"]
        )

        # Cell 2: R = 0.5*(1 + m/q) より sigma_m = 2 * q * sigma_R を用いて計算
        df_c2["est_std_m"] = 2.0 * df_c2["ens_mean_turnout_q"] * df_c2["ens_rep_share_std"]
        df_c2["lambda_snr"] = (df_c2["ens_mean_m"] / df_c2["est_std_m"]) ** 2
        df_c2["two_lambda_sq"] = 2.0 * (df_c2["lambda_snr"] ** 2)
        df_c2["Delta_NG"] = (
            df_c2["two_lambda_sq"]
            - 3.0 * ((1.0 + df_c2["lambda_snr"]) ** 2) * df_c2["binder_cumulant_U4"]
        )
        return df_c2, df_c3
    else:
        raise FileNotFoundError(
            "Both 'us50_1M_10M_posterior_calibration_summary.csv' and "
            "'us50_post_midterm_ensemble_summary.csv' must be present in the working directory."
        )


def plot_binder_phase_diagram(df_cell2, df_cell3, out_filename="binder_cumulant_phase_diagram_cell2_vs_cell3.png"):
    fig = plt.figure(figsize=(16.5, 6.8), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.25)

    # ==========================================================================
    # Panel A: (lambda_i, Delta_NG,i) Phase Diagram & U_4 = 0 Boundary Curve
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0])

    lam_grid = np.linspace(0.0, 2.8, 300)
    del_grid = np.linspace(-3.5, 5.8, 300)
    LAM, DEL = np.meshgrid(lam_grid, del_grid)
    U4_SURF = (2.0 * (LAM ** 2) - DEL) / (3.0 * ((1.0 + LAM) ** 2))

    cf = ax1.contourf(LAM, DEL, np.clip(U4_SURF, -0.35, 0.67), levels=28, cmap="RdYlBu_r", alpha=0.82)
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label(r"Binder Cumulant $U_4(\lambda, \Delta_{\mathrm{NG}})$", fontsize=10)

    # U_4 = 0 境界曲線: Delta_NG = 2 * lambda^2
    boundary_del = 2.0 * (lam_grid ** 2)
    ax1.plot(lam_grid, boundary_del, color="black", lw=2.8, ls="--",
             label=r"Zero Boundary $U_4 = 0$ ($\Delta_{\mathrm{NG}} = 2\lambda^2$)")
    ax1.axhline(0.0, color="dimgray", lw=1.2, ls=":", label=r"Gaussian Reference ($\Delta_{\mathrm{NG}} = 0$)")

    # 臨界領域 (lambda <= 2.7) の州を抽出してプロット
    c2_crit = df_cell2[df_cell2["lambda_snr"] <= 2.7].copy()
    c3_crit = df_cell3[df_cell3["lambda_snr"] <= 2.7].copy()

    ax1.scatter(c2_crit["lambda_snr"], c2_crit["Delta_NG"],
                c=np.where(c2_crit["binder_cumulant_U4"] < 0, "#d62728", "#ff7f0e"),
                marker="o", s=95, edgecolors="black", lw=1.1, zorder=6,
                label="Cell 2 (Single Gamma Shock: NC, GA, AZ < 0; NV, PA, WI, MI > 0)")

    for _, r in c2_crit.iterrows():
        offset_y = 0.22 if r["state"] not in ["WI", "PA"] else (-0.38 if r["state"] == "WI" else 0.22)
        offset_x = 0.04 if r["state"] != "MI" else -0.22
        ax1.annotate(f"{r['state']} (U4={r['binder_cumulant_U4']:.2f})",
                     (r["lambda_snr"] + offset_x, r["Delta_NG"] + offset_y),
                     fontsize=8.5, fontweight="bold", color="#800000")

    ax1.scatter(c3_crit["lambda_snr"], c3_crit["Delta_NG"],
                c="#1f77b4", marker="s", s=85, edgecolors="black", lw=1.1, zorder=6,
                label="Cell 3 (Post-Midterm 3-Issue: MN, NH, ME, NM, VA, CO > 0)")

    for _, r in c3_crit.iterrows():
        off_y = -0.35 if r["state"] in ["MN", "ME"] else 0.18
        off_x = 0.04
        ax1.annotate(f"{r['state']} (U4={r['binder_cumulant_U4']:.2f})",
                     (r["lambda_snr"] + off_x, r["Delta_NG"] + off_y),
                     fontsize=8.5, fontweight="bold", color="#08306b")

    ax1.text(0.10, 4.1, "Negative Binder Regime\n" + r"$U_4 < 0 \,\, (\Delta_{\mathrm{NG}} > 2\lambda^2)$" + "\nLeptokurtic Gamma Tail",
             fontsize=9, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#8b0000", alpha=0.78))

    ax1.text(1.30, -2.6, "Positive Binder Regime\n" + r"$U_4 > 0 \,\, (\Delta_{\mathrm{NG}} < 2\lambda^2)$" + "\nPlatykurtic / Ordered",
             fontsize=9, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#08306b", alpha=0.78))

    ax1.set_xlim(-0.08, 2.75)
    ax1.set_ylim(-3.4, 5.8)
    ax1.set_xlabel(r"Signal-to-Noise Ratio $\lambda_i = (\mu_i / \sigma_i)^2$", fontsize=10.5)
    ax1.set_ylabel(r"Non-Gaussian Cumulant Residual $\Delta_{\mathrm{NG},i} = 4(\mu_i/\sigma_i)\gamma_{3,i} + \kappa_{4,i}$", fontsize=10.5)
    ax1.set_title(r"(A) $(\lambda_i, \Delta_{\mathrm{NG},i})$ Phase Diagram & $U_4 = 0$ Boundary ($\Delta_{\mathrm{NG}} = 2\lambda^2$)",
                  fontsize=11.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # ==========================================================================
    # Panel B: 50-State Master Scaling Curve U_4 vs lambda (Log Scale)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1])

    lam_log = np.logspace(-3, 3.2, 500)
    u4_gauss_curve = (2.0 * (lam_log ** 2)) / (3.0 * ((1.0 + lam_log) ** 2))

    ax2.plot(lam_log, u4_gauss_curve, color="black", lw=2.2, ls="-",
             label=r"Gaussian Baseline $U_4^{\mathrm{Gauss}}(\lambda) = \frac{2\lambda^2}{3(1+\lambda)^2}$")
    ax2.axhline(2.0 / 3.0, color="darkgreen", lw=1.4, ls="--", label=r"Ordered Limit $U_4 \to 2/3 \approx 0.6667$")
    ax2.axhline(0.0, color="red", lw=1.4, ls=":", label=r"Critical Zero Line $U_4 = 0$")

    ax2.scatter(df_cell2["lambda_snr"], df_cell2["binder_cumulant_U4"],
                c="#d62728", marker="o", s=55, alpha=0.85, edgecolors="black", lw=0.7, zorder=5,
                label="All 50 States: Cell 2 (Scandal-Economy Ensemble)")
    ax2.scatter(df_cell3["lambda_snr"], df_cell3["binder_cumulant_U4"],
                c="#1f77b4", marker="s", s=50, alpha=0.85, edgecolors="black", lw=0.7, zorder=5,
                label="All 50 States: Cell 3 (Post-Midterm 3-Issue Ensemble)")

    for st_code, dy in [("NC", -0.045), ("GA", -0.045), ("AZ", 0.032), ("NV", -0.045), ("PA", 0.035)]:
        r2 = df_cell2[df_cell2["state"] == st_code].iloc[0]
        ax2.annotate(f"{st_code}(C2)", (r2["lambda_snr"], r2["binder_cumulant_U4"] + dy),
                     fontsize=8, fontweight="bold", color="#8b0000")

    for st_code, dy in [("NH", 0.032), ("ME", 0.032), ("MN", -0.045), ("NM", -0.045), ("VA", 0.032), ("PA", 0.032), ("OH", -0.045)]:
        r3 = df_cell3[df_cell3["state"] == st_code].iloc[0]
        ax2.annotate(f"{st_code}(C3)", (r3["lambda_snr"], r3["binder_cumulant_U4"] + dy),
                     fontsize=8, fontweight="bold", color="#08306b")

    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 1.5e3)
    ax2.set_ylim(-0.22, 0.74)
    ax2.set_xlabel(r"Signal-to-Noise Ratio $\lambda_i = (\mu_i / \sigma_i)^2$ [Log Scale]", fontsize=10.5)
    ax2.set_ylabel(r"Fourth-Order Binder Cumulant $U_{4,i} = 1 - \frac{\langle m_i^4 \rangle}{3\langle m_i^2 \rangle^2}$", fontsize=10.5)
    ax2.set_title(r"(B) 50-State Master Scaling Curve $U_{4,i}(\lambda_i)$ Across Both Ensembles",
                  fontsize=11.5, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    plt.suptitle(r"Binder Cumulant Phase Boundary Analysis: $\Delta_{\mathrm{NG},i} = 2\lambda_i^2$ Criterion and 50-State Scaling (Cell 2 vs. Cell 3)",
                 fontsize=13, fontweight="bold", y=1.01)

    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_filename}")


if __name__ == "__main__":
    df_c2, df_c3 = load_or_reconstruct_ensemble_data()
    plot_binder_phase_diagram(df_c2, df_c3)
