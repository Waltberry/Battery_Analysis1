# run_stage4.py (56-state version, spherical FVM solid diffusion)

from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Headless plotting for clusters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import control as ct
from scipy.linalg import block_diag

import jax
import jax.numpy as jnp
import diffrax
from jax_sysid.models import CTModel

from galvani import BioLogic
import src.data_processing as dp


# =========================
# Plot helpers
# =========================
def save_voltage_plot(
    path: str,
    t: np.ndarray,
    y: np.ndarray,
    yhat: Optional[np.ndarray] = None,
    title: str = "Voltage",
):
    plt.figure(figsize=(10, 4))
    plt.plot(t, y[:, 0], label="Truth")
    if yhat is not None:
        plt.plot(t, yhat[:, 0], "--", label="Pred")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("V [V]")
    plt.title(title)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =========================
# Config
# =========================
@dataclass
class Config:
    # Physical constants
    R: float = 8.314462618
    F: float = 96485.33212
    T: float = 298.15
    T_ref: float = 298.15

    # Geometry
    L1: float = 25e-6
    L2: float = 20e-6
    L3: float = 25e-6
    Rn: float = 5e-6
    Rp: float = 5e-6
    A: float = 1.0

    # Transport
    Dn: float = 1e-14
    Dp: float = 1e-14
    De: float = 7.23e-10
    eps: float = 0.30

    # Effective conductivities for R_el
    kappa_n_eff: float = 1.0
    kappa_s_eff: float = 1.0
    kappa_p_eff: float = 1.0

    # Kinetics / active surface
    a_s_n: float = 1.0e6
    a_s_p: float = 1.0e6
    k_n0: float = 2.0e-11
    k_p0: float = 2.0e-11

    # Capacity-scale (stoichiometries)
    csn_max: float = 3.1e4
    csp_max: float = 3.1e4

    # Electrolyte & electrical
    ce0: float = 1000.0
    t_plus: float = 0.38
    k_f: float = 1.0
    R_ohm: float = 0.0
    use_dynamic_film: bool = False
    Rf: float = 0.0
    L_sei: float = 0.0
    kappa_sei: float = 1.0

    # Flags & conventions
    ce_is_deviation: bool = True
    discharge_positive: bool = False  # your dataset convention: discharge current negative
    ln_orientation: str = "right_over_left"
    eta_mode: str = "diff"

    # Numerical guards / stability
    theta_guard: float = 1e-3
    bv_scale: float = 0.7
    N_series: int = 1

    # Discretization (56 states)
    Nn: int = 16
    Np: int = 16
    Ne_n: int = 8
    Ne_s: int = 8
    Ne_p: int = 8


# =========================
# OCVs
# =========================
def ocp_p(xp: np.ndarray) -> np.ndarray:
    x = np.clip(xp, 1e-9, 1 - 1e-9)
    return 4.15 - 0.12 * np.tanh((x - 0.60) / 0.08)

def ocp_n(xn: np.ndarray) -> np.ndarray:
    x = np.clip(xn, 1e-9, 1 - 1e-9)
    return 0.10 + 0.80 * (1.0 / (1.0 + np.exp(-(x - 0.50) / 0.04)))


# =========================
# Dynamic index map
# =========================
def make_idx(cfg: Config) -> Dict[str, Any]:
    Nn = int(cfg.Nn)
    Np = int(cfg.Np)
    Ne = int(cfg.Ne_n + cfg.Ne_s + cfg.Ne_p)

    off_cp = Nn
    off_ce = Nn + Np

    return dict(
        Nn=Nn, Np=Np, Ne=Ne, nx=Nn + Np + Ne,
        cn=slice(0, Nn),
        cp=slice(off_cp, off_cp + Np),
        ce=slice(off_ce, off_ce + Ne),
        cn_surf=Nn - 1,
        cp_surf=off_cp + (Np - 1),
        ce_left=off_ce + 0,
        ce_right=off_ce + (Ne - 1),
    )


# =========================
# Spherical FVM for solid diffusion
# =========================
def build_solid_fvm_spherical(N: int, D: float, R: float) -> np.ndarray:
    """
    Conservative spherical finite-volume discretization of:
        ∂c/∂t = (1/r^2) ∂/∂r ( D r^2 ∂c/∂r )
    on r in [0, R], using shell control volumes.

    - Control volume i spans [r_{i-1/2}, r_{i+1/2}]
    - Unknown c_i is volume-average over that shell
    - Flux at face r_f:  J_f = -D ∂c/∂r |_f
      Net molar rate out of cell uses area 4π r_f^2, but 4π cancels if we use
      reduced volumes/areas (r^3/3 and r^2), which we do for numerical stability.

    Returns A such that:
        c_dot = A c + b * j_surf   (surface flux handled separately)
    """
    if N < 2:
        raise ValueError("N must be >= 2")

    r_f = np.linspace(0.0, R, N + 1, dtype=np.float64)          # faces
    r_c = 0.5 * (r_f[:-1] + r_f[1:])                            # centers

    # Reduced volumes and areas (4π cancels):
    # V_i ∝ (r_{i+1/2}^3 - r_{i-1/2}^3)/3
    V = (r_f[1:]**3 - r_f[:-1]**3) / 3.0                        # shape (N,)
    A_face = r_f**2                                              # shape (N+1,)

    A = np.zeros((N, N), dtype=np.float64)

    # Internal faces: i between cells i-1 and i at face r_f[i]
    for i_face in range(1, N):
        rf = r_f[i_face]
        Af = A_face[i_face]
        # Distance between cell centers adjacent to this face
        dr = r_c[i_face] - r_c[i_face - 1]
        if dr <= 0:
            raise RuntimeError("Non-positive dr encountered in grid construction.")

        G = D * Af / dr  # conductance-like term

        iL = i_face - 1
        iR = i_face

        # Cell L gets +G/V_L * c_R and -G/V_L * c_L
        A[iL, iL] -= G / V[iL]
        A[iL, iR] += G / V[iL]

        # Cell R gets +G/V_R * c_L and -G/V_R * c_R
        A[iR, iR] -= G / V[iR]
        A[iR, iL] += G / V[iR]

    # r=0 boundary: symmetry => zero flux; face area is 0 anyway (rf=0), so nothing to do.

    # r=R boundary: handled via input vector (surface flux) not inside A.
    return A


def build_solid_input_spherical(N: int, R: float, cfg: Config, electrode: str) -> np.ndarray:
    """
    Surface Neumann boundary flux enters ONLY the outermost control volume.

    PDE boundary:
        -D ∂c/∂r |_{r=R} = j_surf   [mol / m^2 / s]
    Finite-volume update for last CV:
        dc_N/dt += (A_R / V_N) * j_surf
    using reduced A_R = R^2 and reduced V_N = (R^3 - r_{N-1/2}^3)/3.

    Then j_surf is linked to applied current:
        j_n =  + I / (F a_s_n A L1)   (sign depends on convention)
        j_p =  - I / (F a_s_p A L3)

    We keep your existing "discharge_positive" switch:
      - If discharge_positive=False, your data has discharge current negative.
    """
    r_f = np.linspace(0.0, R, N + 1, dtype=np.float64)
    V_last = (r_f[-1]**3 - r_f[-2]**3) / 3.0
    A_R = r_f[-1]**2

    # coefficient multiplying j_surf into dc/dt for last state
    surf_to_state = A_R / V_last

    # current-to-flux mapping
    if electrode == "n":
        # Typical SPM sign: j_n = -I/(F a A L) if I>0 is discharge.
        # Your convention: discharge is negative current when discharge_positive=False.
        # So we define:
        #   if discharge_positive=False: discharge I<0 should yield j_n > 0 (lithiation)
        # A robust way: j_n = -(I_eff)/(F a A L) where I_eff is "discharge-positive"
        I_to_j = 1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1)
        sign = -1.0  # j_n = - I_eff * I_to_j
    elif electrode == "p":
        I_to_j = 1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3)
        sign = +1.0  # j_p = + I_eff * I_to_j  (opposite electrode)
    else:
        raise ValueError("electrode must be 'n' or 'p'")

    # Convert "your dataset current" to "discharge-positive" internal current I_eff
    # If discharge_positive=False, discharge current is negative => I_eff = -I_data
    # If discharge_positive=True, discharge current is positive => I_eff = +I_data
    Ieff_sign = +1.0 if cfg.discharge_positive else -1.0

    coef = surf_to_state * sign * Ieff_sign * I_to_j

    b = np.zeros((N, 1), dtype=np.float64)
    b[-1, 0] = coef
    return b


# =========================
# Electrolyte diffusion (unchanged)
# =========================
def build_electrolyte_1d(Nn: int, Ns: int, Np: int, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D electrolyte diffusion across 3 regions (n/sep/p).
    """
    Ne = Nn + Ns + Np
    K = cfg.De / cfg.eps

    Ln, Ls, Lp = cfg.L1, cfg.L2, cfg.L3
    dx_n = Ln / Nn
    dx_s = Ls / Ns
    dx_p = Lp / Np

    A = np.zeros((Ne, Ne), dtype=np.float64)

    def dx(i: int) -> float:
        if i < Nn:
            return dx_n
        if i < Nn + Ns:
            return dx_s
        return dx_p

    for i in range(Ne):
        if i > 0:
            dxL = dx(i - 1)
            dxi = dx(i)
            wL = K / (0.5 * (dxL + dxi) ** 2)
            A[i, i] -= wL
            A[i, i - 1] += wL
        if i < Ne - 1:
            dxi = dx(i)
            dxR = dx(i + 1)
            wR = K / (0.5 * (dxi + dxR) ** 2)
            A[i, i] -= wR
            A[i, i + 1] += wR

    b = np.zeros((Ne, 1), dtype=np.float64)

    # Current source term: keep your prior mapping (works as a tunable surrogate)
    sign_left = -1.0 if cfg.discharge_positive else +1.0
    sign_right = +1.0 if cfg.discharge_positive else -1.0

    s1 = sign_left * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L1 * cfg.eps)
    s3 = sign_right * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L3 * cfg.eps)

    b[0, 0] = s1
    if Nn > 1:
        b[1, 0] = s1
    b[-1, 0] = s3
    if Np > 1:
        b[-2, 0] = s3

    return A, b


# =========================
# Assemble full linear system
# =========================
def assemble_system(cfg: Config):
    idx = make_idx(cfg)

    # solid diffusion (spherical FVM)
    An = build_solid_fvm_spherical(idx["Nn"], cfg.Dn, cfg.Rn)
    Ap = build_solid_fvm_spherical(idx["Np"], cfg.Dp, cfg.Rp)
    Bn = build_solid_input_spherical(idx["Nn"], cfg.Rn, cfg, electrode="n")
    Bp = build_solid_input_spherical(idx["Np"], cfg.Rp, cfg, electrode="p")

    # electrolyte
    Ae, Be = build_electrolyte_1d(cfg.Ne_n, cfg.Ne_s, cfg.Ne_p, cfg)

    # global
    Aglob = block_diag(An, Ap, Ae)
    Bglob = np.vstack([Bn, Bp, Be])

    state_names = (
        [f"cn{i}" for i in range(1, idx["Nn"] + 1)]
        + [f"cp{i}" for i in range(1, idx["Np"] + 1)]
        + [f"ce{i}" for i in range(1, idx["Ne"] + 1)]
    )

    S = ct.ss(Aglob, Bglob, np.eye(Aglob.shape[0]), np.zeros((Aglob.shape[0], 1)))
    return S, Aglob, Bglob, (An, Ap, Ae, Bn, Bp, Be), state_names, idx


def make_x0(cfg: Config, idx: Dict[str, Any], theta_n0=0.6, theta_p0=0.6, ce0=0.0):
    x0 = np.zeros(idx["nx"], dtype=np.float64)
    x0[idx["cn"]] = float(theta_n0) * cfg.csn_max
    x0[idx["cp"]] = float(theta_p0) * cfg.csp_max
    x0[idx["ce"]] = float(ce0)
    return x0


# =========================
# Cycle prep helpers (unchanged)
# =========================
def _median_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.nanmedian(np.abs(x)))

def _sign_fraction(x: np.ndarray, tol: float = 1e-12) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    pos = float(np.mean(x > +tol))
    neg = float(np.mean(x < -tol))
    zer = float(np.mean(np.abs(x) <= tol))
    return {"pos": pos, "neg": neg, "zero": zer}

def _guess_I_in_amps(I_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    med = _median_abs(I_raw)
    if med > 5.0:
        return (np.asarray(I_raw, dtype=np.float64) * 1e-3), f"guessed units: mA -> A (median|I_raw|={med:.6g})"
    return (np.asarray(I_raw, dtype=np.float64)), f"guessed units: already A (median|I_raw|={med:.6g})"

def _apply_v_ref(y: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if mode == "none":
        return y
    if mode == "first":
        return y - float(y[0])
    if mode == "mean":
        return y - float(np.mean(y))
    raise ValueError("v_ref must be one of: none, first, mean")

def prepare_cycle_for_sysid(
    cycle_df: pd.DataFrame,
    *,
    i_col: str,
    v_col: str,
    force_units: Optional[str] = "mA",    # "A", "mA", "auto"/None
    v_ref: str = "none",                 # none|first|mean
    enforce_discharge_only: bool = True,
    discharge_sign: str = "negative",    # negative or positive
    tol_I: float = 1e-6,
    name: str = "cycle",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(cycle_df.index.to_numpy(), dtype=np.float64).reshape(-1)
    I_raw = np.asarray(cycle_df[i_col].to_numpy(), dtype=np.float64).reshape(-1)
    V_raw = np.asarray(cycle_df[v_col].to_numpy(), dtype=np.float64).reshape(-1)

    print("=" * 80)
    print(f"[prepare] {name}")
    print("describe:")
    print(cycle_df[[i_col, v_col]].describe().to_string())
    print("raw current sign fractions:", _sign_fraction(I_raw, tol=tol_I))

    # Units
    if force_units is None or str(force_units).lower() == "auto":
        I_A, note = _guess_I_in_amps(I_raw)
        print("units:", note)
    elif str(force_units).lower() == "ma":
        I_A = I_raw * 1e-3
    elif str(force_units).lower() == "a":
        I_A = I_raw
    else:
        raise ValueError("force_units must be one of: None/'auto', 'mA', 'A'")

    # Rebase time
    t = t - t[0]

    # Strictly increasing time
    keep_time = np.ones_like(t, dtype=bool)
    keep_time[1:] = t[1:] > t[:-1]
    t = t[keep_time]
    I_A = I_A[keep_time]
    V_raw = V_raw[keep_time]

    # Enforce discharge-only if mixed sign
    frac = _sign_fraction(I_A, tol=tol_I)
    mixed = (frac["pos"] > 0.05) and (frac["neg"] > 0.05)
    if enforce_discharge_only and mixed:
        if discharge_sign == "negative":
            mask = I_A < -tol_I
        elif discharge_sign == "positive":
            mask = I_A > +tol_I
        else:
            raise ValueError("discharge_sign must be 'negative' or 'positive'")
        before = len(I_A)
        t, I_A, V_raw = t[mask], I_A[mask], V_raw[mask]
        print(f"[prepare] discharge-only filter: kept {len(I_A)}/{before} samples")
        if len(t) > 0:
            t = t - t[0]

    # Optional flip if using discharge positive convention externally
    if discharge_sign == "positive":
        I_A = -I_A

    V = _apply_v_ref(V_raw, v_ref)

    t_np = t.reshape(-1)
    U_np = I_A.reshape(-1, 1)
    Y_np = V.reshape(-1, 1)

    print("[prepare] final current sign fractions:", _sign_fraction(U_np[:, 0], tol=tol_I))
    print("[prepare] U range [A]:", float(U_np.min()), float(U_np.max()))
    print("[prepare] Y range [V]:", float(Y_np.min()), float(Y_np.max()))
    print("=" * 80)
    return t_np, U_np, Y_np

def resample_uniform(t: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float):
    t0, t1 = float(t[0]), float(t[-1])
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"resample_uniform: dt must be >0, got {dt}")
    tg = np.arange(t0, t1 + dt, dt, dtype=np.float64)
    u1 = np.interp(tg, t, u[:, 0]).reshape(-1, 1)
    y1 = np.interp(tg, t, y[:, 0]).reshape(-1, 1)
    return tg, u1, y1


# =========================
# Voltage terms
# =========================
def electrolyte_resistance(cfg: Config) -> float:
    return (cfg.L1 / cfg.kappa_n_eff + 2.0 * cfg.L2 / cfg.kappa_s_eff + cfg.L3 / cfg.kappa_p_eff) / (2.0 * cfg.A)

def film_resistance(cfg: Config) -> float:
    if cfg.use_dynamic_film and cfg.L_sei > 0.0:
        return cfg.L_sei / (cfg.kappa_sei * cfg.a_s_n * cfg.A * cfg.L1)
    return cfg.Rf


# =========================
# JAX models (Stage2 warm start + Stage4)
# =========================
def build_stage2_and_stage4_models(cfg: Config, idx: Dict[str, Any], theta_guard: float):
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    DTYPE = jnp.float64

    _, A_np, B_np, (An_np, Ap_np, Ae_np, Bn_np, Bp_np, Be_np), _, _idx2 = assemble_system(cfg)
    assert _idx2["nx"] == idx["nx"], "IDX mismatch; rebuild idx from same cfg before calling."

    A_fix = jnp.array(A_np, dtype=DTYPE)
    B_fix = jnp.array(B_np, dtype=DTYPE)

    An0 = jnp.array(An_np, dtype=DTYPE)
    Ap0 = jnp.array(Ap_np, dtype=DTYPE)
    Ae0 = jnp.array(Ae_np, dtype=DTYPE)
    Bn0 = jnp.array(Bn_np, dtype=DTYPE)
    Bp0 = jnp.array(Bp_np, dtype=DTYPE)
    Be0 = jnp.array(Be_np, dtype=DTYPE)

    R_fixed = float(cfg.R_ohm + electrolyte_resistance(cfg) + film_resistance(cfg))

    @jax.jit
    def ocp_p_jax(xp):
        xp = jnp.clip(xp, 1e-9, 1 - 1e-9)
        return 4.15 - 0.12 * jnp.tanh((xp - 0.60) / 0.08)

    @jax.jit
    def ocp_n_jax(xn):
        xn = jnp.clip(xn, 1e-9, 1 - 1e-9)
        return 0.10 + 0.80 * (1.0 / (1.0 + jnp.exp(-(xn - 0.50) / 0.04)))

    USE_V0 = True

    def theta_map(raw_theta: jnp.ndarray) -> dict:
        Gamma_p = jnp.exp(raw_theta[0])
        Gamma_n = jnp.exp(raw_theta[1])
        K_e     = jnp.exp(raw_theta[2])
        R0      = jnp.exp(raw_theta[3])
        V0      = raw_theta[4] if USE_V0 else DTYPE(0.0)
        return dict(
            csn_max=DTYPE(cfg.csn_max),
            csp_max=DTYPE(cfg.csp_max),
            Gamma_p=Gamma_p,
            Gamma_n=Gamma_n,
            K_e=K_e,
            R0=R0,
            V0=V0,
        )

    def voltage_from_params(x, I, p: dict):
        xp = jnp.clip(x[idx["cp_surf"]] / p["csp_max"], 1e-9, 1 - 1e-9)
        xn = jnp.clip(x[idx["cn_surf"]] / p["csn_max"], 1e-9, 1 - 1e-9)

        Up = ocp_p_jax(xp)
        Un = ocp_n_jax(xn)

        ceL_raw = x[idx["ce_left"]]
        ceR_raw = x[idx["ce_right"]]
        ceL = (DTYPE(cfg.ce0) + ceL_raw) if cfg.ce_is_deviation else ceL_raw
        ceR = (DTYPE(cfg.ce0) + ceR_raw) if cfg.ce_is_deviation else ceR_raw
        ceL = jnp.maximum(ceL, 1e-12)
        ceR = jnp.maximum(ceR, 1e-12)
        ce_avg = jnp.clip(0.5 * (ceL + ceR), 1e-12, 1e12)

        xp_eff = jnp.clip(xp, theta_guard, 1.0 - theta_guard)
        xn_eff = jnp.clip(xn, theta_guard, 1.0 - theta_guard)

        denom_p = p["Gamma_p"] * jnp.sqrt(ce_avg) * jnp.sqrt(xp_eff * (1 - xp_eff)) + 1e-18
        denom_n = p["Gamma_n"] * jnp.sqrt(ce_avg) * jnp.sqrt(xn_eff * (1 - xn_eff)) + 1e-18

        kappa = (2.0 * DTYPE(cfg.R) * DTYPE(cfg.T) / DTYPE(cfg.F))
        eta_p = DTYPE(cfg.bv_scale) * kappa * jnp.arcsinh(I / (2.0 * denom_p))
        eta_n = DTYPE(cfg.bv_scale) * kappa * jnp.arcsinh(I / (2.0 * denom_n))
        eta_combo = (eta_p - eta_n) if (cfg.eta_mode == "diff") else (eta_p + eta_n)

        ln_arg = (ceR / ceL) if (cfg.ln_orientation == "right_over_left") else (ceL / ceR)
        dphi_e = p["K_e"] * jnp.log(jnp.maximum(ln_arg, 1e-12))

        ohmic = -I * (DTYPE(R_fixed) + p["R0"])
        V_cell = (Up - Un) + eta_combo + dphi_e + ohmic
        V = DTYPE(cfg.N_series) * V_cell + p["V0"]
        return V

    @jax.jit
    def state_fcn_stage2(x, u, t, params):
        I = u[0]
        return A_fix @ x + (B_fix[:, 0] * I)

    @jax.jit
    def output_fcn_stage2(x, u, t, params):
        (raw_theta,) = params
        p = theta_map(raw_theta)
        I = u[0]
        V = voltage_from_params(x, I, p)
        return jnp.array([V], dtype=DTYPE)

    def block_diag_jax(A, B, C):
        z1 = jnp.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        z2 = jnp.zeros((A.shape[0], C.shape[1]), dtype=A.dtype)
        z3 = jnp.zeros((B.shape[0], A.shape[1]), dtype=A.dtype)
        z4 = jnp.zeros((B.shape[0], C.shape[1]), dtype=A.dtype)
        z5 = jnp.zeros((C.shape[0], A.shape[1]), dtype=A.dtype)
        z6 = jnp.zeros((C.shape[0], B.shape[1]), dtype=A.dtype)
        top = jnp.concatenate([A, z1, z2], axis=1)
        mid = jnp.concatenate([z3, B, z4], axis=1)
        bot = jnp.concatenate([z5, z6, C], axis=1)
        return jnp.concatenate([top, mid, bot], axis=0)

    def ab_map(raw_ab: jnp.ndarray) -> dict:
        return dict(
            sDn=jnp.exp(raw_ab[0]),
            sDp=jnp.exp(raw_ab[1]),
            sDe=jnp.exp(raw_ab[2]),
            gn=jnp.exp(raw_ab[3]),
            gp=jnp.exp(raw_ab[4]),
            ge=jnp.exp(raw_ab[5]),
        )

    def build_AB_from_ab(raw_ab: jnp.ndarray):
        ab = ab_map(raw_ab)
        A = block_diag_jax(ab["sDn"] * An0, ab["sDp"] * Ap0, ab["sDe"] * Ae0)
        B = jnp.concatenate([ab["gn"] * Bn0, ab["gp"] * Bp0, ab["ge"] * Be0], axis=0)
        return A, B

    def unpack_stage4(raw: jnp.ndarray):
        raw_theta = raw[:5]
        raw_ab = raw[5:11]
        return raw_theta, raw_ab

    @jax.jit
    def state_fcn_stage4(x, u, t, params):
        (raw,) = params
        _, raw_ab = unpack_stage4(raw)
        A, B = build_AB_from_ab(raw_ab)
        I = u[0]
        return A @ x + (B[:, 0] * I)

    @jax.jit
    def output_fcn_stage4(x, u, t, params):
        (raw,) = params
        raw_theta, _ = unpack_stage4(raw)
        p = theta_map(raw_theta)
        I = u[0]
        V = voltage_from_params(x, I, p)
        return jnp.array([V], dtype=DTYPE)

    return {
        "DTYPE": DTYPE,
        "theta_map": theta_map,
        "ab_map": ab_map,
        "state_fcn_stage2": state_fcn_stage2,
        "output_fcn_stage2": output_fcn_stage2,
        "state_fcn_stage4": state_fcn_stage4,
        "output_fcn_stage4": output_fcn_stage4,
        "R_fixed": R_fixed,
        "nx": int(idx["nx"]),
    }


def fit_stage2_warmstart(
    cfg: Config,
    idx: Dict[str, Any],
    t_np: np.ndarray,
    U_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    dt0_div: float = 50.0,
    adam_epochs: int = 300,
    lbfgs_epochs: int = 40,
    adam_eta: float = 1e-3,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 2_000_000,
) -> np.ndarray:
    parts = build_stage2_and_stage4_models(cfg, idx, theta_guard=cfg.theta_guard)
    DTYPE = parts["DTYPE"]
    nx = int(parts["nx"])

    Ke_phys = (2.0 * cfg.R * cfg.T / cfg.F) * (1.0 - cfg.t_plus) * cfg.k_f
    theta0_raw = np.array([np.log(1.0), np.log(1.0), np.log(Ke_phys), np.log(1e-6), 0.0], dtype=np.float64)

    x0_init = make_x0(cfg, idx, theta_n0=0.6, theta_p0=0.6, ce0=0.0)

    m2 = CTModel(nx, 1, 1, state_fcn=parts["state_fcn_stage2"], output_fcn=parts["output_fcn_stage2"])
    m2.init(params=[theta0_raw], x0=np.array(x0_init, dtype=np.float64))

    rho_x0 = 1e6
    rho_th = 1e-8
    try:
        m2.loss(rho_x0=rho_x0, rho_th=rho_th, xsat=1e9)
    except TypeError:
        m2.loss(rho_x0=rho_x0, rho_th=rho_th)

    m2.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, adam_eta=adam_eta)

    dt0 = float(t_np[1] - t_np[0]) / float(dt0_div)
    m2.integration_options(
        ode_solver=diffrax.Tsit5(),
        dt0=dt0,
        max_steps=int(max_steps),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
    )

    print("[stage2 warm] prefit...")
    Y0, _ = m2.predict(m2.x0, U_np, t_np)
    print("[stage2 warm] prefit max|err|:", float(np.max(np.abs(np.asarray(Y0) - Y_np))))

    print("[stage2 warm] fit...")
    m2.fit(Y_np, U_np, t_np)

    theta_hat = np.asarray(m2.params[0]).copy()
    print("[stage2 warm] theta_hat_raw:", theta_hat)
    p = parts["theta_map"](jnp.array(theta_hat, dtype=DTYPE))
    print("[stage2 warm] learned:", {k: float(p[k]) for k in ["Gamma_p", "Gamma_n", "K_e", "R0", "V0"]})
    return theta_hat


def stage4_multistart(
    cfg: Config,
    idx: Dict[str, Any],
    t_np: np.ndarray,
    U_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    theta_center_raw: np.ndarray,
    n_trials: int = 100,
    x0_trials: int = 16,
    seed: int = 0,
    trial_start: int = 0,
    trial_count: int = -1,
    outdir: str = "outputs",
    checkpoint_every: int = 1,
    ab_lo: float = 0.5,
    ab_hi: float = 2.0,
    theta_sigma: float = 0.50,
    x0_theta_lo: float = 0.05,
    x0_theta_hi: float = 0.95,
    x0_ce0: float = 0.0,
    rho_x0: float = 0.0,
    rho_th: float = 1e-8,
    adam_epochs: int = 300,
    lbfgs_epochs: int = 80,
    adam_eta: float = 1e-3,
    dt0_div: float = 25.0,
    max_steps: int = 5_000_000,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    parts = build_stage2_and_stage4_models(cfg, idx, theta_guard=cfg.theta_guard)
    DTYPE = parts["DTYPE"]
    nx = int(parts["nx"])

    os.makedirs(outdir, exist_ok=True)
    ckpt_csv = os.path.join(outdir, "stage4_trials_partial.csv")
    ckpt_best = os.path.join(outdir, "stage4_best_partial.json")

    if trial_count is None or int(trial_count) < 0:
        i_start = int(trial_start)
        i_end = int(trial_start) + int(n_trials)
    else:
        i_start = int(trial_start)
        i_end = int(trial_start) + int(trial_count)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    def unpack_stage4_np(raw: np.ndarray):
        raw_theta = jnp.array(raw[:5], dtype=DTYPE)
        raw_ab = jnp.array(raw[5:11], dtype=DTYPE)
        return raw_theta, raw_ab

    for i in range(i_start, i_end):
        rng = np.random.default_rng(seed + i)

        raw_theta0 = theta_center_raw + rng.normal(0.0, theta_sigma, size=(5,))
        raw_ab0 = rng.uniform(np.log(ab_lo), np.log(ab_hi), size=(6,))
        raw0 = np.concatenate([raw_theta0, raw_ab0]).astype(np.float64)

        for j in range(max(1, x0_trials)):
            tn0 = float(rng.uniform(x0_theta_lo, x0_theta_hi))
            tp0 = float(rng.uniform(x0_theta_lo, x0_theta_hi))
            x0_init = make_x0(cfg, idx, theta_n0=tn0, theta_p0=tp0, ce0=x0_ce0)

            t_start_wall = time.time()
            try:
                m4 = CTModel(nx, 1, 1, state_fcn=parts["state_fcn_stage4"], output_fcn=parts["output_fcn_stage4"])
                m4.init(params=[raw0.astype(np.float64)], x0=np.array(x0_init, dtype=np.float64))

                try:
                    m4.loss(rho_x0=rho_x0, rho_th=rho_th, xsat=1e9)
                except TypeError:
                    m4.loss(rho_x0=rho_x0, rho_th=rho_th)

                m4.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, adam_eta=adam_eta)

                dt0 = float(t_np[1] - t_np[0]) / float(dt0_div)
                m4.integration_options(
                    ode_solver=diffrax.Tsit5(),
                    dt0=dt0,
                    max_steps=int(max_steps),
                    stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                )

                Y0, _ = m4.predict(m4.x0, U_np, t_np)
                pre_max = float(np.max(np.abs(np.asarray(Y0) - Y_np)))

                m4.fit(Y_np, U_np, t_np)

                Yhat, _ = m4.predict(m4.x0, U_np, t_np)
                err = np.asarray(Yhat) - Y_np
                post_max = float(np.max(np.abs(err)))
                mse = float(np.mean(err**2))

                raw_final = np.asarray(m4.params[0]).copy()
                raw_theta_f, raw_ab_f = unpack_stage4_np(raw_final)
                p_f = parts["theta_map"](raw_theta_f)
                ab_f = parts["ab_map"](raw_ab_f)

                th = {
                    "Gamma_p": float(p_f["Gamma_p"]),
                    "Gamma_n": float(p_f["Gamma_n"]),
                    "K_e": float(p_f["K_e"]),
                    "R0": float(p_f["R0"]),
                    "V0": float(p_f["V0"]),
                }
                ab = {k: float(ab_f[k]) for k in ["sDn", "sDp", "sDe", "gn", "gp", "ge"]}

                ok = bool(np.isfinite(mse) and np.isfinite(post_max))
                errtxt = ""
            except Exception as e:
                ok = False
                pre_max = float("inf")
                post_max = float("inf")
                mse = float("inf")
                Yhat = None
                th = {}
                ab = {}
                raw_final = None
                errtxt = repr(e)

            wall = time.time() - t_start_wall
            rows.append({
                "trial": int(i),
                "x0trial": int(j),
                "ok_run": int(ok),
                "pre_max": float(pre_max),
                "post_max": float(post_max),
                "mse": float(mse),
                "wall_s": float(wall),
                "err": errtxt,
                "xn0": float(tn0),
                "xp0": float(tp0),
                **th,
                **ab,
            })

            if ok and (best is None or mse < best["metrics"]["mse"]):
                best = {
                    "Yhat": np.asarray(Yhat),
                    "metrics": {"pre_max": float(pre_max), "post_max": float(post_max), "mse": float(mse)},
                    "theta": th,
                    "ab": ab,
                    "x0": x0_init,
                    "raw_final": raw_final,
                    "trial": int(i),
                    "x0trial": int(j),
                }

        print(f"[stage4] trial {i} done. best_mse={best['metrics']['mse'] if best else np.inf:.6g}")

        if checkpoint_every > 0 and ((i - i_start + 1) % checkpoint_every == 0):
            df_partial = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
            df_partial.to_csv(ckpt_csv, index=False)
            if best is not None:
                best_to_save = {
                    "trial": best["trial"],
                    "x0trial": best["x0trial"],
                    "metrics": best["metrics"],
                    "theta": best["theta"],
                    "ab": best["ab"],
                    "x0": best["x0"].tolist(),
                }
                with open(ckpt_best, "w", encoding="utf-8") as f:
                    json.dump(best_to_save, f, indent=2)
            print(f"[stage4] checkpoint saved: {ckpt_csv}")

    df = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    summary = {
        "trial_start": int(i_start),
        "trial_end_exclusive": int(i_end),
        "total_configs": int(len(df)),
        "runs_completed": int(df["ok_run"].sum()) if len(df) else 0,
        "best_mse": float(df.iloc[0]["mse"]) if len(df) else float("inf"),
        "best_post_max": float(df.iloc[0]["post_max"]) if len(df) else float("inf"),
    }

    print("\n=== Stage 4 summary ===")
    print(summary)
    if len(df):
        print("\nTop 10 by MSE:")
        print(df.head(10).to_string(index=False))

    return best, df, summary


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mpr", type=str, default="data/sample.mpr")
    ap.add_argument("--outdir", type=str, default="outputs")

    ap.add_argument("--i_col", type=str, default="control/mA")
    ap.add_argument("--v_col", type=str, default="Ewe/V")

    ap.add_argument("--force_units", type=str, default="mA", help="A|mA|auto")
    ap.add_argument("--v_ref", type=str, default="none", help="none|first|mean")

    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--x0_trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--trial_start", type=int, default=0)
    ap.add_argument("--trial_count", type=int, default=-1)

    ap.add_argument("--tmax", type=float, default=-1.0)
    ap.add_argument("--resample", action="store_true")
    ap.add_argument("--dt0_div", type=float, default=25.0)
    ap.add_argument("--max_steps", type=int, default=5_000_000)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)

    ap.add_argument("--ab_lo", type=float, default=0.50)
    ap.add_argument("--ab_hi", type=float, default=2.0)
    ap.add_argument("--theta_sigma", type=float, default=0.50)

    ap.add_argument("--warm_adam", type=int, default=300)
    ap.add_argument("--warm_lbfgs", type=int, default=40)

    ap.add_argument("--stage4_adam", type=int, default=300)
    ap.add_argument("--stage4_lbfgs", type=int, default=80)

    ap.add_argument("--checkpoint_every", type=int, default=1)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[run] cwd:", os.getcwd())
    print("[run] loading mpr:", args.mpr)

    if not os.path.isfile(args.mpr):
        raise FileNotFoundError(
            f"Could not find MPR file: {args.mpr}\n"
            f"Fix by passing --mpr /full/path/to/file.mpr or placing it under data/."
        )

    mpr = BioLogic.MPRfile(args.mpr)
    df_all = pd.DataFrame(mpr.data)

    if "time/s" not in df_all.columns:
        raise RuntimeError("Expected 'time/s' column not found in MPR data.")

    if args.i_col not in df_all.columns or args.v_col not in df_all.columns:
        raise RuntimeError(f"Missing columns. Have: {df_all.columns.tolist()}")

    df = df_all.set_index("time/s")[[args.i_col, args.v_col]]

    cycles = dp.find_discharging_cycles(df)
    if not cycles:
        raise RuntimeError("dp.find_discharging_cycles returned no cycles.")

    cycle1 = cycles[0].copy()
    print(f"[run] found {len(cycles)} cycles, using cycle1 with {len(cycle1)} samples")

    CFG = Config()
    CFG.N_series = 1
    CFG.discharge_positive = False  # keep your dataset convention

    idx = make_idx(CFG)
    print(f"[run] model nx={idx['nx']} (Nn,Np,Ne)=({idx['Nn']},{idx['Np']},{idx['Ne']})")

    force_units = None if str(args.force_units).lower() == "auto" else args.force_units

    t_np, U_np, Y_np = prepare_cycle_for_sysid(
        cycle1,
        i_col=args.i_col,
        v_col=args.v_col,
        force_units=force_units,
        v_ref=args.v_ref,
        enforce_discharge_only=True,
        discharge_sign="negative",
        tol_I=1e-6,
        name="Cycle 1",
    )

    if len(t_np) < 3:
        raise RuntimeError("Cycle 1 too short after filtering. Need >=3 points.")

    if args.resample:
        dt_med = float(np.median(np.diff(t_np)))
        if not (np.isfinite(dt_med) and dt_med > 0):
            raise RuntimeError(f"Bad dt_med={dt_med} from time diffs; cannot resample.")
        t_np, U_np, Y_np = resample_uniform(t_np, U_np, Y_np, dt=dt_med)
        print("[run] resampled dt_med:", dt_med, "new_len:", len(t_np))

    if args.tmax > 0:
        mask = t_np <= float(args.tmax)
        t_np, U_np, Y_np = t_np[mask], U_np[mask], Y_np[mask]
        print("[run] applied tmax:", args.tmax, "new_len:", len(t_np))

    save_voltage_plot(
        os.path.join(args.outdir, "cycle1_voltage_truth.png"),
        t_np, Y_np, None,
        title=f"Cycle1 truth (prepared) v_ref={args.v_ref}"
    )

    print("[run] stage2 warm-start to get theta_center_raw...")
    theta_center_raw = fit_stage2_warmstart(
        CFG, idx, t_np, U_np, Y_np,
        dt0_div=max(25.0, args.dt0_div),
        adam_epochs=args.warm_adam,
        lbfgs_epochs=args.warm_lbfgs,
        max_steps=min(args.max_steps, 2_000_000),
        rtol=args.rtol,
        atol=args.atol,
    )

    print("[run] stage4 multistart...")
    best, df_trials, summary = stage4_multistart(
        CFG, idx, t_np, U_np, Y_np,
        theta_center_raw=theta_center_raw,
        n_trials=args.n_trials,
        x0_trials=args.x0_trials,
        seed=args.seed,
        trial_start=args.trial_start,
        trial_count=args.trial_count,
        outdir=args.outdir,
        checkpoint_every=args.checkpoint_every,
        ab_lo=args.ab_lo,
        ab_hi=args.ab_hi,
        theta_sigma=args.theta_sigma,
        adam_epochs=args.stage4_adam,
        lbfgs_epochs=args.stage4_lbfgs,
        dt0_div=args.dt0_div,
        max_steps=args.max_steps,
        rtol=args.rtol,
        atol=args.atol,
    )

    csv_path = os.path.join(args.outdir, "stage4_trials.csv")
    df_trials.to_csv(csv_path, index=False)
    print("[run] saved:", csv_path)

    summary_path = os.path.join(args.outdir, "stage4_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[run] saved:", summary_path)

    if best is None:
        print("[run] no successful stage4 run.")
        return

    best_path = os.path.join(args.outdir, "stage4_best.json")
    best_to_save = {
        "trial": best["trial"],
        "x0trial": best["x0trial"],
        "metrics": best["metrics"],
        "theta": best["theta"],
        "ab": best["ab"],
        "x0": best["x0"].tolist(),
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_to_save, f, indent=2)
    print("[run] saved:", best_path)

    save_voltage_plot(
        os.path.join(args.outdir, "stage4_best_fit.png"),
        t_np, Y_np, best["Yhat"],
        title="Stage 4 best fit (cycle1) 56-state"
    )
    print("[run] saved best plot.")


if __name__ == "__main__":
    main()
