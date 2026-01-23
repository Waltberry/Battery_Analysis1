# run_stage4generateddata.py
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Headless plotting (cluster-friendly)
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


# =========================================================
# Plot helpers
# =========================================================
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


# =========================================================
# Index map
# =========================================================
IDX = {
    "cn": slice(0, 4),
    "cp": slice(4, 8),
    "ce": slice(8, 14),
    "cn_surf": 3,
    "cp_surf": 7,
    "ce_left": 8,
    "ce_right": 13,
}


# =========================================================
# Config (synthetic truth + ident model share this)
# =========================================================
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
    use_arrhenius: bool = False
    Ea_n: float = 0.0
    Ea_p: float = 0.0

    # LAM (optional)
    lam_n: float = 0.0
    lam_p: float = 0.0

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

    # IMPORTANT:
    # Synthetic generator usually uses DISCHARGE POSITIVE current (I>0).
    # Keep this consistent with your build_Bn/build_Bp sign logic.
    discharge_positive: bool = True

    ln_orientation: str = "right_over_left"
    eta_mode: str = "diff"

    # Numerical guards / stability
    theta_guard: float = 1e-3
    bv_scale: float = 0.7
    N_series: int = 3


# =========================================================
# Simple OCVs (same functional form as your synthetic code)
# =========================================================
def ocp_p(xp: np.ndarray) -> np.ndarray:
    x = np.clip(xp, 1e-9, 1 - 1e-9)
    return 4.15 - 0.12 * np.tanh((x - 0.60) / 0.08)

def ocp_n(xn: np.ndarray) -> np.ndarray:
    x = np.clip(xn, 1e-9, 1 - 1e-9)
    return 0.10 + 0.80 * (1.0 / (1.0 + np.exp(-(x - 0.50) / 0.04)))


# =========================================================
# Build blocks (same as your run_stage4 real code)
# =========================================================
def build_An(cfg: Config) -> np.ndarray:
    s = cfg.Dn / (cfg.Rn ** 2)
    A = np.zeros((4, 4))
    A[0, 0], A[0, 1] = -24 * s, 24 * s
    A[1, 0], A[1, 1], A[1, 2] = 16 * s, -40 * s, 24 * s
    A[2, 1], A[2, 2], A[2, 3] = 16 * s, -40 * s, 24 * s
    A[3, 2], A[3, 3] = 16 * s, -16 * s
    return A

def build_Bn(cfg: Config) -> np.ndarray:
    sign = -1.0 if cfg.discharge_positive else +1.0
    b = np.zeros((4, 1))
    b[-1, 0] = sign * (6.0 / cfg.Rn) * (1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1))
    return b

def build_Ap(cfg: Config) -> np.ndarray:
    s = cfg.Dp / (cfg.Rp ** 2)
    A = np.zeros((4, 4))
    A[0, 0], A[0, 1] = -24 * s, 24 * s
    A[1, 0], A[1, 1], A[1, 2] = 16 * s, -40 * s, 24 * s
    A[2, 1], A[2, 2], A[2, 3] = 16 * s, -40 * s, 24 * s
    A[3, 2], A[3, 3] = 16 * s, -16 * s
    return A

def build_Bp(cfg: Config) -> np.ndarray:
    sign = +1.0 if cfg.discharge_positive else -1.0
    b = np.zeros((4, 1))
    b[-1, 0] = sign * (6.0 / cfg.Rp) * (1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3))
    return b

def build_Ae(cfg: Config) -> np.ndarray:
    K = cfg.De / cfg.eps
    Ae = np.zeros((6, 6))
    w_in = lambda L: K * 4.0 / (L ** 2)
    w_intf = lambda La, Lb: K * 16.0 / ((La + Lb) ** 2)

    w11 = w_in(cfg.L1)
    w12 = w_intf(cfg.L1, cfg.L2)
    w23 = w_in(cfg.L2)
    w34 = w_intf(cfg.L2, cfg.L3)
    w45 = w_in(cfg.L3)

    Ae[0, 0] = -(w11);                 Ae[0, 1] = +(w11)
    Ae[1, 0] = +(w11); Ae[1, 1] = -(w11 + w12); Ae[1, 2] = +(w12)
    Ae[2, 1] = +(w12); Ae[2, 2] = -(w12 + w23); Ae[2, 3] = +(w23)
    Ae[3, 2] = +(w23); Ae[3, 3] = -(w23 + w34); Ae[3, 4] = +(w34)
    Ae[4, 3] = +(w34); Ae[4, 4] = -(w34 + w45); Ae[4, 5] = +(w45)
    Ae[5, 4] = +(w45); Ae[5, 5] = -(w45)
    return Ae

def build_Be(cfg: Config) -> np.ndarray:
    b = np.zeros((6, 1))
    sign_left = -1.0 if cfg.discharge_positive else +1.0
    sign_right = +1.0 if cfg.discharge_positive else -1.0
    s1 = sign_left * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L1 * cfg.eps)
    s3 = sign_right * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L3 * cfg.eps)
    b[0, 0] = s1; b[1, 0] = s1
    b[4, 0] = s3; b[5, 0] = s3
    return b

def assemble_system(cfg: Config):
    An = build_An(cfg); Ap = build_Ap(cfg); Ae = build_Ae(cfg)
    Bn = build_Bn(cfg); Bp = build_Bp(cfg); Be = build_Be(cfg)

    Aglob = block_diag(An, Ap, Ae)
    Bglob = np.vstack([Bn, Bp, Be])

    state_names = (
        [f"cn{i}" for i in range(1, 5)]
        + [f"cp{i}" for i in range(1, 5)]
        + [f"ce{i}" for i in range(1, 7)]
    )

    S = ct.ss(Aglob, Bglob, np.eye(Aglob.shape[0]), np.zeros((Aglob.shape[0], 1)))
    return S, Aglob, Bglob, (An, Ap, Ae, Bn, Bp, Be), state_names

def make_x0(cfg: Config, theta_n0=0.8, theta_p0=0.4, ce0=0.0):
    x0 = np.zeros(14)
    x0[IDX["cn"]] = float(theta_n0) * cfg.csn_max
    x0[IDX["cp"]] = float(theta_p0) * cfg.csp_max
    x0[IDX["ce"]] = float(ce0)
    return x0


# =========================================================
# Voltage terms used by jax model
# =========================================================
def electrolyte_resistance(cfg: Config) -> float:
    return (cfg.L1 / cfg.kappa_n_eff + 2.0 * cfg.L2 / cfg.kappa_s_eff + cfg.L3 / cfg.kappa_p_eff) / (2.0 * cfg.A)

def film_resistance(cfg: Config) -> float:
    if cfg.use_dynamic_film and cfg.L_sei > 0.0:
        return cfg.L_sei / (cfg.kappa_sei * cfg.a_s_n * cfg.A * cfg.L1)
    return cfg.Rf


# =========================================================
# Synthetic truth generator (control.nlsys)
# =========================================================
def terminal_voltage_truth(x: np.ndarray, cfg: Config, I: float) -> float:
    xp = np.clip(x[IDX["cp_surf"]] / cfg.csp_max, 1e-9, 1 - 1e-9)
    xn = np.clip(x[IDX["cn_surf"]] / cfg.csn_max, 1e-9, 1 - 1e-9)
    Up = float(ocp_p(np.array([xp]))[0])
    Un = float(ocp_n(np.array([xn]))[0])

    ceL_raw = float(x[IDX["ce_left"]])
    ceR_raw = float(x[IDX["ce_right"]])
    ceL = (cfg.ce0 + ceL_raw) if cfg.ce_is_deviation else ceL_raw
    ceR = (cfg.ce0 + ceR_raw) if cfg.ce_is_deviation else ceR_raw
    ceL = max(ceL, 1e-12)
    ceR = max(ceR, 1e-12)

    # A simplified BV-like term (perfect synthetic model assumptions).
    # We keep the same "shape" as your learning model (arcsinh),
    # but with truth kinetics embedded in i0_current_scales style.
    # For generated data, we just use a stable, smooth nonlinear term:
    kappa = (2.0 * cfg.R * cfg.T / cfg.F)
    eta_combo = cfg.bv_scale * kappa * np.arcsinh(I / (2.0 * 1.0))  # stable default

    ln_arg = (ceR / ceL) if (cfg.ln_orientation == "right_over_left") else (ceL / ceR)
    Ke_phys = (2.0 * cfg.R * cfg.T / cfg.F) * (1.0 - cfg.t_plus) * cfg.k_f
    dphi_e = Ke_phys * np.log(max(ln_arg, 1e-12))

    ohmic = -I * (cfg.R_ohm + electrolyte_resistance(cfg) + film_resistance(cfg))
    V_cell = (Up - Un) + eta_combo + dphi_e + ohmic
    return float(cfg.N_series * V_cell)

def battery_update(t, x, u, params):
    A = params["A"]; B = params["B"]; cfg = params["cfg"]
    if u is None or (hasattr(u, "size") and u.size == 0):
        I = float(params["I_const"])
    else:
        I = float(np.asarray(u).reshape(-1)[0])
    return A @ x + B[:, 0] * I

def battery_output(t, x, u, params):
    cfg = params["cfg"]
    if u is None or (hasattr(u, "size") and u.size == 0):
        I = float(params["I_const"])
    else:
        I = float(np.asarray(u).reshape(-1)[0])
    V = terminal_voltage_truth(x, cfg, I=I)
    return np.hstack([x, V])

def generate_discharge_data(
    cfg: Config,
    *,
    I_const: float,
    sim_t_end: float,
    sim_dt: float,
    theta_n0: float,
    theta_p0: float,
    ce0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = np.arange(0.0, sim_t_end + sim_dt, sim_dt)

    _, A, B, _, state_names = assemble_system(cfg)
    nl_params = dict(A=A, B=B, cfg=cfg, I_const=float(I_const))

    battery_nl = ct.nlsys(
        battery_update, battery_output,
        name="battery_synth",
        params=nl_params,
        states=state_names,
        outputs=state_names + ["V"],
        inputs=0,
    )

    x0 = make_x0(cfg, theta_n0=theta_n0, theta_p0=theta_p0, ce0=ce0)
    resp = ct.input_output_response(battery_nl, T, 0, X0=x0)

    X = resp.states.T
    Y_full = resp.outputs.T
    V = Y_full[:, -1:].copy()
    U = np.full((len(T), 1), float(I_const), dtype=np.float64)

    return T.astype(np.float64), U.astype(np.float64), X.astype(np.float64), V.astype(np.float64)


def resample_uniform(t: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float):
    t0, t1 = float(t[0]), float(t[-1])
    tg = np.arange(t0, t1 + dt, dt, dtype=np.float64)
    u1 = np.interp(tg, t, u[:, 0]).reshape(-1, 1)
    y1 = np.interp(tg, t, y[:, 0]).reshape(-1, 1)
    return tg, u1, y1


# =========================================================
# JAX models (stage2 warm start + stage4)
# Synthetic default: NO V0 unless enabled
# =========================================================
def build_stage2_and_stage4_models(cfg: Config, theta_guard: float, use_v0: bool):
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    DTYPE = jnp.float64

    _, A_np, B_np, (An_np, Ap_np, Ae_np, Bn_np, Bp_np, Be_np), _ = assemble_system(cfg)
    A_fix = jnp.array(A_np, dtype=DTYPE)
    B_fix = jnp.array(B_np, dtype=DTYPE)

    # base blocks for AB-scaling
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

    # ---- theta map
    def theta_map(raw_theta: jnp.ndarray) -> dict:
        Gamma_p = jnp.exp(raw_theta[0])
        Gamma_n = jnp.exp(raw_theta[1])
        K_e     = jnp.exp(raw_theta[2])
        R0      = jnp.exp(raw_theta[3])
        V0      = raw_theta[4] if use_v0 else DTYPE(0.0)
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
        xp = jnp.clip(x[IDX["cp_surf"]] / p["csp_max"], 1e-9, 1 - 1e-9)
        xn = jnp.clip(x[IDX["cn_surf"]] / p["csn_max"], 1e-9, 1 - 1e-9)

        Up = ocp_p_jax(xp)
        Un = ocp_n_jax(xn)

        ceL_raw = x[IDX["ce_left"]]
        ceR_raw = x[IDX["ce_right"]]
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

    # ---- Stage2 (A,B frozen)
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

    # ---- Stage4: AB scaling
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

    theta_len = 5 if use_v0 else 4

    def unpack_stage4(raw: jnp.ndarray):
        raw_theta = raw[:theta_len]
        raw_ab = raw[theta_len:theta_len + 6]
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
        "theta_len": theta_len,
    }


def fit_stage2_warmstart(
    cfg: Config,
    t_np: np.ndarray,
    U_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    use_v0: bool,
    dt0_div: float = 50.0,
    adam_epochs: int = 300,
    lbfgs_epochs: int = 40,
    adam_eta: float = 1e-3,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 2_000_000,
) -> np.ndarray:
    parts = build_stage2_and_stage4_models(cfg, theta_guard=cfg.theta_guard, use_v0=use_v0)

    Ke_phys = (2.0 * cfg.R * cfg.T / cfg.F) * (1.0 - cfg.t_plus) * cfg.k_f
    if use_v0:
        theta0_raw = np.array([np.log(1.0), np.log(1.0), np.log(Ke_phys), np.log(1e-6), 0.0], dtype=np.float64)
    else:
        theta0_raw = np.array([np.log(1.0), np.log(1.0), np.log(Ke_phys), np.log(1e-6)], dtype=np.float64)

    x0_init = make_x0(cfg, theta_n0=0.6, theta_p0=0.6, ce0=0.0)

    m2 = CTModel(14, 1, 1, state_fcn=parts["state_fcn_stage2"], output_fcn=parts["output_fcn_stage2"])
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

    # pretty print
    DTYPE = parts["DTYPE"]
    p = parts["theta_map"](jnp.array(theta_hat, dtype=DTYPE))
    keys = ["Gamma_p", "Gamma_n", "K_e", "R0"] + (["V0"] if use_v0 else [])
    learned = {k: float(p[k]) for k in keys}
    print("[stage2 warm] learned:", learned)

    return theta_hat


def stage4_multistart(
    cfg: Config,
    t_np: np.ndarray,
    U_np: np.ndarray,
    Y_np: np.ndarray,
    *,
    use_v0: bool,
    theta_center_raw: np.ndarray,
    X_truth: Optional[np.ndarray],
    mode: str,  # "locked" or "free"
    n_trials: int = 50,
    x0_trials: int = 8,
    seed: int = 0,
    trial_start: int = 0,
    trial_count: int = -1,
    outdir: str = "outputs_generated",
    checkpoint_every: int = 1,
    ab_lo: float = 0.90,
    ab_hi: float = 1.10,
    theta_sigma: float = 0.10,
    x0_theta_lo: float = 0.05,
    x0_theta_hi: float = 0.95,
    x0_ce0: float = 0.0,
    rho_x0_locked: float = 1e6,
    rho_x0_free: float = 0.0,
    rho_th: float = 1e-8,
    adam_epochs: int = 300,
    lbfgs_epochs: int = 80,
    adam_eta: float = 1e-3,
    dt0_div: float = 25.0,
    max_steps: int = 5_000_000,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    parts = build_stage2_and_stage4_models(cfg, theta_guard=cfg.theta_guard, use_v0=use_v0)
    DTYPE = parts["DTYPE"]
    theta_len = int(parts["theta_len"])

    os.makedirs(outdir, exist_ok=True)
    ckpt_csv = os.path.join(outdir, "stage4_trials_partial.csv")
    ckpt_best = os.path.join(outdir, "stage4_best_partial.json")

    if trial_count is None or int(trial_count) < 0:
        i_start = int(trial_start)
        i_end = int(trial_start) + int(n_trials)
    else:
        i_start = int(trial_start)
        i_end = int(trial_start) + int(trial_count)

    rows = []
    best: Optional[Dict[str, Any]] = None

    def unpack_stage4_np(raw: np.ndarray):
        raw_theta = jnp.array(raw[:theta_len], dtype=DTYPE)
        raw_ab = jnp.array(raw[theta_len:theta_len + 6], dtype=DTYPE)
        return raw_theta, raw_ab

    for i in range(i_start, i_end):
        rng = np.random.default_rng(seed + i)

        raw_theta0 = theta_center_raw + rng.normal(0.0, theta_sigma, size=(theta_len,))
        raw_ab0 = rng.uniform(np.log(ab_lo), np.log(ab_hi), size=(6,))
        raw0 = np.concatenate([raw_theta0, raw_ab0]).astype(np.float64)

        # choose x0 list + rho_x0 based on mode
        if mode == "locked":
            if X_truth is None:
                raise RuntimeError("mode='locked' requires X_truth from generator (so x0 = X_truth[0]).")
            x0_list = [np.array(X_truth[0], dtype=np.float64)]
            rho_x0 = float(rho_x0_locked)
        elif mode == "free":
            x0_list = []
            for _ in range(max(1, x0_trials)):
                tn0 = float(rng.uniform(x0_theta_lo, x0_theta_hi))
                tp0 = float(rng.uniform(x0_theta_lo, x0_theta_hi))
                x0_list.append(make_x0(cfg, theta_n0=tn0, theta_p0=tp0, ce0=x0_ce0))
            rho_x0 = float(rho_x0_free)
        else:
            raise ValueError("mode must be 'locked' or 'free'")

        for j, x0_init in enumerate(x0_list):
            t_start_wall = time.time()
            try:
                m4 = CTModel(14, 1, 1, state_fcn=parts["state_fcn_stage4"], output_fcn=parts["output_fcn_stage4"])
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

                th_keys = ["Gamma_p", "Gamma_n", "K_e", "R0"] + (["V0"] if use_v0 else [])
                th = {k: float(p_f[k]) for k in th_keys}
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

            row = {
                "mode": mode,
                "trial": int(i),
                "x0trial": int(j),
                "ok_run": int(ok),
                "pre_max": float(pre_max),
                "post_max": float(post_max),
                "mse": float(mse),
                "wall_s": float(wall),
                "err": errtxt,
                **th,
                **ab,
            }

            if mode == "free":
                tn0 = float(np.clip(x0_init[IDX["cn_surf"]] / cfg.csn_max, 0.0, 1.0))
                tp0 = float(np.clip(x0_init[IDX["cp_surf"]] / cfg.csp_max, 0.0, 1.0))
                row.update({"xn0": tn0, "xp0": tp0})

            rows.append(row)

            if ok and (best is None or mse < best["metrics"]["mse"]):
                best = {
                    "Yhat": np.asarray(Yhat),
                    "metrics": {"pre_max": float(pre_max), "post_max": float(post_max), "mse": float(mse)},
                    "theta": th,
                    "ab": ab,
                    "x0": np.array(x0_init, dtype=np.float64),
                    "raw_final": raw_final,
                    "trial": int(i),
                    "x0trial": int(j),
                    "mode": mode,
                }

        print(f"[stage4] trial {i} done. best_mse={best['metrics']['mse'] if best else np.inf:.6g}")

        # checkpoint
        if checkpoint_every > 0 and ((i - i_start + 1) % checkpoint_every == 0):
            df_partial = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
            df_partial.to_csv(ckpt_csv, index=False)
            if best is not None:
                best_to_save = {
                    "mode": best["mode"],
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
        "mode": mode,
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


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", type=str, default="outputs_generated")

    # synthetic generation knobs
    ap.add_argument("--I_const", type=float, default=2.0)
    ap.add_argument("--t_end", type=float, default=1000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--theta_n0", type=float, default=0.8)
    ap.add_argument("--theta_p0", type=float, default=0.4)
    ap.add_argument("--ce0", type=float, default=0.0)
    ap.add_argument("--N_series", type=int, default=3)

    # data prep knobs
    ap.add_argument("--tmax", type=float, default=-1.0, help="If >0, keep only t<=tmax seconds")
    ap.add_argument("--resample", action="store_true", help="Resample to uniform dt (median dt)")

    # stage2 warmstart knobs
    ap.add_argument("--use_v0", action="store_true", help="Enable V0 offset in theta (usually OFF for synthetic).")
    ap.add_argument("--warm_adam", type=int, default=300)
    ap.add_argument("--warm_lbfgs", type=int, default=40)

    # stage4 knobs (multistart)
    ap.add_argument("--mode", type=str, default="locked", help="locked|free")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--x0_trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trial_start", type=int, default=0)
    ap.add_argument("--trial_count", type=int, default=-1)
    ap.add_argument("--checkpoint_every", type=int, default=1)

    ap.add_argument("--ab_lo", type=float, default=0.90)
    ap.add_argument("--ab_hi", type=float, default=1.10)
    ap.add_argument("--theta_sigma", type=float, default=0.10)

    ap.add_argument("--stage4_adam", type=int, default=300)
    ap.add_argument("--stage4_lbfgs", type=int, default=80)
    ap.add_argument("--adam_eta", type=float, default=1e-3)

    ap.add_argument("--dt0_div", type=float, default=25.0)
    ap.add_argument("--max_steps", type=int, default=5_000_000)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[run] cwd:", os.getcwd())
    print("[run] generating synthetic data...")

    CFG = Config()
    CFG.N_series = int(args.N_series)

    # Generate synthetic truth
    t_np, U_np, X_truth, Y_np = generate_discharge_data(
        CFG,
        I_const=float(args.I_const),
        sim_t_end=float(args.t_end),
        sim_dt=float(args.dt),
        theta_n0=float(args.theta_n0),
        theta_p0=float(args.theta_p0),
        ce0=float(args.ce0),
    )

    # Optional tmax
    if args.tmax > 0:
        mask = t_np <= float(args.tmax)
        t_np, U_np, Y_np = t_np[mask], U_np[mask], Y_np[mask]
        X_truth = X_truth[mask]
        print("[run] applied tmax:", args.tmax, "new_len:", len(t_np))

    # Optional resample
    if args.resample:
        dt_med = float(np.median(np.diff(t_np)))
        t_np, U_np, Y_np = resample_uniform(t_np, U_np, Y_np, dt=dt_med)
        # X_truth no longer aligned after resampling; only safe for "free" mode now.
        X_truth = None
        print("[run] resampled dt_med:", dt_med, "new_len:", len(t_np))
        print("[run] NOTE: X_truth invalid after resample -> forcing mode to 'free' unless you disable resample.")
        if str(args.mode).lower() == "locked":
            args.mode = "free"

    # Save truth plot
    save_voltage_plot(
        os.path.join(args.outdir, "synthetic_voltage_truth.png"),
        t_np, Y_np, None,
        title=f"Synthetic truth (I={args.I_const}, N_series={args.N_series})"
    )

    # Stage2 warmstart -> theta_center_raw
    print("[run] stage2 warm-start to get theta_center_raw...")
    theta_center_raw = fit_stage2_warmstart(
        CFG, t_np, U_np, Y_np,
        use_v0=bool(args.use_v0),
        dt0_div=max(25.0, float(args.dt0_div)),
        adam_epochs=int(args.warm_adam),
        lbfgs_epochs=int(args.warm_lbfgs),
        adam_eta=float(args.adam_eta),
        max_steps=min(int(args.max_steps), 2_000_000),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    # Stage4 multistart
    print("[run] stage4 multistart...")
    best, df_trials, summary = stage4_multistart(
        CFG, t_np, U_np, Y_np,
        use_v0=bool(args.use_v0),
        theta_center_raw=theta_center_raw,
        X_truth=X_truth,
        mode=str(args.mode).lower(),
        n_trials=int(args.n_trials),
        x0_trials=int(args.x0_trials),
        seed=int(args.seed),
        trial_start=int(args.trial_start),
        trial_count=int(args.trial_count),
        outdir=str(args.outdir),
        checkpoint_every=int(args.checkpoint_every),
        ab_lo=float(args.ab_lo),
        ab_hi=float(args.ab_hi),
        theta_sigma=float(args.theta_sigma),
        adam_epochs=int(args.stage4_adam),
        lbfgs_epochs=int(args.stage4_lbfgs),
        adam_eta=float(args.adam_eta),
        dt0_div=float(args.dt0_div),
        max_steps=int(args.max_steps),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    # Save outputs
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
        "mode": best["mode"],
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
        title=f"Stage 4 best fit (synthetic) mode={best['mode']}"
    )
    print("[run] saved best plot.")


if __name__ == "__main__":
    main()
