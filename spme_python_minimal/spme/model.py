from dataclasses import dataclass
import numpy as np
from .parameters import Geometry, PorosityBrugg, Transport, Kinetics, Thermo, ElectrodeProps, Discretization
from .ocp import U_p, U_n

@dataclass
class SPMeParams:
    geom: Geometry = Geometry()
    por: PorosityBrugg = PorosityBrugg()
    tr: Transport = Transport()
    kin: Kinetics = Kinetics()
    thermo: Thermo = Thermo()
    props: ElectrodeProps = ElectrodeProps()
    disc: Discretization = Discretization()

class SPMe:
    """
    A compact Single Particle Model with electrolyte (SPMe).
    - Solid diffusion in each electrode uses a parabolic profile approximation:
        dc_avg/dt = -3*j/R
        c_s_surf = c_avg - j*R/(5*D_s)
    - Electrolyte 1D diffusion/migration reduced to diffusion with source terms due to reaction:
        eps * dc_e/dt = d/dx(D_eff dc_e/dx) + (1 - t_plus) * a * j
      with piecewise-constant D_eff and source in electrodes only.
    - Terminal voltage:
        V = U_p(theta_p_s) - U_n(theta_n_s) + eta_p - eta_n
            + i_app*(l_p/sigma_p + l_n/sigma_n)
            + i_app*(l_p/kappa_eff_p + l_s/kappa_eff_s + l_n/kappa_eff_n)
            + 2*R*T/F*(1 - t_plus)*np.log(c_e_n/c_e_p)
    where i_app is current density [A/m^2], positive for discharge.
    """
    def __init__(self, params: SPMeParams = SPMeParams()):
        self.p = params
        # geometry helpers
        g, por = self.p.geom, self.p.por
        self.a_p = 3*(1-por.eps_p)/g.R_p
        self.a_n = 3*(1-por.eps_n)/g.R_n

    def D_eff_regions(self):
        p = self.p
        De = p.tr.D_e
        De_p = De * (p.por.eps_p ** p.por.brug_p)
        De_s = De * (p.por.eps_s ** p.por.brug_s)
        De_n = De * (p.por.eps_n ** p.por.brug_n)
        return De_p, De_s, De_n

    def kappa_eff_regions(self):
        p = self.p
        kap = p.tr.kappa
        k_p = kap * (p.por.eps_p ** p.por.brug_p)
        k_s = kap * (p.por.eps_s ** p.por.brug_s)
        k_n = kap * (p.por.eps_n ** p.por.brug_n)
        return k_p, k_s, k_n

    def pack_grid(self):
        d = self.p.disc
        g = self.p.geom
        N = d.N_x
        # allocate nodes by thickness proportion
        L = g.l_p + g.l_s + g.l_n
        Np = max(3, int(np.round(N * g.l_p / L)))
        Ns = max(3, int(np.round(N * g.l_s / L)))
        Nn = max(3, N - Np - Ns)
        # build x and region ids
        xs_p = np.linspace(0.0, g.l_p, Np, endpoint=False)
        xs_s = np.linspace(g.l_p, g.l_p+g.l_s, Ns, endpoint=False)
        xs_n = np.linspace(g.l_p+g.l_s, L, Nn)
        x = np.concatenate([xs_p, xs_s, xs_n])
        reg = np.array(['p']*len(xs_p) + ['s']*len(xs_s) + ['n']*len(xs_n))
        return x, reg

    def initial_state(self, theta_p0=0.9, theta_n0=0.1):
        # states: [c_e (N), c_s_avg_p, c_s_avg_n]
        x, reg = self.pack_grid()
        ce0 = np.full_like(x, self.p.props.c_e0, dtype=float)
        cs_avg_p = theta_p0 * self.p.props.c_s_max_p
        cs_avg_n = theta_n0 * self.p.props.c_s_max_n
        y0 = np.concatenate([ce0, [cs_avg_p, cs_avg_n]])
        return y0

    def split_state(self, y):
        x, reg = self.pack_grid()
        N = len(x)
        ce = y[:N]
        cs_avg_p, cs_avg_n = y[N], y[N+1]
        return ce, cs_avg_p, cs_avg_n

    def reaction_fluxes(self, i_app):
        # Uniform reaction flux per electrode from current density
        p = self.p
        jp =  i_app / (p.thermo.F * self.a_p * p.geom.l_p)
        jn = -i_app / (p.thermo.F * self.a_n * p.geom.l_n)
        return jp, jn

    def solid_surface_conc(self, cs_avg_p, cs_avg_n, jp, jn):
        p = self.p
        cs_s_p = cs_avg_p - jp * p.geom.R_p / (5*p.tr.D_s_p)
        cs_s_n = cs_avg_n - jn * p.geom.R_n / (5*p.tr.D_s_n)
        # clamp to physical bounds
        cs_s_p = np.clip(cs_s_p, 1e-12, p.props.c_s_max_p-1e-12)
        cs_s_n = np.clip(cs_s_n, 1e-12, p.props.c_s_max_n-1e-12)
        return cs_s_p, cs_s_n

    def overpotentials(self, ce_p, ce_n, cs_s_p, cs_s_n, jp, jn):
        p, th = self.p, self.p.thermo
        # Butler-Volmer inverse: eta = (2RT/F/alpha) * asinh( j / (2k sqrt(ce) sqrt(cs*(cmax-cs))) )
        rtF = th.R*th.T/th.F
        denom_p = 2*p.kin.k_p*np.sqrt(max(ce_p,1e-12))*np.sqrt(cs_s_p*(p.props.c_s_max_p-cs_s_p))
        denom_n = 2*p.kin.k_n*np.sqrt(max(ce_n,1e-12))*np.sqrt(cs_s_n*(p.props.c_s_max_n-cs_s_n))
        eta_p = (2*rtF/p.kin.alpha)*np.arcsinh(jp/denom_p)
        eta_n = (2*rtF/p.kin.alpha)*np.arcsinh(jn/denom_n)
        return eta_p, eta_n

    def voltage(self, y, i_app):
        x, reg = self.pack_grid()
        ce, cs_avg_p, cs_avg_n = self.split_state(y)
        # region boundary concentrations
        ce_p = float(np.mean(ce[reg=='p'][:2])) if np.sum(reg=='p')>=2 else float(ce[reg=='p'][0])
        ce_n = float(np.mean(ce[reg=='n'][-2:])) if np.sum(reg=='n')>=2 else float(ce[reg=='n'][-1])
        jp, jn = self.reaction_fluxes(i_app)
        cs_s_p, cs_s_n = self.solid_surface_conc(cs_avg_p, cs_avg_n, jp, jn)
        # OCPs
        theta_p = cs_s_p / self.p.props.c_s_max_p
        theta_n = cs_s_n / self.p.props.c_s_max_n
        Up = U_p(theta_p)
        Un = U_n(theta_n)
        # overpotentials
        eta_p, eta_n = self.overpotentials(ce_p, ce_n, cs_s_p, cs_s_n, jp, jn)
        # ohmic drops
        k_p, k_s, k_n = self.kappa_eff_regions()
        sig_p, sig_n = self.p.tr.sigma_p, self.p.tr.sigma_n
        drop_solid = i_app*(self.p.geom.l_p/sig_p + self.p.geom.l_n/sig_n)
        drop_elec  = i_app*(self.p.geom.l_p/k_p + self.p.geom.l_s/k_s + self.p.geom.l_n/k_n)
        # concentration term
        rtF = self.p.thermo.R*self.p.thermo.T/self.p.thermo.F
        conc_term = 2*rtF*(1-self.p.tr.t_plus)*np.log(max(ce_n,1e-12)/max(ce_p,1e-12))
        V = Up - Un + (eta_p - eta_n) + drop_solid + drop_elec + conc_term
        return float(V)

    def rhs(self, t, y, i_app):
        x, reg = self.pack_grid()
        N = len(x)
        ce, cs_avg_p, cs_avg_n = self.split_state(y)
        De_p, De_s, De_n = self.D_eff_regions()
        eps_p, eps_s, eps_n = self.p.por.eps_p, self.p.por.eps_s, self.p.por.eps_n
        # spacings (uniform per region)
        g = self.p.geom
        Np = np.sum(reg=='p'); Ns = np.sum(reg=='s'); Nn = np.sum(reg=='n')
        hx_p = g.l_p / max(Np-1,1)
        hx_s = g.l_s / max(Ns-1,1)
        hx_n = g.l_n / max(Nn-1,1)

        # Build second derivative with Neumann (zero flux) at current collectors and continuity at interfaces
        dce_dt = np.zeros_like(ce)
        def laplace(u, h, D):
            # central with Neumann at ends
            L = np.zeros_like(u)
            L[0]   = (u[1] - u[0]) / (h*h)      # du/dx = 0 -> ghost u[-1]=u[0]
            L[-1]  = (u[-2] - u[-1]) / (h*h)    # ghost u[+1]=u[-1]
            if len(u) > 2:
                L[1:-1] = (u[0:-2] - 2*u[1:-1] + u[2:])/(h*h)
            return D*L

        # split regions
        up = ce[reg=='p']
        us = ce[reg=='s']
        un = ce[reg=='n']
        Lp = laplace(up, hx_p, De_p)
        Ls = laplace(us, hx_s, De_s)
        Ln = laplace(un, hx_n, De_n)

        # source terms in electrodes
        jp, jn = self.reaction_fluxes(i_app)
        src_p = (1 - self.p.tr.t_plus)*self.a_p*jp / eps_p
        src_n = (1 - self.p.tr.t_plus)*self.a_n*jn / eps_n

        dce_dt[reg=='p'] = Lp/eps_p + src_p
        dce_dt[reg=='s'] = Ls/eps_s
        dce_dt[reg=='n'] = Ln/eps_n + src_n

        # solid averages
        dcs_p = -3*jp / g.R_p
        dcs_n = -3*jn / g.R_n

        dydt = np.concatenate([dce_dt, [dcs_p, dcs_n]])
        return dydt
