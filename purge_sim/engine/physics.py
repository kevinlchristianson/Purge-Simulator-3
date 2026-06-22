"""
Pure physics functions for the purge simulator.
No side effects, no I/O — all functions are stateless and unit-testable.
Ported and cleaned from Purge_Modeling_Program_v29.py.
"""

import math
import numpy as np

from .constants import (
    ATM_PSI, STD_TEMP_F,
    N2_MW_KG_PER_MOL, N2_CRIT_T_K, N2_CRIT_P_PA, N2_ACENTRIC, N2_VISCOSITY_PA_S,
)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * (5.0 / 9.0) + 273.15


def psia_to_pa(p_psia: float) -> float:
    return float(p_psia) * 6894.757


def pa_to_psia(p_pa: float) -> float:
    return float(p_pa) / 6894.757


def psig_to_psia(p_psig: float) -> float:
    return float(p_psig) + ATM_PSI


def psia_to_psig(p_psia: float) -> float:
    return float(p_psia) - ATM_PSI


def ft3_to_m3(v_ft3: float) -> float:
    return float(v_ft3) * 0.028316846592


def bph_to_fts(bph: float, area_ft2: float) -> float:
    """Convert barrels per hour to feet per second through a given pipe cross-section."""
    # 1 bbl = 5.614583 ft³
    return float(bph) * 5.614583 / 3600.0 / max(1e-12, float(area_ft2))


def fts_to_bph(v_fts: float, area_ft2: float) -> float:
    """Convert pipe velocity (ft/s) to barrels per hour."""
    return float(v_fts) * float(area_ft2) * 3600.0 / 5.614583


def fts_to_mph(v_fts: float) -> float:
    return float(v_fts) * 3600.0 / 5280.0


def mph_to_fts(v_mph: float) -> float:
    return float(v_mph) * 5280.0 / 3600.0


def pipe_area_ft2(od_in: float, wt_in: float) -> float:
    """Internal cross-sectional area in ft²."""
    id_in = float(od_in) - 2.0 * float(wt_in)
    return math.pi / 4.0 * (id_in / 12.0) ** 2


def pipe_volume_ft3(od_in: float, wt_in: float, length_mi: float) -> float:
    """Internal volume in ft³ for a pipe segment of given length."""
    return pipe_area_ft2(od_in, wt_in) * float(length_mi) * 5280.0


def static_head_psi(elev_delta_ft: float, fluid_sg: float) -> float:
    """Pressure change (psi) due to elevation change. Positive when flowing uphill."""
    return float(elev_delta_ft) * float(fluid_sg) * 62.4 / 144.0


# ---------------------------------------------------------------------------
# Peng-Robinson EOS for nitrogen
# ---------------------------------------------------------------------------

def z_factor_n2(p_psia: float, t_f: float) -> float:
    """Nitrogen compressibility factor Z (Peng-Robinson EOS)."""
    p_pa = max(1.0, psia_to_pa(p_psia))
    t_k  = max(1.0, f_to_k(t_f))
    R    = 8.314462618

    kappa = 0.37464 + 1.54226 * N2_ACENTRIC - 0.26992 * N2_ACENTRIC ** 2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(t_k / N2_CRIT_T_K))) ** 2
    a = 0.45724 * R**2 * N2_CRIT_T_K**2 / N2_CRIT_P_PA
    b = 0.07780 * R * N2_CRIT_T_K / N2_CRIT_P_PA
    A = a * alpha * p_pa / (R**2 * t_k**2)
    B = b * p_pa / (R * t_k)

    roots = np.roots([1.0, -(1.0 - B), A - 3.0*B**2 - 2.0*B, -(A*B - B**2 - B**3)])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B]
    if not real_roots:
        return 1.0
    Z = max(real_roots)
    return float(Z) if math.isfinite(Z) and Z > 0 else 1.0


def n2_density_kg_m3(p_psia: float, t_f: float) -> float:
    """Nitrogen density (kg/m³) from Peng-Robinson EOS."""
    p_pa = psia_to_pa(p_psia)
    t_k  = f_to_k(t_f)
    Z    = z_factor_n2(p_psia, t_f)
    R    = 8.314462618
    return float(p_pa * N2_MW_KG_PER_MOL / (max(1e-12, Z) * R * t_k))


# ---------------------------------------------------------------------------
# N2 mole / SCF conversions (real-gas)
# ---------------------------------------------------------------------------

def n2_moles_from_pressure_volume(p_psia: float, V_ft3: float, t_f: float) -> float:
    """Moles of N2 in a vessel at given pressure, volume, and temperature."""
    p_pa = psia_to_pa(max(0.01, p_psia))
    V_m3 = ft3_to_m3(max(1e-12, V_ft3))
    t_k  = f_to_k(t_f)
    Z    = max(1e-6, z_factor_n2(p_psia, t_f))
    R    = 8.314462618
    return float(p_pa * V_m3 / (Z * R * t_k))


def n2_pressure_psia_from_moles(n_mol: float, V_ft3: float, t_f: float) -> float:
    """Pressure (psia) of N2 given moles, volume, and temperature (iterative PR)."""
    V_m3 = max(1e-12, ft3_to_m3(V_ft3))
    t_k  = f_to_k(t_f)
    R    = 8.314462618
    p_pa = max(1.0, n_mol * R * t_k / V_m3)
    for _ in range(25):
        Z     = z_factor_n2(pa_to_psia(p_pa), t_f)
        p_new = n_mol * R * t_k * Z / V_m3
        if abs(p_new - p_pa) / max(1.0, p_pa) < 1e-9:
            p_pa = p_new
            break
        p_pa = p_new
    return float(pa_to_psia(p_pa))


def scf_to_moles(scf: float) -> float:
    """Convert SCF at standard conditions (60°F, 14.7 psia) to moles of N2."""
    p_std_pa = psia_to_pa(ATM_PSI)
    t_std_k  = f_to_k(STD_TEMP_F)
    V_m3     = ft3_to_m3(float(scf))
    Z_std    = z_factor_n2(ATM_PSI, STD_TEMP_F)
    R        = 8.314462618
    return float(p_std_pa * V_m3 / (max(1e-12, Z_std) * R * t_std_k))


def moles_to_scf(n_mol: float) -> float:
    """Convert moles of N2 to SCF at standard conditions (60°F, 14.7 psia)."""
    p_std_pa = psia_to_pa(ATM_PSI)
    t_std_k  = f_to_k(STD_TEMP_F)
    Z_std    = z_factor_n2(ATM_PSI, STD_TEMP_F)
    R        = 8.314462618
    V_m3     = n_mol * Z_std * R * t_std_k / p_std_pa
    return float(V_m3 / 0.028316846592)


def pressure_psia_from_scf(scf: float, V_ft3: float, t_f: float) -> float:
    """Pressure (psia) of N2 given SCF inventory, vessel volume, and temperature."""
    n_mol = scf_to_moles(scf)
    return n2_pressure_psia_from_moles(n_mol, V_ft3, t_f)


def scf_from_pressure_volume(p_psia: float, V_ft3: float, t_f: float) -> float:
    """SCF of N2 at standard conditions given pressure, volume, and temperature."""
    n_mol = n2_moles_from_pressure_volume(p_psia, V_ft3, t_f)
    return moles_to_scf(n_mol)


# ---------------------------------------------------------------------------
# Friction losses (Darcy-Weisbach, Swamee-Jain / Haaland)
# ---------------------------------------------------------------------------

def _darcy_friction_factor(Re: float, eps_m: float, D_m: float) -> float:
    """Darcy friction factor via Swamee-Jain (turbulent) or Hagen-Poiseuille (laminar)."""
    if Re < 2300.0:
        return 64.0 / max(1.0, Re)
    return 0.25 / (math.log10(eps_m / (3.7 * D_m) + 5.74 / Re**0.9)) ** 2


def gas_friction_loss_psi(L_ft: float, D_ft: float, v_fts: float,
                           p_avg_psia: float, t_f: float, eps_ft: float) -> float:
    """Darcy-Weisbach dp (psi) for N2 gas over length L at average pressure p_avg."""
    if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
        return 0.0
    m = 0.3048
    L, D, v, eps = L_ft*m, D_ft*m, v_fts*m, eps_ft*m
    rho = n2_density_kg_m3(p_avg_psia, t_f)
    Re  = rho * v * D / max(1e-12, N2_VISCOSITY_PA_S)
    f   = _darcy_friction_factor(Re, eps, D)
    return float(f * (L / D) * 0.5 * rho * v**2 / 6894.757)


def liquid_friction_loss_psi(L_ft: float, D_ft: float, v_fts: float,
                              sg: float, viscosity_cst: float, eps_ft: float) -> float:
    """Darcy-Weisbach dp (psi) for liquid slug ahead of the pig."""
    if L_ft <= 0 or v_fts <= 0 or D_ft <= 0:
        return 0.0
    m = 0.3048
    L, D, v, eps = L_ft*m, D_ft*m, v_fts*m, eps_ft*m
    rho = 999.0 * float(sg)
    nu  = float(viscosity_cst) * 1e-6
    Re  = v * D / max(1e-12, nu)
    f   = _darcy_friction_factor(Re, eps, D)
    return float(f * (L / D) * 0.5 * rho * v**2 / 6894.757)


# ---------------------------------------------------------------------------
# SCFM injection rate
# ---------------------------------------------------------------------------

def scfm_from_pig_velocity(v_fts: float, area_ft2: float, p_pig_face_psia: float,
                            t_f: float) -> float:
    """
    SCFM of N2 needed to sustain pig displacement at velocity v_fts.

    Uses pig-face pressure density (not average of injection + pig face),
    which is correct when boosters decouple injection pressure from pig pressure.
    The mass flow equals rho_at_pig_face * area * velocity; convert to standard ft³/min.
    """
    if v_fts <= 0 or area_ft2 <= 0:
        return 0.0
    m = 0.3048
    area_m2  = float(area_ft2) * m**2
    v_m_s    = float(v_fts) * m
    rho_pig  = n2_density_kg_m3(float(p_pig_face_psia), t_f)
    m_dot    = rho_pig * v_m_s * area_m2

    rho_std  = n2_density_kg_m3(ATM_PSI, STD_TEMP_F)
    q_std_m3_s = m_dot / max(1e-12, rho_std)
    return float(q_std_m3_s * 35.3146667 * 60.0)


# ---------------------------------------------------------------------------
# Geospatial
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1) * math.cos(la2) * math.sin(dlon/2)**2
    return R * 2.0 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Drive envelope and exit pressure helpers (ported from v29)
# ---------------------------------------------------------------------------

def compute_safe_max_drive_envelope(
    pig_mileposts: np.ndarray,
    pig_elevations_ft: np.ndarray,
    system_mileposts: np.ndarray,
    system_elevations_ft: np.ndarray,
    fluid_sg: float,
    maop_psig: float,
    max_drive_psig: float,
) -> np.ndarray:
    """
    Per-position safe max behind-pig gas pressure (psig).
    Conservative: considers lowest elevation downstream from pig + fluid head.
    """
    maop      = float(maop_psig)
    max_drive = float(max_drive_psig)

    if not math.isfinite(maop) or maop <= 0:
        return np.full(len(pig_mileposts), max_drive, dtype=float)

    grad     = float(fluid_sg) * 62.4 / 144.0
    sys_mps  = np.asarray(system_mileposts, dtype=float)
    sys_elev = np.asarray(system_elevations_ft, dtype=float)
    if sys_mps.size == 0:
        return np.full(len(pig_mileposts), min(max_drive, maop), dtype=float)

    min_elev_from = np.minimum.accumulate(sys_elev[::-1])[::-1]

    pig_mps  = np.asarray(pig_mileposts, dtype=float)
    pig_elev = np.asarray(pig_elevations_ft, dtype=float)
    out      = np.empty_like(pig_mps, dtype=float)

    for i, mp in enumerate(pig_mps):
        j          = int(np.searchsorted(sys_mps, mp, side="left"))
        j          = max(0, min(j, len(min_elev_from) - 1))
        min_ahead  = float(min_elev_from[j])
        head_delta = grad * (float(pig_elev[i]) - min_ahead)
        safe       = maop - max(0.0, head_delta)
        out[i]     = max(0.0, min(safe, maop, max_drive))

    return out


def target_exit_pressure(mp: float, cfg: dict,
                          purge_start: float, purge_end: float,
                          throttle_down: float, t_hours: float = None) -> float:
    """
    Outlet/endpoint pressure (psig) based on the selected behavior program.
    Behaviors: taper_last_n_miles (default), step_last_n_miles, linear_ramp,
               constant_run, constant_end.
    """
    run_p = float(cfg.get('exit_pressure_run', 0.0))
    end_p = float(cfg.get('exit_pressure_end', run_p))
    behavior = str(cfg.get('exit_pressure_behavior', 'taper_last_n_miles')).strip().lower()

    aliases = {
        'taper':     'taper_last_n_miles',
        'taper_last':'taper_last_n_miles',
        'linear':    'linear_ramp',
        'ramp':      'linear_ramp',
        'hold_run':  'constant_run',
        'hold_end':  'constant_end',
        'step':      'step_last_n_miles',
    }
    behavior = aliases.get(behavior, behavior)

    purge_len     = max(1e-12, float(purge_end) - float(purge_start))
    dist_from_end = float(purge_end) - float(mp)
    td            = max(0.0, float(throttle_down or 0.0))

    if behavior == 'constant_end':
        return end_p
    if behavior == 'constant_run':
        return run_p
    if behavior == 'linear_ramp':
        frac = max(0.0, min(1.0, (float(mp) - float(purge_start)) / purge_len))
        return run_p * (1.0 - frac) + end_p * frac
    if behavior == 'step_last_n_miles':
        return end_p if (td > 0.0 and dist_from_end <= td) else run_p

    # Default: taper_last_n_miles
    if td > 0.0 and dist_from_end <= td:
        taper = max(0.0, min(1.0, dist_from_end / td))
        return run_p * taper + end_p * (1.0 - taper)
    return run_p
