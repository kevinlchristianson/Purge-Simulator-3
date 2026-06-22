"""
Pipeline and fluid constants, lookup tables.
Ported from Purge_Modeling_Program_v29.py.
"""

# Atmospheric pressure at standard conditions
ATM_PSI = 14.7

# Standard conditions (SCFM reference)
STD_TEMP_F = 60.0
STD_PRESS_PSIA = ATM_PSI

# Nitrogen molecular / critical properties (Peng-Robinson EOS)
N2_MW_KG_PER_MOL = 0.0280134
N2_CRIT_T_K      = 126.192
N2_CRIT_P_PA     = 3.3958e6   # ~492.3 psia
N2_ACENTRIC      = 0.0372
N2_VISCOSITY_PA_S = 1.75e-5

# Default nitrogen temperature at which the gas column operates
N2_TEMP_F_DEFAULT = 45.0

# NPS lookup: nominal pipe size string → outside diameter (inches)
NPS_OD_IN: dict[str, float] = {
    "1/8":  0.405, "1/4":  0.540, "3/8":  0.675, "1/2":  0.840,
    "3/4":  1.050, "1":    1.315, "1 1/4": 1.660, "1 1/2": 1.900,
    "2":    2.375, "2 1/2": 2.875, "3":   3.500, "3 1/2": 4.000,
    "4":    4.500, "5":    5.563, "6":    6.625, "8":    8.625,
    "10":   10.750,"12":  12.750, "14":  14.000, "16":  16.000,
    "18":   18.000,"20":  20.000, "22":  22.000, "24":  24.000,
    "26":   26.000,"28":  28.000, "30":  30.000, "32":  32.000,
    "34":   34.000,"36":  36.000, "40":  40.000, "42":  42.000,
    "44":   44.000,"48":  48.000, "52":  52.000, "56":  56.000,
    "60":   60.000,"64":  64.000, "68":  68.000, "72":  72.000,
    "80":   80.000,
}

# Roughness lookup: index → (label, roughness in feet)
ROUGHNESS: dict[int, tuple[str, float]] = {
    1: ("New Welded Steel",            0.00015),
    2: ("Rusted/Corroded Welded Steel", 0.0005),
    3: ("Welded HDPE",                 0.000005),
}

# Fluid lookup: index → {name, sg, viscosity_cst}
# sg and viscosity_cst are None for crude oil (user-supplied API / viscosity)
FLUIDS: dict[int, dict] = {
    1: {"name": "Diesel",        "sg": 0.84, "viscosity_cst": 2.7},
    2: {"name": "Gasoline",      "sg": 0.74, "viscosity_cst": 0.6},
    3: {"name": "Crude Oil",     "sg": None, "viscosity_cst": None},
    4: {"name": "Water",         "sg": 1.0,  "viscosity_cst": 1.0},
    5: {"name": "NGL (Y1-grade)","sg": 0.6,  "viscosity_cst": 0.3},
}

# Column name aliases for flexible TXT/CSV/ILI parser (all lowercase)
COL_LAT  = ['latitude', 'lat', 'gps_lat', 'gps latitude', 'gps lat']
COL_LON  = ['longitude', 'lon', 'long', 'gps_lon', 'gps longitude', 'gps lon']
COL_ELEV = ['altitude (ft)', 'altitude', 'elevation', 'elev', 'alt', 'height',
             'altitude_ft', 'elevation_ft', 'elev_ft', 'altitude (m)', 'elevation (m)',
             'altitude(ft)', 'elevation(ft)', 'z', 'gps_alt', 'gps altitude']
COL_MP   = ['milepost', 'log distance', 'log_distance', 'chainage', 'distance',
             'mp', 'station', 'odometer', 'dist_mi', 'dist_ft', 'km', 'meters',
             'log dist', 'log_dist', 'calculated mile post [mi.]']
COL_OD   = ['od', 'outside diameter', 'pipe od', 'outer diameter', 'nominal od',
             'od_in', 'od (in)', 'diameter (in)', 'diameter_in', 'o.d.', 'o.d. (in)',
             'pipe diameter [in.]']
COL_WT   = ['wall thickness', 'wt', 'wall_thickness', 'nominal wt', 'wt_in',
             'wt (in)', 'thickness', 'spec wt', 'wall thickness (in)', 'wall_thickness_in',
             'nom. wt', 'nom wt', 'spec. wt',
             'pipe nominal wall thickness [in.]']
COL_MOP  = ['maximum operating pressure (mop) [psi]', 'mop', 'maop',
             'max operating pressure', 'maximum operating pressure']


def resolve_nps(nps_val) -> str:
    """Map a GUI NPS value (str or float) to a key present in NPS_OD_IN."""
    if nps_val in NPS_OD_IN:
        return str(nps_val)
    try:
        as_float = float(nps_val)
        cand = str(int(round(as_float)))
        if cand in NPS_OD_IN:
            return cand
        for k in NPS_OD_IN:
            try:
                if abs(float(k) - as_float) < 1e-9:
                    return k
            except ValueError:
                continue
    except (TypeError, ValueError):
        pass
    k = str(nps_val)
    if k in NPS_OD_IN:
        return k
    raise KeyError(f"NPS '{nps_val}' not found. Available: {list(NPS_OD_IN.keys())}")


def od_in_from_nps(nps_val) -> float:
    """Return OD in inches for a given NPS string or numeric value."""
    return NPS_OD_IN[resolve_nps(nps_val)]


def roughness_ft(roughness_num: int) -> float:
    """Return roughness in feet for a roughness index (1/2/3)."""
    if roughness_num not in ROUGHNESS:
        raise KeyError(f"Roughness index {roughness_num} not found. Use 1, 2, or 3.")
    return ROUGHNESS[roughness_num][1]
