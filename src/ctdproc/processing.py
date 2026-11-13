import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import gsw

from seabirdscientific import processing as proc

logger = logging.getLogger(__name__)

BAD_FLAG_VALUE = np.float64(-9.99e-29)


STEP_DESCRIPTIONS = {
    "align": "time-shift a sensor to correct lag relative to another (e.g. conductivity cell lag behind temperature)",
    "wild_edit": "flag outlier scans via a two-pass standard-deviation test",
    "low_filter": "low-pass filter to smooth high-frequency noise",
    "celltm": "conductivity cell thermal-mass correction + recomputed salinity",
    "slope_corr": "scale conductivity by a fitted slope (gain) + recomputed salinity",
    "loop_edit": "flag scans affected by ship-roll-induced pressure reversals/slowdowns",
    "bin": "bin-average the cast by elapsed_time/scans/pressure",
}


def validate_features(features, valid_names):
    """Check every `features:` step name against valid_names.

    INPUT:
      features     =  step dicts, each with a 'name' key    [ list ]
      valid_names  =  allowed step names                     [ list[str] ]

    OUTPUT:
      raises ValueError listing every invalid name, otherwise None
    """
    bad_names = sorted(
        {
            step.get("name")
            for step in (features or [])
            if step.get("name") not in valid_names
        },
        key=str,
    )
    if bad_names:
        options = "\n".join(
            f"  - {name}: {STEP_DESCRIPTIONS.get(name, '(no description)')}"
            for name in valid_names
        )
        raise ValueError(
            f"config error: unknown features step name(s) {bad_names}. "
            f"Valid options here are:\n{options}"
        )


def check_step_lists(step_name, **lists):
    """Check that same-named config lists for one step all have equal length.

    INPUT:
      step_name  =  name of the step, for the error message    [ str ]
      **lists    =  config lists to compare, by name            [ list ]

    OUTPUT:
      raises ValueError if lengths differ, otherwise None
    """
    lengths = {name: len(lst) for name, lst in lists.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in lengths.items())
        raise ValueError(
            f"config error in '{step_name}' step: list lengths don't match "
            f"({detail}). Check for a misspelled key or a missing/extra "
            f"entry - every list under this step needs the same number of "
            f"entries, in the same order."
        )


def filt_interp(
    data: pd.DataFrame, param: str, tm: float, si: float, flag: bool
) -> np.ndarray:
    """Low-pass filter one variable, after masking and interpolating flagged scans.

    INPUT:
      data   =  cast data                                [ DataFrame ]
      param  =  column name to filter                     [ str ]
      tm     =  filter time constant                       [ s ]
      si     =  sample interval                            [ s ]
      flag   =  restore bad_flag_value at masked spots      [ bool ]

    OUTPUT:
      filtered values for `param`, same length as data      [ array ]
    """
    data_i = data[param].copy()
    indx = data_i == BAD_FLAG_VALUE
    data_i[indx] = np.nan

    data_i = data_i.interpolate(limit_direction="both")

    data_interp = proc.low_pass_filter(
        x=data_i.values, time_constant=tm, sample_interval=si
    )

    if flag:
        data_interp[indx] = BAD_FLAG_VALUE

    return data_interp


def crosshigh(
    data: pd.DataFrame,
    param: list,
    maxL: int,
    pi: float,
    pf: float,
    si: float,
    high: bool,
) -> list:
    """Cross-correlate two variables against each other, separately for the
    down- and up-cast.

    INPUT:
      data   =  cast data                                 [ DataFrame ]
      param  =  [pressure, var1, var2] column names         [ list[str] ]
      maxL   =  maximum lag to search                        [ scans ]
      pi     =  lower bound of param[0]                      [ param[0] units ]
      pf     =  upper bound of param[0]                       [ param[0] units ]
      si     =  sample interval                               [ s ]
      high   =  high-pass filter both variables first         [ bool ]

    OUTPUT:
      [up_corr, up_lag, down_corr, down_lag] - peak correlation
      and lag, each cast direction                             [ list ]
    """
    data_i = data.copy()
    indx1 = data_i[param[1]] == BAD_FLAG_VALUE
    indx2 = data_i[param[2]] == BAD_FLAG_VALUE

    data_i.loc[indx1, param[1]] = np.nan
    data_i.loc[indx2, param[2]] = np.nan
    data_i[param[1]] = data_i[param[1]].interpolate(limit_direction="both")
    data_i[param[2]] = data_i[param[2]].interpolate(limit_direction="both")

    peak_pressure_idx = np.argmax(data_i[param[0]])

    fpar1_dn = data_i[param[1]][0:peak_pressure_idx]
    fpar1_up = data_i[param[1]][peak_pressure_idx + 1 :]
    fpar2_dn = data_i[param[2]][0:peak_pressure_idx]
    fpar2_up = data_i[param[2]][peak_pressure_idx + 1 :]
    fpar0_dn = data_i[param[0]][0:peak_pressure_idx]
    fpar0_up = data_i[param[0]][peak_pressure_idx + 1 :]

    if high:
        b, a = butter(3, 0.005)
        fpar1_dn_smoothed = filtfilt(b, a, fpar1_dn)
        fpar1_up_smoothed = filtfilt(b, a, fpar1_up)
        fpar2_dn_smoothed = filtfilt(b, a, fpar2_dn)
        fpar2_up_smoothed = filtfilt(b, a, fpar2_up)

        fpar1_dn = fpar1_dn - fpar1_dn_smoothed
        fpar1_up = fpar1_up - fpar1_up_smoothed
        fpar2_dn = fpar2_dn - fpar2_dn_smoothed
        fpar2_up = fpar2_up - fpar2_up_smoothed

    mask_dn = (fpar0_dn >= pi) & (fpar0_dn <= pf)
    mask_up = (fpar0_up >= pi) & (fpar0_up <= pf)

    lag_dn, corre_dn, _, _ = plt.xcorr(
        fpar1_dn[mask_dn], fpar2_dn[mask_dn], maxlags=maxL
    )
    lag_up, corre_up, _, _ = plt.xcorr(
        fpar1_up[mask_up], fpar2_up[mask_up], maxlags=maxL
    )

    best_corr = [
        corre_up[np.argmax(np.abs(corre_up))],
        si * lag_up[np.argmax(np.abs(corre_up))],
        corre_dn[np.argmax(np.abs(corre_dn))],
        si * lag_dn[np.argmax(np.abs(corre_dn))],
    ]
    return best_corr


def alp_tau_fast(
    p_dn, t_dn, c_dn, 
    p_up, t_up, c_up, 
    temp_bins
) -> float:
    """Score one cell-thermal-mass correction: bin down- and up-cast
    salinity by temperature and compare.

    INPUT:
      p_dn, t_dn, c_dn  =  down-cast pressure, temperature,
                           conductivity                     [ dbar, degC, S/m ]
      p_up, t_up, c_up  =  same, up-cast                     [ dbar, degC, S/m ]
      temp_bins         =  temperature bin edges              [ array, degC ]

    OUTPUT:
      mean absolute difference between down- and up-cast
      binned salinity                                          [ float ]
    """


    sal_dn = gsw.SP_from_C(10 * c_dn, t=t_dn, p=p_dn)
    sal_up = gsw.SP_from_C(10 * c_up, t=t_up, p=p_up)


    sal_dn = np.nan_to_num(sal_dn, nan=np.nan)
    sal_up = np.nan_to_num(sal_up, nan=np.nan)


    counts_dn, _ = np.histogram(t_dn, bins=temp_bins)
    counts_up, _ = np.histogram(t_up, bins=temp_bins)
    
    sum_sal_dn, _ = np.histogram(t_dn, bins=temp_bins, weights=sal_dn)
    sum_sal_up, _ = np.histogram(t_up, bins=temp_bins, weights=sal_up)


    with np.errstate(divide='ignore', invalid='ignore'):
        avg_sal_dn = sum_sal_dn / counts_dn
        avg_sal_up = sum_sal_up / counts_up


    valid_mask = (counts_dn > 0) & (counts_up > 0) & (~np.isnan(avg_sal_dn)) & (~np.isnan(avg_sal_up))

    if not np.any(valid_mask):
        return np.nan


    return float(np.mean(np.abs(avg_sal_dn[valid_mask] - avg_sal_up[valid_mask])))




def find_opt_alp_tat_fast(
    alpha_r: np.ndarray,
    tau_r: np.ndarray,
    data: pd.DataFrame,
    param: list,
    pi: float,
    pf: float,
    tbin: float,
    sample_interval: float,
) -> np.ndarray:
    """Grid-search cell_thermal_mass's (alpha, tau) over alpha_r x tau_r.

    INPUT:
      alpha_r          =  amplitude values to search            [ array ]
      tau_r            =  time-constant values to search         [ array, s ]
      data             =  cast data                               [ DataFrame ]
      param            =  [pressure, temperature, conductivity]
                          column names                             [ list[str] ]
      pi               =  lower pressure bound                     [ dbar ]
      pf               =  upper pressure bound                      [ dbar ]
      tbin             =  temperature bin width                     [ degC ]
      sample_interval  =  time between scans                        [ s ]

    OUTPUT:
      error matrix, shape (len(alpha_r), len(tau_r))                 [ array ]
    """

    data_i = data.copy()

    for p in param[:3]:
        data_i.loc[data_i[p] == BAD_FLAG_VALUE, p] = np.nan
        data_i[p] = data_i[p].interpolate(limit_direction="both")


    pres = data_i[param[0]].values
    temp = data_i[param[1]].values
    cond = data_i[param[2]].values


    peak_idx = np.argmax(pres)

    p_dn_raw, p_up_raw = pres[:peak_idx], pres[peak_idx + 1:]
    t_dn_raw, t_up_raw = temp[:peak_idx], temp[peak_idx + 1:]
    c_dn_raw, c_up_raw = cond[:peak_idx], cond[peak_idx + 1:]


    mask_dn = (p_dn_raw >= pi) & (p_dn_raw <= pf)
    mask_up = (p_up_raw >= pi) & (p_up_raw <= pf)


    p_dn, t_dn, c_dn = p_dn_raw[mask_dn], t_dn_raw[mask_dn], c_dn_raw[mask_dn]
    p_up, t_up, c_up = p_up_raw[mask_up], t_up_raw[mask_up], c_up_raw[mask_up]


    min_max = np.min([t_dn.max(), t_up.max()])
    max_min = np.max([t_dn.min(), t_up.min()])
    temp_bins = np.arange(max_min, min_max + tbin, tbin)


    error_matrix = np.full((len(alpha_r), len(tau_r)),np.nan)


    for r, alpha in enumerate(alpha_r):
        for c, tau in enumerate(tau_r):

            c_dn_corr = proc.cell_thermal_mass(
                temperature_C=t_dn, conductivity_Sm=c_dn,
                amplitude=alpha, time_constant=tau, sample_interval=sample_interval
            )
            c_up_corr = proc.cell_thermal_mass(
                temperature_C=t_up, conductivity_Sm=c_up,
                amplitude=alpha, time_constant=tau, sample_interval=sample_interval
            )

            error_matrix[r, c] = alp_tau_fast(
                p_dn, t_dn, c_dn_corr,
                p_up, t_up, c_up_corr,
                temp_bins
            )

    return error_matrix




def alp_tau(
        data: pd.DataFrame, param: list, pi: float, pf: float, tbin: float, figure: bool = False
) -> float:
    """Score one cell-thermal-mass correction already applied to `data`:
    bin down- and up-cast salinity by temperature and compare.

    INPUT:
      data    =  cast data, with correction already applied    [ DataFrame ]
      param   =  [pressure, temperature, conductivity]
                 column names                                    [ list[str] ]
      pi      =  lower pressure bound                             [ dbar ]
      pf      =  upper pressure bound                              [ dbar ]
      tbin    =  temperature bin width                             [ degC ]
      figure  =  plot down/up-cast salinity profiles                [ bool ]

    OUTPUT:
      mean absolute difference between down- and up-cast
      binned salinity                                                [ float ]
    """
    data_i = data.copy()
    indx1 = data_i[param[1]] == BAD_FLAG_VALUE
    indx2 = data_i[param[2]] == BAD_FLAG_VALUE

    data_i.loc[indx1, param[1]] = np.nan
    data_i.loc[indx2, param[2]] = np.nan
    data_i[param[1]] = data_i[param[1]].interpolate(limit_direction="both")
    data_i[param[2]] = data_i[param[2]].interpolate(limit_direction="both")
    data_i[param[2]] = gsw.SP_from_C(10*data_i[param[2]].values, t=data_i[param[1]].values, p=data_i[param[0]].values)
    
    peak_pressure_idx = np.argmax(data_i[param[0]])

    fpar1_dn = data_i[param[1]][0:peak_pressure_idx]
    fpar1_up = data_i[param[1]][peak_pressure_idx + 1 :]
    fpar2_dn = data_i[param[2]][0:peak_pressure_idx]
    fpar2_up = data_i[param[2]][peak_pressure_idx + 1 :]
    fpar0_dn = data_i[param[0]][0:peak_pressure_idx]
    fpar0_up = data_i[param[0]][peak_pressure_idx + 1 :]

    mask_dn = (fpar0_dn >= pi) & (fpar0_dn <= pf)
    mask_up = (fpar0_up >= pi) & (fpar0_up <= pf)

    min_max = np.min([fpar1_dn[mask_dn].max(), fpar1_up[mask_up].max()])
    max_min = np.max([fpar1_dn[mask_dn].min(), fpar1_up[mask_up].min()])
    temp_group = np.arange(max_min, min_max, tbin)

    pro_dn = pd.DataFrame({param[1]: fpar1_dn[mask_dn], param[2]: fpar2_dn[mask_dn]})
    pro_up = pd.DataFrame({param[1]: fpar1_up[mask_up], param[2]: fpar2_up[mask_up]})

    pro_dn[param[1] + "group"] = pd.cut(pro_dn[param[1]], bins=temp_group)
    pro_up[param[1] + "group"] = pd.cut(pro_up[param[1]], bins=temp_group)

    avg_fpar2_dn = pro_dn.groupby(
        param[1] + "group", observed=False
    )[param[2]].mean().reset_index()
    avg_fpar2_up = pro_up.groupby(
        param[1] + "group", observed=False
    )[param[2]].mean().reset_index()

    avg_fpar2_dn["temp_mid"] = (
        avg_fpar2_dn[param[1] + "group"]
        .apply(lambda x: x.mid if pd.notnull(x) else np.nan)
        .astype(float)
    )
    avg_fpar2_up["temp_mid"] = (
        avg_fpar2_up[param[1] + "group"]
        .apply(lambda x: x.mid if pd.notnull(x) else np.nan)
        .astype(float)
    )

    if figure:
        ax = avg_fpar2_dn.plot(
            x=param[2], y="temp_mid", color="red", label="dn", linestyle="-."
        )
        avg_fpar2_up.plot(
            ax=ax, y="temp_mid", x=param[2], color="blue", label="up", linestyle="-."
        )

    return float(np.mean(np.abs(avg_fpar2_dn[param[2]] - avg_fpar2_up[param[2]])))


def find_opt_alp_tat(
    alpha_r: np.ndarray,
    tau_r: np.ndarray,
    data: pd.DataFrame,
    param: list,
    pi: float,
    pf: float,
    tbin: float,
    figure: bool,
) -> np.ndarray:
    """Grid-search cell_thermal_mass's (alpha, tau) over alpha_r x tau_r,
    via the pandas-based alp_tau scorer.

    INPUT:
      alpha_r  =  amplitude values to search                 [ array ]
      tau_r    =  time-constant values to search              [ array, s ]
      data     =  cast data                                   [ DataFrame ]
      param    =  column names, scorer-specific                [ list[str] ]
      pi       =  lower pressure bound                          [ dbar ]
      pf       =  upper pressure bound                           [ dbar ]
      tbin     =  temperature bin width                          [ degC ]
      figure   =  plot each scored profile                       [ bool ]

    OUTPUT:
      error matrix, shape (len(alpha_r), len(tau_r))              [ array ]
    """
    data_i = data.copy()
    max_value = np.zeros((len(alpha_r), len(tau_r)))

    for r, i in enumerate(alpha_r):
        for c, j in enumerate(tau_r):
            data_i[param[3]] = proc.cell_thermal_mass(
                temperature_C=data_i[param[1]],
                conductivity_Sm=data_i[param[2]],
                amplitude=i,  # alpha
                time_constant=j,  # 1/beta (tau)
                # hardcoded: only correct if the data being scored is actually 24 Hz
                sample_interval=1 / 24,
            )
            max_value[r, c] = alp_tau(
                data_i, [param[0], param[1], param[3]], pi, pf, tbin, figure=False
            )
    return max_value


def bottle_avg(data_ctd,data_bl,scan_offset,scan_duration,samp_int):
    """Average each bottle-stop's CTD channels over a scan window.

    INPUT:
      data_ctd       =  CTD cast data, with a 'scan' column        [ DataFrame ]
      data_bl        =  bottle log (.bl), with 'start_scan' and
                        'position' columns                          [ DataFrame ]
      scan_offset    =  window start, before start_scan               [ s ]
      scan_duration  =  window length                                 [ s ]
      samp_int       =  sample interval                               [ s ]

    OUTPUT:
      one row per bottle, indexed by its real Niskin position          [ DataFrame ]
    """
    scanpsecond=1/samp_int
    data_vec=np.zeros((len(data_bl['start_scan']),8))
    columnas = [
        'bottle', 
        'conductivity', 
        'conductivity2', 
        'temperature', 
        'temperature2', 
        'latitude', 
        'longitude', 
        'pressure'
    ]
    
    bottle_positions = data_bl['position'].values
    for j,k in enumerate(data_bl['start_scan']):
        init_scan=k-scanpsecond*scan_offset
        finl_scan=init_scan+scanpsecond*scan_duration
        cond_blst=data_ctd['conductivity'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        pres_blst=data_ctd['pressure'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        temp_blst=data_ctd['temperature'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        cond_blst2=data_ctd['conductivity2'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        temp_blst2=data_ctd['temperature2'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        lat_blst=data_ctd['latitude'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()
        lon_blst=data_ctd['longitude'][ (data_ctd['scan']<=finl_scan) & (data_ctd['scan']>=init_scan)].mean()

        data_vec[j,:]=[bottle_positions[j],cond_blst,cond_blst2,temp_blst,temp_blst2,lat_blst,lon_blst,pres_blst]
        df_bottles = pd.DataFrame(data_vec, columns=columnas)

    return df_bottles  

def slope_for_correction(a, b):
    """Least-squares gain fitting b ~= m*a, forced through the origin.
    NaN-masks a/b internally.

    INPUT:
      a, b  =  paired values, NaNs allowed              [ array ]

    OUTPUT:
      m = sum(a*b)/sum(a*a), minimizing sum((b - m*a)**2)    [ float ]
    """
    mask_nan = ~np.isnan(a) & ~np.isnan(b)
    aaS = np.sum(a[mask_nan]*a[mask_nan])
    abS = np.sum(a[mask_nan]*b[mask_nan])
    m = abS/aaS
    return m
        
