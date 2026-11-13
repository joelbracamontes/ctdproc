import logging

import gsw
import numpy as np
import xarray as xr
import seabirdscientific.processing as proc
from .processing import filt_interp, check_step_lists

logger = logging.getLogger(__name__)


def _bin_average(data, **kwargs):
    """Bin-average a DataFrame via seabirdscientific's bin_average.

    INPUT:
      data      =  cast data                                   [ DataFrame ]
      **kwargs  =  passed through to bin_average                 [ see seabirdscientific.processing.bin_average ]

    OUTPUT:
      bin-averaged cast data                                      [ DataFrame ]
    """
    # seabirdscientific 3.1.2's bin_average takes/returns an xr.Dataset (was
    # pd.DataFrame in 2.7.8); the rest of this pipeline is still DataFrame-based.
    binned = proc.bin_average(dataset=xr.Dataset.from_dataframe(data), **kwargs)
    return binned.to_dataframe().reset_index(drop=True)



def apply_align(data, step, dt, bad_flag_value):
    """Time-shift each listed variable by its offset, to correct sensor lag.

    INPUT:
      data            =  cast data                            [ DataFrame ]
      step            =  config dict with 'variables'/'offset' lists    [ dict ]
      dt              =  sample interval                        [ s ]
      bad_flag_value  =  sentinel for flagged scans               [ float ]

    OUTPUT:
      data  =  updated cast data                                 [ DataFrame ]
      text  =  .cnv-style processing log lines                      [ list[str] ]
    """

    text = []

    variables = step.get("variables", [])
    offsets = step.get("offset", [])
    check_step_lists("align", variables=variables, offset=offsets)

    for var, offset in zip(variables, offsets):

        data[var] = proc.align_ctd(
            x=data[var],
            offset=offset,
            sample_interval=dt,
            flag_value=bad_flag_value,
        )

        text.append(f"* alignctd {var} {offset} s\n")

    text.append("\n")

    return data, text


def apply_low_filter(data, step, dt):
    """Low-pass filter each listed variable, to smooth high-frequency noise.

    INPUT:
      data  =  cast data                                      [ DataFrame ]
      step  =  config dict with 'variables'/'time_constant' lists    [ dict ]
      dt    =  sample interval                                  [ s ]

    OUTPUT:
      data  =  updated cast data                                [ DataFrame ]
      text  =  .cnv-style processing log lines                     [ list[str] ]
    """

    text = []

    variables = step.get("variables", [])
    time_constants = step.get("time_constant", [])
    check_step_lists("low_filter", variables=variables, time_constant=time_constants)

    for var, time_constant in zip(variables, time_constants):

        data[var] = filt_interp(
            data,
            var,
            time_constant,
            dt,
            False
        )

        text.append(f"* filter_low_pass {var} {time_constant} s\n")

    text.append("\n")

    return data, text


def apply_wild_edit(data, step, bad_flag_value):
    """Flag outlier scans in each listed variable via a two-pass
    standard-deviation test.

    INPUT:
      data            =  cast data                           [ DataFrame ]
      step            =  config dict with 'variables'/'std_pass_1'/
                         'std_pass_2'/'scans_per_block'/
                         'distance_to_mean' lists               [ dict ]
      bad_flag_value  =  sentinel for flagged scans              [ float ]

    OUTPUT:
      data  =  updated cast data                                 [ DataFrame ]
      text  =  .cnv-style processing log lines                      [ list[str] ]
    """

    text = []

    variables = step.get("variables", [])
    std_pass_1 = step.get("std_pass_1", [])
    std_pass_2 = step.get("std_pass_2", [])
    scans_per_block = step.get("scans_per_block", [])
    distance_to_mean = step.get("distance_to_mean", [])
    check_step_lists(
        "wild_edit",
        variables=variables,
        std_pass_1=std_pass_1,
        std_pass_2=std_pass_2,
        scans_per_block=scans_per_block,
        distance_to_mean=distance_to_mean,
    )

    for var, std1, std2, scans, distance in zip(
        variables, std_pass_1, std_pass_2, scans_per_block, distance_to_mean
    ):

        data[var] = proc.wild_edit(
            data=data[var].values,
            flags=data["flag"].values,
            std_pass_1=std1,
            std_pass_2=std2,
            scans_per_block=scans,
            distance_to_mean=distance,
            exclude_bad_flags=True,
            flag_value=bad_flag_value,
        )

        text.append(f"* wildedit {var}\n")
        text.append(f"* wildedit pass1 nstd {std1}\n")
        text.append(f"* wildedit pass2 nstd {std2}\n")
        text.append(f"* wildedit midelta {distance}\n")
        text.append(f"* wildedit npoint {scans}\n")

    text.append("\n")

    return data, text
    
    


def apply_celltm(data, step, sample_interval, bad_flag_value):
    """Apply the cell thermal mass correction to each listed conductivity
    channel, and recompute the matching salinity from corrected conductivity.

    INPUT:
      data             =  cast data                           [ DataFrame ]
      step             =  config dict with 'variables'/'amplitude'/
                          'time_constant' lists                  [ dict ]
      sample_interval  =  time between scans                     [ s ]
      bad_flag_value   =  sentinel for flagged scans                [ float ]

    OUTPUT:
      data  =  updated cast data                                 [ DataFrame ]
      text  =  .cnv-style processing log lines                      [ list[str] ]
    """

    text = []

    variables = step.get("variables", [])
    amplitudes = step.get("amplitude", [])
    time_constants = step.get("time_constant", [])
    check_step_lists(
        "celltm", variables=variables, amplitude=amplitudes, time_constant=time_constants
    )

    for var, amplitude, time_constant in zip(variables, amplitudes, time_constants):

        if var == "conductivity":
            temp_var = "temperature"
            sal_var = "salinity"
        else:
            temp_var = "temperature2"
            sal_var = "salinity2"

        # cell_thermal_mass differentiates temperature (dT/dt) through a
        # recursive filter - a flagged scan left at bad_flag_value would look
        # like a genuine multi-degree jump and corrupt a trailing stretch of
        # corrected conductivity, not just that one scan. Mask flagged scans
        # and interpolate before calling it, then restore the flag afterward -
        # same pattern find_opt_alp_tat_fast (processing.py) already uses when
        # deriving these alpha/tau values in the first place. Keeping the flag
        # (rather than silently healing it) matters downstream too: bin
        # averaging excludes flagged scans from its means.
        bad_ctm = (data[temp_var].values == bad_flag_value) | (data[var].values == bad_flag_value)

        temp_clean = data[temp_var].copy()
        temp_clean[bad_ctm] = np.nan
        temp_clean = temp_clean.interpolate(limit_direction="both")

        cond_clean = data[var].copy()
        cond_clean[bad_ctm] = np.nan
        cond_clean = cond_clean.interpolate(limit_direction="both")

        corrected = proc.cell_thermal_mass(
            temperature_C=temp_clean.values,
            conductivity_Sm=cond_clean.values,
            amplitude=amplitude,
            time_constant=time_constant,
            sample_interval=sample_interval,
        )
        corrected[bad_ctm] = bad_flag_value
        data[var] = corrected

        # salinity also depends on pressure - flag it there too, even though
        # pressure isn't an input to cell_thermal_mass itself
        bad_sal = bad_ctm | (data["pressure"].values == bad_flag_value)
        sal = gsw.SP_from_C(
            10 * corrected,
            t=temp_clean.values,
            p=data["pressure"].values
        )
        sal[bad_sal] = bad_flag_value
        data[sal_var] = sal

        text.append(f"* celltm {var}\n")
        text.append(f"* celltm A {amplitude}\n")
        text.append(f"* celltm B {time_constant}\n")

    text.append("\n")


    return data, text

def apply_slope_correction(data, step, bad_flag_value=np.float64(-9.99e-29)):
    """Scale each listed conductivity channel by its fitted slope (gain),
    and recompute the matching salinity from corrected conductivity.

    INPUT:
      data            =  cast data                            [ DataFrame ]
      step            =  config dict with 'variables'/'slope' lists    [ dict ]
      bad_flag_value  =  sentinel for flagged scans              [ float ]

    OUTPUT:
      data  =  updated cast data                                 [ DataFrame ]
      text  =  .cnv-style processing log lines                      [ list[str] ]
    """
    text = []
    data_copy=data.copy()
    variables = step.get('variables', [])
    slopes = step.get('slope', [])
    check_step_lists("slope_corr", variables=variables, slope=slopes)

    for var, m in zip(variables, slopes):

        if var == "conductivity":
            temp_var = "temperature"
            sal_var = "salinity"
        else:
            temp_var = "temperature2"
            sal_var = "salinity2"
        data[var]=m*data_copy[var]

        # a flagged conductivity scaled by m is no longer exactly
        # bad_flag_value (floating point), so check the pre-scale value
        bad_cond = data_copy[var].values == bad_flag_value
        bad_sal = (
            bad_cond
            | (data_copy[temp_var].values == bad_flag_value)
            | (data_copy["pressure"].values == bad_flag_value)
        )
        sal = gsw.SP_from_C(
            10 * data[var].values,
            t=data[temp_var].values,
            p=data["pressure"].values
        )
        sal[bad_sal] = bad_flag_value
        data[sal_var] = sal

        # Restore the clean sentinel in the scaled variable itself too - `m * bad_flag_value`
        # is a corrupted, non-canonical value (not bit-identical to bad_flag_value unless
        # m==1), so anything reading this column directly, not just the derived salinity,
        # sees the real flag rather than an arithmetic artifact of it. Uses conductivity's
        # own bad positions (bad_cond), not the broader bad_sal (which also covers temp/
        # pressure issues unrelated to conductivity's own validity).
        data.loc[bad_cond, var] = bad_flag_value

        text.append(f"* slope_correction {var}\n")
        text.append(f"* slope_correction m =  {m}\n")

    text.append("\n")
    return data, text


def apply_loop_edit(data, step, sample_interval, bad_flag_value):
    """Flag scans affected by ship-roll-induced pressure reversals/slowdowns.
    Not suitable for calibration-dip casts (deliberate stationary bottle-stops).

    INPUT:
      data             =  cast data                          [ DataFrame ]
      step             =  config dict with loop-edit parameters    [ dict ]
      sample_interval  =  time between scans                   [ s ]
      bad_flag_value   =  sentinel for flagged scans             [ float ]

    OUTPUT:
      data  =  updated cast data, with 'flag' column set         [ DataFrame ]
      text  =  .cnv-style processing log lines                     [ list[str] ]
    """

    text = []

    def _scalar(val):
        return val[0] if isinstance(val, list) else val

    min_velocity = _scalar(step.get("min_velocity"))
    window_size = _scalar(step.get("window_size"))
    mean_speed_percent = _scalar(step.get("mean_speed_percent"))
    remove_surface_soak = _scalar(step.get("remove_surface_soak"))
    min_soak_depth = _scalar(step.get("min_soak_depth"))
    max_soak_depth = _scalar(step.get("max_soak_depth"))
    use_deck_pressure_offset = _scalar(step.get("use_deck_pressure_offset"))

    data["flag"] = proc.loop_edit_pressure(
        pressure=data["pressure"].values,
        latitude=float(np.nanmedian(data["latitude"].values)),
        # .copy(): loop_edit_depth writes into this array in-place (flag[:] = 0.0)
        # when exclude_flags=False - under pandas's Copy-on-Write, a view via
        # .values straight off the DataFrame is read-only, so it needs its own array.
        flag=data["flag"].values.copy(),
        sample_interval=sample_interval,
        min_velocity_type=proc.MinVelocityType.FIXED,
        min_velocity=min_velocity,
        window_size=window_size,
        mean_speed_percent=mean_speed_percent,
        remove_surface_soak=remove_surface_soak,
        min_soak_depth=min_soak_depth,
        max_soak_depth=max_soak_depth,
        use_deck_pressure_offset=use_deck_pressure_offset,
        exclude_flags=False,
        flag_value=bad_flag_value,
    )

    text.append("* loopedit\n")
    text.append(f"* loopedit_minvel {min_velocity}\n")
    text.append(f"* loopedit: min_depth {min_soak_depth}\n")
    text.append(f"* loopedit: max_depth {max_soak_depth}\n")
    text.append("\n")

    return data, text


def apply_bin(data, step, bad_flag_value, base_name):
    """Bin-average the cast for each bin_type/bin_size/type_profile triplet,
    split by cast direction.

    INPUT:
      data            =  cast data                             [ DataFrame ]
      step            =  config dict with 'bin_type'/'bin_size'/
                         'type_profile' lists                     [ dict ]
      bad_flag_value  =  sentinel for flagged scans                [ float ]
      base_name       =  cast base filename, for output names       [ str ]

    OUTPUT:
      bin_results   =  one DataFrame per triplet                    [ list[DataFrame] ]
      bin_logs      =  one .cnv-style log per triplet                 [ list[list[str]] ]
      output_names  =  one output filename per triplet                 [ list[str] ]
    """

    bin_results = []
    bin_logs = []
    output_names = []

    bin_types = step.get("bin_type", [])
    bin_sizes = step.get("bin_size", [])
    profiles = step.get("type_profile", [])
    check_step_lists("bin", bin_type=bin_types, bin_size=bin_sizes, type_profile=profiles)

    data["elapsed_time"] = (
        data["elapsed_time"] - data["elapsed_time"].iloc[0]
    )

    for bin_type, bin_size, profile in zip(
        bin_types,
        bin_sizes,
        profiles
    ):

        bin_text = []

        if profile == "BOTH":
            cast_type = proc.CastType.BOTH
            cast_name = ""

        elif profile == "UP":
            cast_type = proc.CastType.UPCAST
            cast_name = "up"

        elif profile == "DOWN":
            cast_type = proc.CastType.DOWNCAST
            cast_name = "down"

        else:
            raise ValueError(
                f"config error in 'bin' step: unknown type_profile '{profile}'. "
                f"Valid options are:\n"
                f"  - BOTH: bin the full cast, up- and downcast together\n"
                f"  - UP: bin only the upcast\n"
                f"  - DOWN: bin only the downcast"
            )


        if bin_type == "elapsed_time":

            bin_data = _bin_average(
                data,
                bin_variable="elapsed_time",
                bin_size=bin_size,
                cast_type=cast_type,
                flag_value=bad_flag_value,
                interpolate=True,
            )

            bin_label = f"{bin_size:g}sec"

            bin_text.append(
                f"* binsize elapsed_time {bin_size}s\n"
            )


        elif bin_type == "scans":

            bin_data = _bin_average(
                data,
                bin_variable="nScan",
                bin_size=bin_size,
                cast_type=cast_type,
                flag_value=bad_flag_value,
                interpolate=True,
            )

            bin_label = f"{bin_size:g}scans"

            bin_text.append(
                f"* binsize nScan {bin_size}\n"
            )


        elif bin_type == "pressure":

            bin_data = _bin_average(
                data,
                bin_variable="pressure",
                bin_size=bin_size,
                cast_type=cast_type,
                flag_value=bad_flag_value,
                interpolate=True,
            )

            bin_label = f"{bin_size:g}dbar"

            bin_text.append(
                f"* binsize pressure {bin_size}dbar\n"
            )


        else:
            raise ValueError(
                f"config error in 'bin' step: unknown bin_type '{bin_type}'. "
                f"Valid options are:\n"
                f"  - elapsed_time: bin by seconds elapsed since cast start\n"
                f"  - scans: bin by raw scan count\n"
                f"  - pressure: bin by pressure (dbar)"
            )


        # remove flag for output
        if "flag" in bin_data.columns:
            bin_data = bin_data.drop(columns=["flag"])


        # filename convention:
        # upMIXSED2_038_1dbar.cnv
        # downMIXSED2_038_1dbar.cnv
        # MIXSED2_038_24scans.cnv

        if cast_name:
            filename = f"{cast_name}{base_name}_{bin_label}.cnv"
        else:
            filename = f"{base_name}_{bin_label}.cnv"


        output_names.append(filename)
        bin_results.append(bin_data)
        bin_logs.append(bin_text)


    return bin_results, bin_logs, output_names
