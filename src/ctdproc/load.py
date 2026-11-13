def _find_companion_xmlcon(hex_file):
    """Find the .XMLCON file next to a .hex file.

    INPUT:
      hex_file  =  path to a raw .hex file           [ str or Path ]

    OUTPUT:
      path to the matching .XMLCON/.xmlcon file       [ Path ]
    """
    from pathlib import Path

    hex_path = Path(hex_file)
    for suffix in (".XMLCON", ".xmlcon"):
        candidate = hex_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No companion .XMLCON/.xmlcon found next to {hex_file} "
        f"(expected {hex_path.with_suffix('.XMLCON')})"
    )


def _strip_truncated_trailing_scan(hex_file):
    """Remove a truncated, NUL-padded final scan line from a raw .hex file.

    Writes a cleaned copy to a temp file and returns its path if the last
    data line looks corrupted; otherwise returns hex_file unchanged. Never
    modifies the original file.

    INPUT:
      hex_file  =  path to a raw .hex file                 [ str or Path ]

    OUTPUT:
      path to a cleaned temp file, or hex_file unchanged     [ str ]
    """
    import tempfile
    import os

    with open(hex_file, "rb") as f:
        raw = f.read()
    lines = raw.splitlines(keepends=True)
    data_line_lengths = [len(l) for l in lines if not l.startswith(b"*") and l.strip()]
    if len(data_line_lengths) < 2:
        return hex_file
    typical_len = data_line_lengths[0]
    last_data_idx = max(i for i, l in enumerate(lines) if not l.startswith(b"*") and l.strip())
    last_line = lines[last_data_idx]
    is_corrupted = b"\x00" in last_line or len(last_line.rstrip(b"\r\n")) < typical_len - 5
    if not is_corrupted:
        return hex_file

    cleaned = b"".join(lines[:last_data_idx] + lines[last_data_idx+1:])
    tmp = tempfile.NamedTemporaryFile(suffix=".hex", delete=False)
    tmp.write(cleaned)
    tmp.close()
    return tmp.name


def load_raw_data(hex_file, xmlconfile=None):
    """Decode a raw SBE .hex file into a DataFrame of raw channel counts.

    Sensor configuration (channels present, order, frequency/voltage word
    counts) is read from the companion .XMLCON.

    INPUT:
      hex_file    =  path to a raw .hex file                       [ str or Path ]
      xmlconfile  =  path to the matching .XMLCON (optional;
                     found next to hex_file if not given)           [ str or Path ]

    OUTPUT:
      raw channel counts, one row per scan                          [ DataFrame ]
    """
    import seabirdscientific.instrument_data as id_sbs
    from .parsers import detect_sensor_config

    if xmlconfile is None:
        xmlconfile = _find_companion_xmlcon(hex_file)

    enabled_sensors, freq_supp, volt_supp = detect_sensor_config(xmlconfile)

    clean_hex_file = _strip_truncated_trailing_scan(hex_file)
    try:
        raw_dataset = id_sbs.read_hex_file(
             filepath=clean_hex_file,
             instrument_type=id_sbs.InstrumentType.SBE911Plus,
             enabled_sensors=enabled_sensors,
             frequency_channels_suppressed=freq_supp,
             voltage_words_suppressed=volt_supp,
          )
    finally:
        if clean_hex_file != hex_file:
            import os
            os.unlink(clean_hex_file)

    # seabirdscientific 3.1.2's read_hex_file returns an xr.Dataset (was
    # pd.DataFrame in 2.7.8); everything downstream in ctdproc still expects
    # a DataFrame, so convert at this one boundary.
    return raw_dataset.to_dataframe()




def convert_data(raw_data,xmlconfile,sample_interval,convert_optional=True):
    """Convert raw channel counts to physical units.

    INPUT:
      raw_data          =  raw channel counts from load_raw_data      [ DataFrame ]
      xmlconfile        =  path to the matching .XMLCON               [ str or Path ]
      sample_interval   =  time between scans                        [ s ]
      convert_optional  =  also convert chlorophyll/turbidity/oxygen
                           (skipped, filled with NaN, if False)       [ bool ]

    OUTPUT:
      pressure, temperature, conductivity, salinity, and optional
      sensors, one row per scan                                       [ DataFrame ]
    """
    from .parsers import CTDCoefficients, load_xmlcon, detect_voltage_channel_map
    import seabirdscientific.conversion as conv
    import seabirdscientific.instrument_data as id_sbs
    import numpy as np
    import pandas as pd
    import gsw
    coeffs = load_xmlcon(xmlconfile)
    # Which 'volt N' column each auxiliary sensor is actually wired to, read
    # from the XMLCON's own SensorArray order - not assumed fixed per cruise.
    channels = detect_voltage_channel_map(xmlconfile)
    temperature = conv.convert_temperature_frequency(
        frequency=raw_data["temperature"],
        coefs=coeffs.temperature_primary,
        standard="ITS90",
        units="C",
         )
    pressure = conv.convert_pressure_digiquartz(
        pressure_count=raw_data["digiquartz pressure"],
        compensation_voltage=raw_data["temperature compensation"],
        coefs=coeffs.pressure,
        units="dbar",
        sample_interval=sample_interval,
        )


    temperature2 = conv.convert_temperature_frequency(
        frequency=raw_data["secondary temperature"],
        coefs=coeffs.temperature_secondary,
        standard="ITS90",
        units="C",
     )

    # seabirdscientific 3.1.2 removed the manual `scalar` multiplier in favor
    # of instrument_type (SBE911Plus applies the same 1/10 scalar internally)
    # plus an explicit output-units param; "S/m" matches the old scalar=1/10
    # behavior exactly, so the `conductivity*10` for gsw below is unchanged.
    conductivity = conv.convert_conductivity(
        conductivity_count=raw_data["conductivity"],
        temperature=temperature,
        pressure=pressure,
        coefs=coeffs.conductivity_primary,
        instrument_type=id_sbs.InstrumentType.SBE911Plus,
        units="S/m",
       )
    conductivity2 = conv.convert_conductivity(
        conductivity_count=raw_data["secondary conductivity"],
        temperature=temperature2,
        pressure=pressure,
        coefs=coeffs.conductivity_secondary,
        instrument_type=id_sbs.InstrumentType.SBE911Plus,
        units="S/m",
       )

    # Optional auxiliary sensors - not every cruise/instrument has these
    # connected (e.g. the 2012/2017/2018 mooring-calibration casts have none
    # of them). Which 'volt N' column belongs to which sensor is read from
    # `channels` (the XMLCON's own SensorArray order), not assumed fixed -
    # a different cruise can wire oxygen to volt 0 and it still works.
    salinity=gsw.SP_from_C(
        C=conductivity*10,
        t=temperature,
        p=pressure
    )

    if not convert_optional:
        chlorophyll = np.full(len(temperature), np.nan)
        turbidity = np.full(len(temperature), np.nan)
        oxygen = np.full(len(temperature), np.nan)
    else:
        if coeffs.chlorophyll is not None and "chlorophyll" in channels:
            chlorophyll = conv.convert_eco(raw=raw_data[channels["chlorophyll"]], coefs=coeffs.chlorophyll)
        else:
            chlorophyll = np.full(len(temperature), np.nan)

        if coeffs.turbidity is not None and "turbidity" in channels:
            turbidity = conv.convert_eco(raw=raw_data[channels["turbidity"]], coefs=coeffs.turbidity)
        else:
            turbidity = np.full(len(temperature), np.nan)

        if coeffs.oxygen_primary is not None and "oxygen_primary" in channels:
            oxygen = conv.convert_sbe43_oxygen(
                voltage=raw_data[channels["oxygen_primary"]],
                temperature=temperature,
                pressure=pressure,
                salinity=salinity,
                coefs=coeffs.oxygen_primary,
                apply_tau_correction=True,
                apply_hysteresis_correction=True,
                window_size=1,
                sample_interval=sample_interval
            )
        else:
            oxygen = np.full(len(temperature), np.nan)

    flag = np.zeros(len(temperature))
    elapsed_time = (np.cumsum((np.ones(len(temperature))*sample_interval))-sample_interval).astype(float)
    data = pd.DataFrame(
         {
             "pressure": pressure,
             "temperature": temperature,
             "conductivity": conductivity,
             "temperature2": temperature2,
             "conductivity2": conductivity2,
             "latitude": raw_data["NMEA Latitude"],
             "longitude": raw_data["NMEA Longitude"],
             "elapsed_time": elapsed_time,
             "oxygen": oxygen,
             "chlorophyll": chlorophyll,
             "turbidity": turbidity,
             "flag": flag,
               }
     )

    return data
