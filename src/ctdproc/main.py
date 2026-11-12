def CTD_process(yaml_file):
    """Run the full per-cast CTD processing pipeline from a YAML config.

    For each cast index listed in the config, loads the raw .hex/.xmlcon pair,
    converts it to physical units, applies each configured processing step
    (align/wild_edit/celltm/slope_corr/loop_edit/bin) in order, and writes an
    SBE-style .cnv file (and/or per-bin files) with an ASCII header rebuilt
    from the original hex file's own header lines.
    """
    import yaml
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="seabirdscientific"
    )

    import datetime as datetime
    import numpy as np
    import pandas as pd
    import gsw

    import seabirdscientific.conversion as conv
    import seabirdscientific.instrument_data as id_sbs
    import seabirdscientific.processing as proc

    from .load import load_raw_data, convert_data
    from .parsers import CTDCoefficients, load_xmlcon
    from .processing import (
        filt_interp,
        crosshigh,
        alp_tau,
        find_opt_alp_tat,
    )
    from .utilities import extract_head_hex, add_timestamp
    from .pipeline import (
    apply_align,
    apply_low_filter,
    apply_wild_edit,
    apply_celltm,
    apply_loop_edit,
    apply_bin,
    apply_slope_correction,
    )
    from pathlib import Path

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    if type(config['constants']['sample_interval'])==str:
        numi,denomi=config['constants']['sample_interval'].split('/')
        sample_interval=float(numi)/float(denomi)

    data_dir = Path(config['constants'].get('data_dir', '../data'))
    output_dir = Path(config['constants'].get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)


    BAD_FLAG_VALUE = np.float64(-9.99e-29)

    decimales = {
        
        'pressure': 3,
        'temperature': 4,
        'conductivity': 6,
        'temperature2': 4,
        'conductivity2': 6,
        'latitude': 5,
        'longitude': 5,
        'salinity': 4,
        'salinity2': 4,
        'elapsed_time': 3,
        'oxygen': 3,
        'chlorophyll': 4,
        'turbidity': 4}
    
    if config['constants'].get('sind') and config['constants'].get('ind'):
        raise ValueError(
            "config has both 'ind' and 'sind' set - use only one"
        )

    if 'convert' not in config['constants']:
        raise ValueError(
            "config is missing 'constants.convert' (e.g. convert: 'yes') - "
            "controls whether a .cnv file is written for each cast"
        )

    if config['constants'].get('sind'):
        j_list = config['constants']['sind']
    else:
        j_list = list(range(config['constants']['ind'][0],config['constants']['ind'][1]+1))

    for j in j_list:

    

        
        base = config['constants']['name'] + '_' + str(j).zfill(3)
        raw_data = load_raw_data(data_dir / (base + '.hex'))
        data = convert_data(raw_data, data_dir / (base + '.xmlcon'), sample_interval, convert_optional=False)
        head=[]
        head_coeff=[]
        head=extract_head_hex(data_dir / (base + '.hex'))
        head_coeff=extract_head_hex(data_dir / (base + '.xmlcon'))
        text_content =[]
        text_content.extend(head)
        text_content.append("* \n")
        text_content.extend(["* " + line for line in head_coeff[19:]])
        text_content.append("* \n")
        text_process=[]

        for step in config.get('features') or []:
            func_name = step.get('name')
            print(func_name)

            if func_name == 'align':
                
                data, log = apply_align(data,step,sample_interval,BAD_FLAG_VALUE,)
                text_process.extend(log)

            elif func_name == 'low_filter':
                data, log = apply_low_filter(data,step,sample_interval,)
                text_process.extend(log)

            elif func_name == 'wild_edit':
                
                data, log = apply_wild_edit(data,step,BAD_FLAG_VALUE,)
                text_process.extend(log)
      
            elif func_name == 'celltm':
                data, log = apply_celltm(data,step,sample_interval,BAD_FLAG_VALUE,)
                text_process.extend(log)

            elif func_name == 'slope_corr':
                data, log = apply_slope_correction(data,step,BAD_FLAG_VALUE,)
                text_process.extend(log)
    
            elif func_name == 'loop_edit':
                data, log = apply_loop_edit(data,step,sample_interval,BAD_FLAG_VALUE,)
                text_process.extend(log)

            elif func_name == 'bin':
                bin_results, bin_logs, filenames = apply_bin(data,step, BAD_FLAG_VALUE, base,)
                filenames = [str(output_dir / f) for f in filenames]

                for bin_data, bin_log, filename in zip(
                    bin_results,
                    bin_logs,
                    filenames
                    ):
                    file_text = []
                    # all processing before binning
                    file_text.extend(text_content)
                    file_text.extend(text_process)
                    # only this bin's own information
                    file_text.extend(bin_log)
                
                    with open(filename, "w", encoding="utf-8") as f:
                        f.writelines(file_text)
                
                    bin_data.round(decimales).to_csv(
                        filename,
                        mode="a",
                        index=False
                    )


        if config['constants']['convert'] == 'yes':
            nominal_name = str(output_dir / (base + "_24hz.cnv"))
        
            ref_time = config.get('constants', {}).get('time_ref')

            if ref_time is not None:

                scans = np.arange(data.shape[0])

                data['timeJ'] = add_timestamp(ref_time,head,scans)


            data['elapsed_time'] = (data['elapsed_time'] - data['elapsed_time'].iloc[0])
        
            data = data.drop(columns=['flag'])

            # P/T/C/salinity only for now - caldip only needs these. Other sensors
            # (oxygen, chlorophyll, turbidity, PAR, ...) deliberately excluded here;
            # they'd need per-sensor-model SBE names from the XMLCON (the separate,
            # not-yet-done sensor-typing work) rather than one fixed name each like
            # T/C/P/S have - come back to this once that's built.
            sbe_names = [
                ('pressure', 'prDM', 'Pressure, Digiquartz', 'db'),
                ('temperature', 't090C', 'Temperature', 'ITS-90, deg C'),
                ('temperature2', 't190C', 'Temperature, 2', 'ITS-90, deg C'),
                ('conductivity', 'c0S/m', 'Conductivity', 'S/m'),
                ('conductivity2', 'c1S/m', 'Conductivity, 2', 'S/m'),
                # salinity/salinity2: recomputed from the final corrected conductivity
                # by apply_slope_correction (pipeline.py) - only true because slope_corr
                # is the last conductivity-touching step in every real config; if that
                # ever changes, this would need recomputing here instead.
                ('salinity', 'sal00', 'Salinity, Practical', 'PSU'),
                ('salinity2', 'sal11', 'Salinity, Practical, 2', 'PSU'),
            ]
            sbe_cols = [c for c in sbe_names if c[0] in data.columns]

            start_time = None
            if ref_time is not None:
                for line in head:
                    if ref_time in line:
                        start_time = line.split('=')[1].strip()
                        break

            sbe_header = []
            sbe_header.append(f"# nvalues = {data.shape[0]}\n")
            for idx, (col, label, desc, units) in enumerate(sbe_cols):
                sbe_header.append(f"# name {idx} = {label}: {desc} [{units}]\n")
            sbe_header.append(f"# interval = seconds: {round(sample_interval, 6)}\n")
            sbe_header.append(f"# start_time = {start_time}\n")
            if 'latitude' in data.columns and pd.notna(data['latitude'].iloc[0]):
                lat = float(data['latitude'].iloc[0])
                lat_deg = int(lat)
                lat_min = abs(lat - lat_deg) * 60.0
                sbe_header.append(f"** Latitude: {lat_deg} {lat_min:.4f}\n")
            sbe_header.append("*END*\n")

            with open(nominal_name, "w", encoding="utf-8") as f:
                f.writelines(sbe_header)

            data[[c[0] for c in sbe_cols]].round(decimales).to_csv(
                nominal_name,
                sep=' ',
                header=False,
                index=False,
                mode="a",
                na_rep="nan",
            )

        print('processed_'+str(j).zfill(3))

