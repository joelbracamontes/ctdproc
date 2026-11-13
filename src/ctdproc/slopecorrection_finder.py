def slopecorrection_finder(yaml_file):
    """Fit a per-sensor-pair conductivity slope (gain) correction from CTD vs.
    bottle/salinometer comparisons, using iterative 3-sigma outlier rejection.

    Reads a YAML config (sensor pairs, bottle-stop offset/duration, bottle
    salinity CSV), reprocesses each station's raw CTD data through the usual
    align/wild_edit/celltm pipeline, averages it at each bottle stop, and
    compares it against bottle conductivity (from measured salinity) to fit
    one slope per sensor pair. Writes slope_report.csv (summary + detailed
    per-bottle log) and two diagnostic ratio plots per sensor pair.
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
    import matplotlib.pyplot as plt

    import seabirdscientific.conversion as conv
    import seabirdscientific.instrument_data as id_sbs
    import seabirdscientific.processing as proc

    from .load import load_raw_data, convert_data
    from .parsers import CTDCoefficients, load_xmlcon
    from .processing import (
        filt_interp,
        crosshigh,
        find_opt_alp_tat_fast,
        slope_for_correction,
        bottle_avg,
        check_step_lists,
        validate_features,


    )
    from .utilities import extract_head_hex, add_timestamp, bl_reader
    from .pipeline import (
    apply_align,
    apply_low_filter,
    apply_wild_edit,
    apply_celltm,
    )

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    if type(config['constants']['sample_interval'])==str:
        numi,denomi=config['constants']['sample_interval'].split('/')
        sample_interval=float(numi)/float(denomi)
    BAD_FLAG_VALUE = np.float64(-9.99e-29)
    path_dir_data = '../data/'
    a_b_stations_m_values = []
    detailed_points = []
    slope_step = config.get("slope_settings",{})
    check_step_lists(
        "slope_settings",
        scan_offset=slope_step.get("scan_offset", []),
        scan_duration=slope_step.get("scan_duration", []),
        temperature_sensors=slope_step.get("temperature_sensors", []),
        conductivity_sensors=slope_step.get("conductivity_sensors", []),
    )
    validate_features(config.get('features'), ['align', 'low_filter', 'wild_edit', 'celltm'])
    data_salt_name = slope_step.get("bottle_sal_file",[])
    data_salt = pd.read_csv(data_salt_name)
    data_salt['No_bottle'] = data_salt['No_bottle'].astype(int)
    for offset, duration, temp, cond in zip(
            

            slope_step.get("scan_offset",[]),
            slope_step.get("scan_duration",[]),
            slope_step.get("temperature_sensors",[]),
            slope_step.get("conductivity_sensors",[])


            ):
        
        print(data_salt,temp,cond,offset,duration)
        a=np.zeros(data_salt.shape[0]) #Create vector for a and b
            #b=np.zeros(data_salt.shape[0])

        processed_stations = {}
    
        unique_stations = data_salt['Station'].unique()


        for station in unique_stations:


            hex_file = path_dir_data+station+".hex"
            xmlconfile = path_dir_data+station+".XMLCON"
            bl_file = path_dir_data+station+'.bl'

            raw_data = load_raw_data(hex_file)
            data=convert_data(raw_data,xmlconfile,sample_interval,convert_optional=False)
            data['scan']=data.index
            for feat in config.get('features') or []:
                func_name = feat.get('name')
                if func_name == 'align':
                    data, log = apply_align(data,feat,sample_interval,BAD_FLAG_VALUE,)
                elif func_name == 'low_filter':
                    data, log = apply_low_filter(data,feat,sample_interval,)
                elif func_name == 'wild_edit':
                    data, log = apply_wild_edit(data,feat,BAD_FLAG_VALUE,)
                elif func_name == 'celltm':
                    data, log = apply_celltm(data,feat,sample_interval,BAD_FLAG_VALUE,)

            bl_data = bl_reader(bl_file)
            from_bl_data_example=bottle_avg(data,bl_data,offset,duration,sample_interval)
            processed_stations[station] = from_bl_data_example.set_index('bottle')
                
        temp_vector = np.zeros(data_salt.shape[0])
        press_vector = np.zeros(data_salt.shape[0])
   


        for k in range(data_salt.shape[0]):

            station = data_salt['Station'].iloc[k]
            bottle_no = data_salt['No_bottle'].iloc[k]
            station_data = processed_stations[station]
            a[k] = station_data.loc[bottle_no, cond]
            temp_vector[k] = station_data.loc[bottle_no, temp]
            press_vector[k] = station_data.loc[bottle_no, 'pressure']
            

        b = gsw.C_from_SP(data_salt['Sal_avg'].values, temp_vector, press_vector) / 10 # S/m
        stations = data_salt['Station'].values
        bottles = data_salt['No_bottle'].values

        # Iterative outlier rejection on fit residuals (WHOI/ODF cruise-processing
        # convention): fit a slope, drop any bottle whose residual exceeds 3
        # standard deviations, refit, and repeat until nothing more is excluded.
        #
        # `slope_for_correction(a, b)` fits bottle conductivity as a function of
        # CTD conductivity (b ~= m*a), so residuals must be taken in that same
        # direction (b - m*a) - the reverse direction destabilizes the std
        # estimate and can collapse the iteration, excluding far more bottles
        # than are actually bad (observed on some years' sensor pairs, not others,
        # depending on how noisy that pair's comparison points are).
        ratio = a / b
        valid = ~np.isnan(ratio)
        OUTLIER_SIGMA = 3.0
        working = valid.copy()
        for _ in range(20):  # safety cap; converges in ~6-7 passes in practice
            a_iter = np.where(working, a, np.nan)
            m_iter = slope_for_correction(a_iter, b)
            residuals = b - m_iter * a
            std_resid = np.nanstd(residuals[working], ddof=1)
            if std_resid == 0 or np.isnan(std_resid):
                break
            new_outliers = working & (np.abs(residuals) > OUTLIER_SIGMA * std_resid)
            if not new_outliers.any():
                break
            working &= ~new_outliers
        is_outlier = valid & ~working

        a_clean = a.copy()
        a_clean[is_outlier] = np.nan  # slope_for_correction already NaN-masks internally
        m = slope_for_correction(a_clean, b)

        a_b_stations_m_values.append({
            'temperature_sensor': temp,
            'conductivity_sensor': cond,
            'slope_coefficient': m,
            'n_outliers_excluded': int(is_outlier.sum()),
            })

        for st, bt, c_ctd, c_btl, outl in zip(stations, bottles, a, b, is_outlier):
            detailed_points.append({
                'temperature_sensor': temp,
                'conductivity_sensor': cond,
                'Station': st,
                'No_bottle': bt,
                'CTD_conductivity_a': c_ctd,
                'Bottle_conductivity_b': c_btl,
                'outlier': bool(outl),
                })



        clean_labels = pd.Series(stations).str.replace(config['constants']['name']+'_', '')

        # Two figures per sensor pair: one with every point (outliers shown
        # in red) and one with the outliers removed from view entirely - the
        # slope line (m, already computed without outliers) is the same on
        # both, only what's drawn differs.
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(clean_labels[~is_outlier], ratio[~is_outlier], '.', markersize=8, color='blue', label='Bottle Ratios (a/b)')
        if is_outlier.any():
            ax.plot(clean_labels[is_outlier], ratio[is_outlier], 'o', markersize=8, color='red',
                    label=f'Outliers excluded from fit (n={int(is_outlier.sum())})')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=m, color='green', linestyle='-.', linewidth=1.5, label=f'Calculated Slope m ({m:.6f})')
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.set_xlabel('Stations')
        ax.set_ylabel('Ratio CTD / Salinometer')
        ax.set_title(f'Conductivity Ratio Validation & Slope Report (all data)\nSensor Configuration: Temp={temp} | Cond={cond}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"Conductivity_ratio_report_{cond}_all.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(clean_labels[~is_outlier], ratio[~is_outlier], '.', markersize=8, color='blue', label='Bottle Ratios (a/b)')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=m, color='green', linestyle='-.', linewidth=1.5, label=f'Calculated Slope m ({m:.6f})')
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.set_xlabel('Stations')
        ax.set_ylabel('Ratio CTD / Salinometer')
        ax.set_title(f'Conductivity Ratio Validation & Slope Report (outliers removed, n={int(is_outlier.sum())})\nSensor Configuration: Temp={temp} | Cond={cond}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"Conductivity_ratio_report_{cond}_clean.png", dpi=150)
        plt.close()


    

    df_summary = pd.DataFrame(a_b_stations_m_values)
    df_detailed = pd.DataFrame(detailed_points)


    with open('slope_report.csv', 'w', encoding='utf-8') as f:
        f.write("# === CAMPAIGN SUMMARY AVERAGES (COMPUTED PARAMETERS) ===\n")
        df_summary.to_csv(f, index=False)

        f.write("\n# === DETAILED LOG PER STATION / BOTTLE / SENSOR ===\n")
        df_detailed.to_csv(f, index=False)





import sys
def main():
    if len(sys.argv) != 2:

        print("Usage: python -m ctdproc.slopecorrection_finder config_saltcorr.yaml")
        sys.exit(1)
        
    yaml_file = sys.argv[1]
    slopecorrection_finder(yaml_file)
if __name__ == "__main__":
    main()
