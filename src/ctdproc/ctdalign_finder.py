def ctdalign_finder(yaml_file):
    """Measure the time lag between a lead/lag variable pair (e.g. pressure
    vs. conductivity) across a range of casts, using crosshigh's cross-
    correlation, and write per-cast + campaign-average lag offsets ready to
    paste into an `align` step's `offset:` list.
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

    import seabirdscientific.conversion as conv
    import seabirdscientific.instrument_data as id_sbs
    import seabirdscientific.processing as proc

    from .load import load_raw_data, convert_data
    from .parsers import CTDCoefficients, load_xmlcon
    from .processing import (
        filt_interp,
        crosshigh,
    )
    from .utilities import extract_head_hex, add_timestamp


    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    if type(config['constants']['sample_interval'])==str:
        numi,denomi=config['constants']['sample_interval'].split('/')
        sample_interval=float(numi)/float(denomi)
        
    offset_alignment_values = []
    step = config.get("alignment_settings",{})
    for j in list(range(config['constants']['ind'][0],config['constants']['ind'][1]+1)):

        raw_data = load_raw_data("../data/"+config['constants']['name']+'_'+str(j).zfill(3)+'.hex')
        data = convert_data(raw_data,"../data/"+config['constants']['name']+'_'+str(j).zfill(3)+'.xmlcon',sample_interval,convert_optional=False)

        for var_lead, var_lag, max_lag, pre_up,pre_lo in zip(

                step.get("variable_lead", []),
                step.get("variable_lag", []),
                step.get("max_lag",[]),
                step.get("pressure_upper",[]),
                step.get("pressure_lower",[])
                ):
            print(var_lead,var_lag,max_lag,pre_up,pre_lo)

            func_name = step.get('name')
            _,lag_up,_,lag_dn=crosshigh(data,['pressure',var_lead,var_lag],
                                      max_lag,pre_up,pre_lo,sample_interval,True)

            # crosshigh's raw sign is backwards for align_ctd's offset convention:
            # crosshigh returns negative when var_lag is physically behind var_lead
            # (the real conductivity-cell-lag case), but align_ctd's offset needs a
            # positive value to pull a lagging series forward into alignment. Negate
            # here so alignment_report.csv's values can be pasted directly into a
            # config's `offset:` list.
            lg_dn = -lag_dn
            lg_up = -lag_up

            offset_alignment_values.append({
             'cast_no': j,
             'variable_lead': var_lead,
             'variable_lag': var_lag,
             'lag_down_cast': lg_dn,
             'lag_up_cast': lg_up
             })


    df_lags = pd.DataFrame(offset_alignment_values)



    df_means = df_lags.groupby(['variable_lead', 'variable_lag'])[['lag_down_cast', 'lag_up_cast']].mean().reset_index()
    df_means['cast_no'] = 'AVERAGE'

    df_means = df_means[['cast_no', 'variable_lead', 'variable_lag', 'lag_down_cast', 'lag_up_cast']]


    with open('alignment_report.csv', 'w', encoding='utf-8') as f:
        f.write("# === DETAILED LOG PER CAST ===\n")
        df_lags.to_csv(f, index=False)
    
        f.write("\n# === CAMPAIGN SUMMARY AVERAGES ===\n")
        df_means.to_csv(f, index=False)

import sys
def main():
    if len(sys.argv) != 2:

        print("Usage: python -m ctdproc.ctdalign_finder config_align.yaml")
        sys.exit(1)
        
    yaml_file = sys.argv[1]
    ctdalign_finder(yaml_file)
if __name__ == "__main__":
    main()
