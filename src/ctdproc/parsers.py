# src/parsers.py

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

# 'modified seabirdscientific'
from seabirdscientific.cal_coefficients import (
    AltimeterCoefficients,
    ConductivityCoefficients,
    ECOCoefficients,
    Oxygen43Coefficients,
    PressureDigiquartzCoefficients,
    TemperatureFrequencyCoefficients,
)


@dataclass
class CTDCoefficients:
    temperature_primary: Optional[TemperatureFrequencyCoefficients] = None
    temperature_secondary: Optional[TemperatureFrequencyCoefficients] = None
    conductivity_primary: Optional[ConductivityCoefficients] = None
    conductivity_secondary: Optional[ConductivityCoefficients] = None
    pressure: Optional[PressureDigiquartzCoefficients] = None
    oxygen_primary: Optional[Oxygen43Coefficients] = None
    oxygen_secondary: Optional[Oxygen43Coefficients] = None
    chlorophyll: Optional[ECOCoefficients] = None
    turbidity: Optional[ECOCoefficients] = None  # It is missing
    altimeter: Optional[AltimeterCoefficients] = None


def detect_sensor_config(xml_file: str):
    """Parse an SBE .XMLCON and derive everything read_hex_file needs to
    decode the matching .hex file.

    INPUT:
      xml_file  =  path to a .XMLCON file                       [ str ]

    OUTPUT:
      enabled_sensors                =  sensors, in wire order    [ list ]
      frequency_channels_suppressed  =  from <Instrument>           [ int ]
      voltage_words_suppressed       =  from <Instrument>            [ int ]
    """
    import seabirdscientific.instrument_data as id_sbs

    root = ET.parse(xml_file).getroot()
    instrument = root.find(".//Instrument")

    freq_supp = int(instrument.findtext("FrequencyChannelsSuppressed", "0"))
    volt_supp = int(instrument.findtext("VoltageWordsSuppressed", "0"))

    sensor_array = instrument.find(".//SensorArray")
    sensors_by_index = {
        int(s.get("index")): list(s)[0].tag  # the sensor's own child tag IS its type
        for s in sensor_array.findall("Sensor")
    }
    n_entries = len(sensors_by_index)

    # Frequency block is always T, C, P (indices 0,1,2). A secondary T/C pair
    # occupies indices 3,4 - only if BOTH are present in that exact order, the
    # standard SBE911/917+ convention.
    has_secondary_tc = (
        sensors_by_index.get(3) == "TemperatureSensor"
        and sensors_by_index.get(4) == "ConductivitySensor"
    )
    n_freq = 5 if has_secondary_tc else 3

    enabled_sensors = [id_sbs.Sensors.Temperature, id_sbs.Sensors.Conductivity, id_sbs.Sensors.Pressure]
    if has_secondary_tc:
        enabled_sensors += [id_sbs.Sensors.SecondaryTemperature, id_sbs.Sensors.SecondaryConductivity]

    # Everything AFTER the frequency block is a voltage channel, in SensorArray
    # order, regardless of what it physically is (oxygen, pH, altimeter, even
    # "NotInUse" placeholders still occupy a real channel slot in the byte
    # stream) - read_hex_file only needs the channel count/position here; which
    # equation to apply to each one happens later, in load_xmlcon/convert_data.
    ext_volt_names = ["ExtVolt0", "ExtVolt1", "ExtVolt2", "ExtVolt3",
                       "ExtVolt4", "ExtVolt5", "ExtVolt6", "ExtVolt7"]
    n_volt = n_entries - n_freq
    if n_volt > len(ext_volt_names):
        raise ValueError(
            f"{xml_file}: SensorArray has {n_volt} voltage-channel entries, "
            f"more than the {len(ext_volt_names)} ExtVolt slots read_hex_file supports."
        )
    for i in range(n_volt):
        enabled_sensors.append(getattr(id_sbs.Sensors, ext_volt_names[i]))

    # NMEA/time fields are also stated directly on the same <Instrument> block.
    if instrument.findtext("NmeaPositionDataAdded", "0") == "1":
        enabled_sensors.append(id_sbs.Sensors.nmeaLocation)
    if instrument.findtext("NmeaTimeAdded", "0") == "1":
        enabled_sensors.append(id_sbs.Sensors.nmeaTime)
    if instrument.findtext("NmeaDepthDataAdded", "0") == "1":
        enabled_sensors.append(id_sbs.Sensors.nmeaDepth)
    if instrument.findtext("ScanTimeAdded", "0") == "1":
        enabled_sensors.append(id_sbs.Sensors.SystemTime)

    return enabled_sensors, freq_supp, volt_supp


# Maps a SensorArray entry's XML tag to the CTDCoefficients field load_xmlcon
# fills in for it, and to the raw_data column convert_data should read for it.
# Add a row here whenever load_xmlcon learns to parse a new sensor type.
# OxygenSensor is NOT listed here - it's handled separately in detect_voltage_channel_map
# (counted like load_xmlcon's oxy_count) because some cruises carry two oxygen sensors on
# two different voltage channels (e.g. 2012 cast 117, 2026_1 cast 073).
_VOLTAGE_SENSOR_COEFF_FIELD = {
    "FluoroWetlabECO_AFL_FL_Sensor": "chlorophyll",
    "TurbidityMeter": "turbidity",
    "AltimeterSensor": "altimeter",
    # NOT YET SUPPORTED - no CTDCoefficients field or load_xmlcon parsing branch exists
    # for these yet, so their voltage channel is decoded but left unnamed/unconverted.
    # Seen in real XMLCONs: UserPolynomialSensor (UVP6) + pH_Sensor + WET_LabsCStar
    # (transmissometer) in 2026_2; FluoroSeatechWetlabsFLF_Sensor (older fluorometer,
    # distinct from FluoroWetlabECO_AFL_FL_Sensor above) in 2012.
}


def detect_voltage_channel_map(xml_file: str) -> dict:
    """Map each recognized auxiliary sensor to its 'volt N' raw_data column.

    INPUT:
      xml_file  =  path to a .XMLCON file               [ str ]

    OUTPUT:
      e.g. {"oxygen_primary": "volt 2", "chlorophyll": "volt 7"}    [ dict ]
    """
    root = ET.parse(xml_file).getroot()
    instrument = root.find(".//Instrument")
    sensor_array = instrument.find(".//SensorArray")
    sensors_by_index = {
        int(s.get("index")): list(s)[0].tag
        for s in sensor_array.findall("Sensor")
    }
    has_secondary_tc = (
        sensors_by_index.get(3) == "TemperatureSensor"
        and sensors_by_index.get(4) == "ConductivitySensor"
    )
    n_freq = 5 if has_secondary_tc else 3

    channel_map = {}
    oxy_count = 0
    for idx in sorted(i for i in sensors_by_index if i >= n_freq):
        tag = sensors_by_index[idx]
        if tag == "OxygenSensor":
            # Same primary/secondary counting convention as load_xmlcon's oxy_count below,
            # so a second oxygen sensor doesn't silently overwrite the first one's mapping.
            field = "oxygen_primary" if oxy_count == 0 else "oxygen_secondary"
            oxy_count += 1
        else:
            field = _VOLTAGE_SENSOR_COEFF_FIELD.get(tag)
        if field is not None:
            channel_map[field] = f"volt {idx - n_freq}"
    return channel_map


def load_xmlcon(xml_file: str) -> CTDCoefficients:
    """Read a .xmlcon to extract calibration coefficients.

    INPUT:
      xml_file  =  path to a .XMLCON file          [ str ]

    OUTPUT:
      calibration coefficients, one field per sensor type    [ CTDCoefficients ]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    coeffs = CTDCoefficients()

    temp_count = 0
    cond_count = 0
    oxy_count = 0

    for sensor in root.findall(".//Sensor"):

        # --------------------------------------------------
        # TEMPERATURE
        # --------------------------------------------------
        temp = sensor.find("TemperatureSensor")
        if temp is not None:
            obj = TemperatureFrequencyCoefficients(
                g=float(temp.findtext("G")),
                h=float(temp.findtext("H")),
                i=float(temp.findtext("I")),
                j=float(temp.findtext("J")),
                f0=float(temp.findtext("F0")),
            )
            if temp_count == 0:
                coeffs.temperature_primary = obj
            else:
                coeffs.temperature_secondary = obj
            temp_count += 1

        # --------------------------------------------------
        # CONDUCTIVITY
        # --------------------------------------------------
        cond = sensor.find("ConductivitySensor")
        if cond is not None:
            eq1 = cond.find('./Coefficients[@equation="1"]')
            if eq1 is not None:  # CHECK
                obj = ConductivityCoefficients(
                    g=float(eq1.findtext("G")),
                    h=float(eq1.findtext("H")),
                    i=float(eq1.findtext("I")),
                    j=float(eq1.findtext("J")),
                    cpcor=float(eq1.findtext("CPcor")),
                    ctcor=float(eq1.findtext("CTcor")),
                    wbotc=float(eq1.findtext("WBOTC")),
                )
                if cond_count == 0:
                    coeffs.conductivity_primary = obj
                else:
                    coeffs.conductivity_secondary = obj
                cond_count += 1

        # --------------------------------------------------
        # PRESSURE
        # --------------------------------------------------
        press = sensor.find("PressureSensor")
        if press is not None:
            coeffs.pressure = PressureDigiquartzCoefficients(
                c1=float(press.findtext("C1")),
                c2=float(press.findtext("C2")),
                c3=float(press.findtext("C3")),
                d1=float(press.findtext("D1")),
                d2=float(press.findtext("D2")),
                t1=float(press.findtext("T1")),
                t2=float(press.findtext("T2")),
                t3=float(press.findtext("T3")),
                t4=float(press.findtext("T4")),
                t5=float(press.findtext("T5")),
                ad590m=float(press.findtext("AD590M")),
                ad590b=float(press.findtext("AD590B")),
                offset=float(press.findtext("Offset")),
                slope=float(press.findtext("Slope")),
            )

        # --------------------------------------------------
        # OXYGEN
        # --------------------------------------------------
        oxy = sensor.find("OxygenSensor")
        if oxy is not None:
            eq1 = oxy.find('./CalibrationCoefficients[@equation="1"]')
            if eq1 is not None:
                obj = Oxygen43Coefficients(
                    soc=float(eq1.findtext("Soc")),
                    v_offset=float(eq1.findtext("offset")),
                    tau_20=float(eq1.findtext("Tau20")),
                    a=float(eq1.findtext("A")),
                    b=float(eq1.findtext("B")),
                    c=float(eq1.findtext("C")),
                    e=float(eq1.findtext("E")),
                    d0=float(eq1.findtext("D0")),
                    d1=float(eq1.findtext("D1")),
                    d2=float(eq1.findtext("D2")),
                    h1=float(eq1.findtext("H1")),
                    h2=float(eq1.findtext("H2")),
                    h3=float(eq1.findtext("H3")),
                )
                if oxy_count == 0:
                    coeffs.oxygen_primary = obj
                else:
                    coeffs.oxygen_secondary = obj
                oxy_count += 1

        # --------------------------------------------------
        # CHLOROPHYLL
        # --------------------------------------------------
        chl = sensor.find("FluoroWetlabECO_AFL_FL_Sensor")
        if chl is not None:
            coeffs.chlorophyll = ECOCoefficients(
                slope=float(chl.findtext("ScaleFactor")),
                offset=float(chl.findtext("Vblank")),
            )

        # --------------------------------------------------
        # TURBIDITY
        # --------------------------------------------------
        tur = sensor.find("TurbidityMeter")
        if tur is not None:
            coeffs.turbidity = ECOCoefficients(
                slope=float(tur.findtext("ScaleFactor")),
                offset=float(tur.findtext("DarkVoltage")),
            )

        # --------------------------------------------------
        # ALTIMETER
        # --------------------------------------------------
        alt = sensor.find("AltimeterSensor")
        if alt is not None:
            coeffs.altimeter = AltimeterCoefficients(
                slope=float(alt.findtext("ScaleFactor")),
                offset=float(alt.findtext("Offset")),
            )

    return coeffs

