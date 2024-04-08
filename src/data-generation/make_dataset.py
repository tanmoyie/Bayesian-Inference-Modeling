# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Data Preparation of PLeR

# Generating data based on EOS tool <br>
# Developed by: TANMOY DAS <br>
# Date: Jan 2021 (Revised Apr 2024)

# Load Excel file of EOS tool & get sheet names
from openpyxl import load_workbook
# import values from Python to Excel
# import xlsxwriter
import numpy as np
from numpy.random import randint, uniform, rand
import xlwings as xw
import pandas as pd

#%%
data_eos_PLeR = []

for i in range(0, 10):
    # get input data
    # write_values_to_eos
    # def data_preparation():
    EOS_tool = load_workbook(filename='EOS_ver.1.1._2020.xlsx')
    S1_basic_data = EOS_tool['Step 1 - Basic data']
    S1_VEC_Persistance = EOS_tool['Step 1 - VEC and Persistence']
    S1_oil_spill_modeling = EOS_tool['Step 1 - Oil spill modelling']
    S2_pollution_assessment = EOS_tool['Step 2 - Pollution Assessment']
    S2_OSR_pros_cons = EOS_tool['Step 2 - OSR pros_cons']
    S3_soot_pollution_index = EOS_tool['Step 3 - Soot pollution index']
    S3_VEC_R = EOS_tool['Step 3 - Recovery time']
    S3_VEC_recruitement_Fraction = EOS_tool['Step 3 - Recruitment_Fractions']

    # Determining the variables

    # S1_basic_data
    name_of_area = 'Nanuvut'
    sea_surface_area = uniform(20, 100)  # [20,100, uniform]
    water_depth = sea_surface_area*0.1
    seawater_volume = sea_surface_area*water_depth
    seabed_area = sea_surface_area*0.5
    shoreline_length = sea_surface_area*0.05
    number_of_seasons = 1

    depth_of_halocline = sea_surface_area*0.04
    seawater_volume_to_halocline = sea_surface_area*0.4
    salinity_above_halocline = 0.5
    oxygen_level_in_bottom_water = 1
    water_temp = -10
    nutritional_conditions = 1
    n = 0
    p = 0

    NEC_zooplankton = uniform(0, 0.5)  # 0 #[0,.5, uniform]
    NEC_bivalves = uniform(0, 0.9)  # 0 #[0,.9, uniform]
    NEC_fish = uniform(0, 0.15)  # [0,.15, uniform]
    LC50_zooplankton = 1
    LC50_bivalves = 2
    LC50_fish = 1
    damage_feather = 0.1
    update_water_in_feather = 3

    # ------------- Step 1 - VEC and Persistance
    species_name = 'Coral reefs'
    displacement = np.random.choice(['yes', 'no'])
    residue_recovery = np.random.choice(['yes', 'no'])
    species_in_sea_surface = randint(0, 2)  # np.random.rand(0,2)0 #[0, 1]
    seawater = randint(0, 2)  # [0, 1]
    seabed = randint(0, 2)  # [0, 1]
    shoreline = randint(0, 2)  # [0, 1]
    retaining_capacity = randint(0, 2)

    # --------------- Step 1 - Oil spill modelling
    oil_spill_size = np.random.choice(['SMALL', 'MEDIUM', 'LARGE'])
    if oil_spill_size == 'SMALL':
        oil_spill_amount = randint(0, 7)  # 7 exclusive
    elif oil_spill_size == 'MEDIUM':
        oil_spill_amount = randint(7, 700)
    else:
        oil_spill_amount = randint(700, 5000)

    number_of_scenarios = 1
    # 'light' # ['light', 'bunker oil' 'diesel oil']
    oil_type = np.random.choice(['light', 'bunker oil', 'diesel oil'])
    oil_density = randint(0, 50)
    oil_viscosity = randint(2, 22)
    oil_amount_to_recover = uniform(0.1, 2)
    duration = randint(20, 180)

    wind = randint(10, 100)  # [10,100, uniform]
    air_temp = randint(-50, 2)  # [-50, -30,-10,0]
    ice = randint(0, 100)  # [0,100, uniform]
    release_point = 1
    start_time = 1
    end_time = 1
    simulation_length = 1
    output_from_modeling_after = 1

    seasurface_volume = randint(0, 500)  # [1,500, uniform]
    seawater_volume = randint(0, 500)  # [1,500, uniform]
    seabed_volume = randint(0, 500)  # [1,500, uniform]
    shoreline_volume = randint(0, 500)  # [1,500, uniform]
    total_volume = seasurface_volume + \
        seawater_volume + seabed_volume + shoreline_volume
    evaporated = 10
    naturally_dispersed = 20
    water_content = 50
    evaporation_and_natural_disperson = randint(10, 100)  # [10, 100, uniform]

    distance_to_inhabitation = randint(10, 500)  # [10,200, 1000]
    distance_to_sensitive_organism = randint(10, 500)  # [10,200, 1000]
    distance_to_permanent_ice = randint(20, 500)  # [10,200, 1000]
    distance_to_sensitive_site = randint(10, 500)  # [10,200, 1000]
    prevailing_wind_direction = 'N'  # ['N','E','S','W', 'NE']

    # --------------- S2
    sufficient_mixing_energy = np.random.choice(['yes', 'no'])
    residual_recovery = np.random.choice(['yes', 'no'])

    ss_mcr = randint(-1, 2)
    sw_mcr = randint(-1, 2)
    sb_mcr = randint(-1, 2)
    sl_mcr = randint(-1, 2)
    ss_cdu = randint(-1, 2)
    sw_cdu = randint(-1, 2)
    sb_cdu = randint(-1, 2)
    sl_cdu = randint(-1, 2)
    ss_isb = randint(-1, 2)
    sw_isb = randint(-1, 2)
    sb_isb = randint(-1, 2)
    sl_isb = randint(-1, 2)
    ss_dn = randint(-1, 2)
    sw_dn = randint(-1, 2)
    sb_dn = randint(-1, 2)
    sl_dn = randint(-1, 2)

    # ----------- S3_soot_pollution_index
    wind_to_inhabitation = np.random.choice(['yes', 'no'])
    wind_to_sensitive_organism = wind_to_inhabitation
    wind_to_ince = wind_to_inhabitation  # np.random.choice(['yes', 'no'])
    wind_to_sensitive_object = wind_to_inhabitation
    # S3_VEC_R
    recovery_time_ss = uniform(0.2, 2)
    recovery_time_sl = uniform(0.2, 2)
    recovery_time_sw = uniform(0.2, 2)
    recovery_time_sb = uniform(0.2, 2)
    recovery_time_limit = 2
    # ----------- S3_VEC_recruitement_Fraction
    biodegradation_potential = np.random.choice(['yes', 'no'])
    sea_surface_area_polluted_limit = 2
    seawater_volume_fraction_polluted_limit = 10
    seawater_volume_fraction_polluted_if_nutrient_not_limiting = 15

    # ------------
    # ---------------- Assign values
    # Assigning values in Step 1 - Basic data
    S1_basic_data['E11'] = sea_surface_area
    S1_basic_data['E12'] = water_depth
    S1_basic_data['E13'] = seawater_volume
    S1_basic_data['E14'] = seabed_area
    S1_basic_data['E15'] = shoreline_length
    S1_basic_data['D17'] = number_of_seasons

    S1_basic_data['E20'] = depth_of_halocline
    S1_basic_data['E21'] = seawater_volume_to_halocline
    S1_basic_data['E22'] = salinity_above_halocline
    S1_basic_data['E23'] = oxygen_level_in_bottom_water
    S1_basic_data['E24'] = water_temp
    S1_basic_data['E25'] = nutritional_conditions
    S1_basic_data['E26'] = n
    S1_basic_data['E27'] = p

    S1_basic_data['E32'] = NEC_zooplankton
    S1_basic_data['F32'] = NEC_bivalves
    S1_basic_data['G32'] = NEC_fish
    S1_basic_data['E33'] = LC50_zooplankton
    S1_basic_data['F33'] = LC50_bivalves
    S1_basic_data['G33'] = LC50_fish
    S1_basic_data['E37'] = damage_feather
    S1_basic_data['E38'] = update_water_in_feather

    # ---------- S1_VEC
    S1_VEC_Persistance['N13'] = displacement
    S1_VEC_Persistance['N14'] = residual_recovery

    S1_VEC_Persistance['C13'] = species_name
    S1_VEC_Persistance['P13'] = species_in_sea_surface
    S1_VEC_Persistance['Q13'] = seawater
    S1_VEC_Persistance['R13'] = seabed
    S1_VEC_Persistance['S13'] = shoreline
    S1_VEC_Persistance['S41'] = retaining_capacity

    # ------- S1_oil
    S1_oil_spill_modeling['D17'] = oil_spill_size
    S1_oil_spill_modeling['D16'] = oil_spill_amount

    S1_oil_spill_modeling['E18'] = number_of_scenarios
    S1_oil_spill_modeling['D21'] = oil_type
    S1_oil_spill_modeling['D22'] = oil_density
    S1_oil_spill_modeling['D23'] = oil_viscosity
    S1_oil_spill_modeling['D24'] = oil_amount_to_recover
    S1_oil_spill_modeling['D25'] = duration

    S1_oil_spill_modeling['D28'] = wind
    S1_oil_spill_modeling['D29'] = air_temp
    S1_oil_spill_modeling['D30'] = ice
    S1_oil_spill_modeling['D32'] = release_point
    S1_oil_spill_modeling['D33'] = start_time
    S1_oil_spill_modeling['D34'] = end_time
    S1_oil_spill_modeling['D35'] = simulation_length
    S1_oil_spill_modeling['D37'] = output_from_modeling_after

    S1_oil_spill_modeling['D41'] = seasurface_volume
    S1_oil_spill_modeling['E41'] = seawater_volume
    S1_oil_spill_modeling['F41'] = seabed_volume
    S1_oil_spill_modeling['G41'] = shoreline_volume
    S1_oil_spill_modeling['H41'] = total_volume
    S1_oil_spill_modeling['I41'] = evaporated
    S1_oil_spill_modeling['J41'] = naturally_dispersed
    S1_oil_spill_modeling['K41'] = water_content
    S1_oil_spill_modeling['P41'] = evaporation_and_natural_disperson

    S1_oil_spill_modeling['D57'] = distance_to_inhabitation
    S1_oil_spill_modeling['E57'] = distance_to_sensitive_organism
    S1_oil_spill_modeling['F57'] = distance_to_permanent_ice
    S1_oil_spill_modeling['G57'] = distance_to_sensitive_site
    S1_oil_spill_modeling['H57'] = prevailing_wind_direction

    # -------S2

    S2_pollution_assessment['E32'] = sufficient_mixing_energy

    S2_OSR_pros_cons['F11'] = ss_mcr
    S2_OSR_pros_cons['J11'] = sw_mcr
    S2_OSR_pros_cons['N11'] = sb_mcr
    S2_OSR_pros_cons['R11'] = sl_mcr

    S2_OSR_pros_cons['F13'] = ss_cdu
    S2_OSR_pros_cons['J13'] = sw_cdu
    S2_OSR_pros_cons['N13'] = sb_cdu
    S2_OSR_pros_cons['R13'] = sl_cdu

    S2_OSR_pros_cons['F15'] = ss_isb
    S2_OSR_pros_cons['J15'] = sw_isb
    S2_OSR_pros_cons['N15'] = sb_isb
    S2_OSR_pros_cons['R15'] = sl_isb

    S2_OSR_pros_cons['F17'] = ss_dn
    S2_OSR_pros_cons['J17'] = sw_dn
    S2_OSR_pros_cons['N17'] = sb_dn
    S2_OSR_pros_cons['R17'] = sl_dn

    # -----------
    S3_soot_pollution_index['F20'] = wind_to_inhabitation
    S3_soot_pollution_index['F21'] = wind_to_sensitive_organism
    S3_soot_pollution_index['F22'] = wind_to_ince
    S3_soot_pollution_index['F23'] = wind_to_sensitive_object

    # --------
    S3_VEC_R['D8'] = recovery_time_ss
    S3_VEC_R['E8'] = recovery_time_sl
    S3_VEC_R['F8'] = recovery_time_sw
    S3_VEC_R['G8'] = recovery_time_sb
    S3_VEC_R['D15'] = recovery_time_limit

    # -------
    S3_VEC_recruitement_Fraction['D12'] = biodegradation_potential
    S3_VEC_recruitement_Fraction['D15'] = sea_surface_area_polluted_limit
    S3_VEC_recruitement_Fraction['D17'] = seawater_volume_fraction_polluted_limit
    S3_VEC_recruitement_Fraction['D18'] = seawater_volume_fraction_polluted_if_nutrient_not_limiting

    # Save workbook before running VBA (so that values are stored)
    EOS_tool.save('EOS_ver.1.1._2020.xlsx')
    EOS_tool.close()

    # run VBA
    data_EOS_DT = xw.Book("data_preparation_VBA_code.xlsm")
  #  macro_closeActive = data_EOS_DT.macro("Module1.macro_closeActive")
  #  macro_closeActive()

    macro1 = data_EOS_DT.macro("Module1.ExtractData")
    macro1()
    # get that cell
    oil_spill_size = data_EOS_DT.sheets['EOS Results']['C3'].value
    evaporation_and_natural_disperson = data_EOS_DT.sheets['EOS Results']['C4'].value
    persistence = data_EOS_DT.sheets['EOS Results']['C5'].value
    oil_amount_to_recover = data_EOS_DT.sheets['EOS Results']['C6'].value
    E_ss = data_EOS_DT.sheets['EOS Results']['C7'].value
    E_sl = data_EOS_DT.sheets['EOS Results']['C8'].value
    E_sw = data_EOS_DT.sheets['EOS Results']['C9'].value
    E_sb = data_EOS_DT.sheets['EOS Results']['C10'].value

    sufficient_mixing_energy = data_EOS_DT.sheets['EOS Results']['E4'].value
    seasurface = data_EOS_DT.sheets['EOS Results']['E5'].value
    seawater = data_EOS_DT.sheets['EOS Results']['E6'].value
    Rtime_ss = data_EOS_DT.sheets['EOS Results']['E7'].value
    Rtime_sw = data_EOS_DT.sheets['EOS Results']['E8'].value
    E_ssC = data_EOS_DT.sheets['EOS Results']['E9'].value
    E_slC = data_EOS_DT.sheets['EOS Results']['E10'].value
    E_swC = data_EOS_DT.sheets['EOS Results']['E11'].value
    E_sbC = data_EOS_DT.sheets['EOS Results']['E12'].value

    soot_pollution = data_EOS_DT.sheets['EOS Results']['G4'].value
    residue_recovery = data_EOS_DT.sheets['EOS Results']['G5'].value
    displacement = data_EOS_DT.sheets['EOS Results']['G6'].value
    E_ssI = data_EOS_DT.sheets['EOS Results']['G7'].value
    E_slI = data_EOS_DT.sheets['EOS Results']['G8'].value
    E_swI = data_EOS_DT.sheets['EOS Results']['G9'].value
    E_sbI = data_EOS_DT.sheets['EOS Results']['G10'].value

    shoreline_length = shoreline_length
    distance_to_inhabitation = distance_to_inhabitation

    mcr_DT_output = list(
        filter(None, data_EOS_DT.sheets['EOS Results']['B3:B15'].value))
    cdu_DT_output = list(
        filter(None, data_EOS_DT.sheets['EOS Results']['D3:D13'].value))
    isb_DT_output = list(
        filter(None, data_EOS_DT.sheets['EOS Results']['F3:F13'].value))

    # Add this iteration into data_PLeR file (list/dataframe)
    row_data = [oil_spill_size, evaporation_and_natural_disperson, persistence, oil_amount_to_recover, E_ss, E_sl, E_sw, E_sb,
                sufficient_mixing_energy, seasurface, seawater, Rtime_ss, Rtime_sw, E_ssC, E_slC, E_swC, E_sbC,
                soot_pollution, residue_recovery, displacement,  E_ssI, E_slI, E_swI, E_sbI,
                shoreline_length, distance_to_inhabitation,
                mcr_DT_output, cdu_DT_output, isb_DT_output]
    data_eos_PLeR.append(row_data)

    data_EOS_DT.close()
    # return row_data

data_eos_PLeR_df = pd.DataFrame(data_eos_PLeR, columns=[
                                'oil_spill_size', 'evaporation_and_natural_disperson', 'persistence', 'oil_amount_to_recover', 'E_ss', 'E_sl', 'E_sw', 'E_sb',
                                'sufficient_mixing_energy', 'seasurface', 'seawater', 'Rtime_ss', 'Rtime_sw', 'E_ssC', 'E_slC', 'E_swC', 'E_sbC',
                                'soot_pollution', 'residue_recovery', 'displacement', 'E_ssI', 'E_slI', 'E_swI', 'E_sbI',
                                'shoreline_length', 'distance_to_inhabitation',
                                'mcr_DT_output', 'cdu_DT_output', 'isb_DT_output'])

data_eos_PLeR_df.to_csv('raw_data/data_04.2024.csv')
df1 = pd.read_csv('raw_data/data_04.2024.csv')
df2 = data_eos_PLeR_df
frames = [df1, df2]
df22 = df1.append(df2)
data_EOS_DT = xw.Book("raw_data/data_EOS_DT.xlsm")
EOS_tool.save('raw_data/data_PLeR.xlsx')
EOS_tool.close
