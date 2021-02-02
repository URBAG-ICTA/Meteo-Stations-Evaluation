from src.noavg_WRFevaluation import WRFEvaluation_stations
from src.noavg_surface_WRFevaluation import surface_WRFEvaluation_stations
import datetime
import os
import pandas as pd

path_out = "../plots/"
plotTS = True #Plot temporal series
plotStats = True #Plot performance statistics
plotWindrose = False #Plot windroses, (still not fully developed)
year1 = 2015 #Starting year as in WRF namelist
year2 = 2015 #Ending year as in WRF namelist
month1 = 7 #Starting month as in WRF namelist
month2 = 7 #Ending month as in WRF namelist
day1 = 2 #Initial day as in WRF namelist +1 (one day of spin-up)
day2 = 9 #Final day as in WRF namelist
starthour = 0 #Starting hour of the validation (for all the time =0)
endhour = 24 #Ending hour of the validation (for all the time =24)
obs_path = "/data/co2flux/common/rsegura/DATA/MET_OBS/OBS_2015/"

path1 = "/data/co2flux/common/rsegura/Outputs/HeatWave-2015/BouLac_8days_Reference_noMORPH/wrfout_d03/"
label1 = "BL-noUM"
path2 = "/data/co2flux/common/rsegura/Outputs/HeatWave-2015/BouLac_8days_Reference/wrfout_d03/"
label2 = "BL-UM"
dir_name = "Sensitivity_study_2015_ALL_" #Name of the directory where plots are going to be generated

initial_date = datetime.datetime(year1, month1, day1)
final_date = datetime.datetime(year2, month2, day2)

we = WRFEvaluation_stations()
we.codes = ['WU', 'X4', 'XG']
we.ext = "wrfout_d03_"
we.end = "_00.nc" #Change this to _00:00:00 if you have not modified the name of WRF output files
we.initialize_evaluation(obs_path, initial_date, final_date)

print("Comparing data from building stations")
print("Extracting WRF data 1")
we.extract_WRF_data(path1, label1)
print("Extracting WRF data 2")
we.extract_WRF_data(path2, label2)


we2 = surface_WRFEvaluation_stations()

we2.codes = ['X8', 'XC', 'X2', 'XV', 'Y9', 'XL', '0076A','D3', 'XF', '0229I', '0189E', 'UG', 'D5', 'UF', '0194D', 'Y7']
we2.ext = "wrfout_d03_"
we2.end = "_00.nc" #Change this to _00:00:00 if you have not modified the name of WRF output files
we2.initialize_evaluation(obs_path, initial_date, final_date)

print("Comparing data from surface stations")
print("Extracting WRF data 1")
we2.extract_WRF_data(path1, label1)
print("Extracting WRF data 2")
we2.extract_WRF_data(path2, label2)

we2.codes = ['WU','X4','XG', 'X8', 'XC', 'X2', 'XV', 'Y9', 'XL', '0076A','D3', 'XF', '0229I', '0189E', 'UG', 'D5', 'UF', '0194D', 'Y7']
we2.dataFrame = pd.concat([we.dataFrame, we2.dataFrame])
we2.save_dataFrame('_MYJ_BouLac_2015')

#we2.reuse_dataFrame('WRFEvaluation_MYJ_BouLac_2015.csv') #Unindent this if you want to reuse a csv file
now = str(datetime.datetime.now())
now = now.replace(' ','_')
now = now[:-7]
rfile = dir_name+now
os.chdir(path_out)
os.mkdir(rfile)
os.chdir(path_out+rfile)
if plotWindrose:
    os.mkdir('Windroses')
print(rfile+' created')

if starthour <= endhour:
    if starthour == 0 and endhour == 24:
        we2.plot_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
    else:
        we2.day_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
else:
    we2.night_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
