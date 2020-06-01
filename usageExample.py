from src.WRFevaluation import WRFEvaluation_stations
import datetime
import os

path_out = "../plots/" #Path to plots directory
plotTS = False #Choose if you want to plot time series of the variables
plotStats = True #Choose if you want to plot statistical metrics plots
plotWindrose = False #Choose if you want to make windrose plots of the selected period
year1 = 2015 #Initial year as in WRF siulation
year2 = 2015 #Final year as in WRF simulation
month1 = 6 #Initial month as in WRF simulation
month2 = 7 #Final month as in WRF simulation
day1 = 25 #Initial day as in WRF simulation
day2 = 15 #Final day as in WRF simulation
starthour = 0 #Starting hour of the evaluation (0 in case of whole day)
endhour = 24 #Final hour of the evaluation (24 in case of whole day)
obs_path = "/nfs/co2flux/rsegura/DATA/MET_OBS/OBS_2015/" #Path to the observed data
path1 = "/nfs/co2flux/rsegura/Outputs/HeatWave-2015/Try_with_MYJ_BEP-BEM/wrfout_d03/" #Path to wrf output files 1
label1 = "MYJ_1month" #Label for the case 1
path2 = "/nfs/co2flux/rsegura/Outputs/HeatWave-2015/Try_with_MYJ_BULK/wrfout_d03/" #Path to wrf output files 2
label2 = "1month_BULK" #Label for the case 2

dir_name = "Comparison_MYJ_BouLac_2015_"

initial_date = datetime.datetime(year1, month1, day1)
final_date = datetime.datetime(year2, month2, day2)

we = WRFEvaluation_stations()
we.codes = ['WU', 'X4', 'X8', 'XG', 'XC', 'X2', 'XV', 'XL', 'D3', 'XF', 'UG', 'WE', 'D5', 'UF', 'Y7']
we.end = "_00.nc"
we.initialize_evaluation(obs_path, initial_date, final_date)
print("Extracting WRF data 1") 
we.extract_WRF_data(path1, label1)
print("Extracting WRF data 2")
we.extract_WRF_data(path2, label2)
#we.reuse_dataFrame('WRFEvaluation_MYJ_BouLac_2015.csv') #Unindent in case you want to reuse csv file
#we.remove_model(label1)
we.save_dataFrame(we.dataFrame, '_MYJ_BouLac_2015')

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

if starthour >= endhour:
    if starthour == 0 and endhour == 24:
        we.plot_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
    else:
        we.day_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
else:
    we.night_results(plotTS, plotStats, plotWindrose, './', starthour, endhour)
