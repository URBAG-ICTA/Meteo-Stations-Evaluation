# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import re
from sklearn.metrics import mean_squared_error

path_out = "./Plots/"
plotTS = True
plotStats = True
year1 = 2018
year2 = 2018
month1 = 7
month2 = 7
day1 = 17 #Initial day as in WRF simulation
day2 = 25 #Final day as in WRF simulation
obs_path = "C:/Users/1361078/Desktop/dadesEstacions/OBS_2018/"
path1 = "C:/Users/1361078/Desktop/Codis_Python/WRFout/WRFout_2018_Test/"
label1 = "new_SST"
path2 = "C:/Users/1361078/Desktop/Codis_Python/WRFout/WRFout_2018_First/"
label2 = "old_SST"


initial_date = datetime.datetime(year1, month1, day1)
final_date = datetime.datetime(year2, month2, day2)



class WRFEvaluation_stations():
    ext = ''
    end = ''
    dataFrame = pd.DataFrame()
    dates = []
    labels = []
    stationsfile = ''
    codes = []
    code_dict = {}
    
    def __init__(self):
        self.ext = "wrfout_d03_"
        self.end = "_00.nc"
        self.dataFrame = pd.DataFrame()
        self.dates = []
        self.labels = []
        self.stationsfile = "dadesEstacions.txt"
        self.codes = ['WU', 'X4', 'X8', 'XG', 'XC', 'X2', 'XV', 'Y9', 'XL', 'D3', 'XF', 'UG', 'WE', 'D5', 'UF', 'Y7']
        self.code_dict = {'WU':'Badalona', 'X4':'Raval', 'X8':'ZUni', 'XG':'ParetsV', 
                          'XC':'Castellbisbal', 'X2':'Zoo', 'XV':'SCugat', 'Y9':'ZALPrat', 'XL':'ElPrat',
                          'D3':'Vallirana', 'XF':'Sabadell', 'UG':'Viladecans', 'WE':'VilanovaV', 'D5':'ObsFabra',
                          'UF':'PNGarraf', 'Y7':'BocanaSud'}
    
    def initialize_evaluation(self, path_to_stations_files, initial_date, final_date):
        self.extract_stations_data(path_to_stations_files)
        self.filter_times(initial_date, final_date)
        info = open(path_to_stations_files+self.stationsfile,"r",encoding="ISO-8859-1")
        lines = info.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.split(';')
        self.addStationVariables(lines)
        self.dates = self.simulated_dates(initial_date, final_date)
        
    def extract_stations_data(self, path_to_stations_files):
        for code in self.codes:
            dataset = pd.read_csv(path_to_stations_files+code+".csv")
            self.dataFrame = pd.concat([self.dataFrame, dataset], ignore_index=True)
    
    def filter_times(self, date1, date2):
        Years = []
        Months = []
        Days = []
        Hours = []
        Minutes = []
        Ins = []
        for i, row in self.dataFrame.iterrows():
            data = row['DATA']
            data = re.split('-|T|:|Z', data)
            Years.append(int(data[0]))
            Months.append(int(data[1]))
            Days.append(int(data[2]))
            Hours.append(int(data[3]))
            Minutes.append(int(data[4]))
            Ins.append(int((datetime.datetime(int(data[0]), int(data[1]), int(data[2])) >= date1) and (datetime.datetime(int(data[0]), int(data[1]), int(data[2])) < date2)))
        self.dataFrame['Year'] = Years
        self.dataFrame['Month'] = Months
        self.dataFrame['Day'] = Days
        self.dataFrame['Hour'] = Hours
        self.dataFrame['Minute'] = Minutes
        self.dataFrame['Inside'] = Ins
        
        self.dataFrame = self.dataFrame[self.dataFrame['Minute'] < 30]
        self.dataFrame = self.dataFrame[self.dataFrame['Inside'] > 0]
        self.dataFrame = self.dataFrame.drop(['Minute', 'Inside'], axis=1)
    
    def addStationVariables(self, lines):
        header = lines.pop(0)
        lat_index = header.index('lat')
        lon_index = header.index('lon')
        code_index = header.index('AWS abr')
        vertT_index = header.index('V level T')
        vertW_index = header.index('V level W')
        
        self.dataFrame['lat'] = np.nan
        self.dataFrame['lon'] = np.nan
        self.dataFrame['VertT'] = np.nan
        self.dataFrame['VertW'] = np.nan

        for station in lines:
            self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == station[code_index], ['lat']] = float(station[lat_index])
            self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == station[code_index], ['lon']] = float(station[lon_index])
            self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == station[code_index], ['VertT']] = int(station[vertT_index])
            self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == station[code_index], ['VertW']] = int(station[vertW_index])
    
    def simulated_dates(self, initial_day, final_day):
        simulation = final_day - initial_day
        dates = []
        for i in range(0, simulation.days):
            day = initial_day + datetime.timedelta(i)
            dates.append(day.strftime("%Y-%m-%d"))
        return dates
        
    def extract_WRF_data(self, path_to_wrf_files, label):
        
        self.dataFrame[label+'_T'] = np.nan
        self.dataFrame[label+'_RH'] = np.nan
        self.dataFrame[label+'_WS'] = np.nan
        self.dataFrame[label+'_WD'] = np.nan
        self.labels.append(label)
        for i, date in enumerate(self.dates):
            file_name = path_to_wrf_files + self.ext + date + self.end
            wfile = Dataset(file_name, 'r')
            if i == 0:
                LON = np.array(wfile.variables['XLONG'][0])
                LAT = np.array(wfile.variables['XLAT'][0])
                LONU = np.array(wfile.variables['XLONG_U'][0])
                LATU = np.array(wfile.variables['XLAT_U'][0])
                LONV = np.array(wfile.variables['XLONG_V'][0])
                LATV = np.array(wfile.variables['XLAT_V'][0])
                
                self.find_cell_index(LON, LAT, '')
                self.find_cell_index(LONU, LATU, 'U')
                self.find_cell_index(LONV, LATV, 'V')

            VS = np.array(wfile.variables['V'])
            US = np.array(wfile.variables['U'])
            QVAPOR = np.array(wfile.variables['QVAPOR'])
            P = np.array(wfile.variables['P'])
            PB = np.array(wfile.variables['PB'])
            THETA = np.array(wfile.variables['T'])
            COSALPHA = np.array(wfile.variables['COSALPHA'])
            SINALPHA = np.array(wfile.variables['SINALPHA'])
            wfile.close()

            self.add_WRF_TEMPandRH(date, THETA, P, PB, QVAPOR, LAT, LON, label)
            self.add_WRF_W(date, US, VS, COSALPHA, SINALPHA, LAT, LON, label)


    def find_cell_index(self, LON, LAT, comp):
        I = 'ISW' + comp
        J = 'JSW' + comp
        self.dataFrame[I] = np.nan
        self.dataFrame[J] = np.nan
        for code in self.codes:
            distance = abs(LON-self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lon'].values[0]) + abs(LAT-self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lat'].values[0])
            minimumValue = np.amin(distance)
            res = np.where(distance == minimumValue)
            self.find_near_points(code, LON, LAT, comp, res[0][0], res[1][0])

    def find_near_points(self, code, LON, LAT, comp, I_station, J_station):
        lat_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lat'].values[0]
        lon_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lon'].values[0]
        I = 'ISW' + comp
        J = 'JSW' + comp
        if LAT[I_station][J_station] > lat_station:
            if LON[I_station][J_station] > lon_station:
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [I]] = I_station - 1 
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [J]] = J_station - 1
            else:
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [I]] = I_station - 1 
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [J]] = J_station
        else:
            if LON[I_station][J_station] > lon_station:
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [I]] = I_station 
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [J]] = J_station - 1
            else:
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [I]] = I_station 
                self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code, [J]] = J_station

    def add_WRF_TEMPandRH(self, date, THETA, P, PB, QVAPOR, LAT, LON, label):
        data = re.split('-', date)       
        for code in self.codes:
            ISW = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["ISW"].values[0])
            JSW = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code] ["JSW"].values[0])
            K = int(self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'] == code]["VertT"].values[0])
            lat_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lat'].values[0]
            lon_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lon'].values[0]
            for hour in range(24):
                TNE = self.temperature(THETA[hour][K][ISW + 1][JSW + 1], P[hour][K][ISW + 1][JSW + 1], PB[hour][K][ISW + 1][JSW + 1]) - 273.15
                TNW = self.temperature(THETA[hour][K][ISW + 1][JSW], P[hour][K][ISW + 1][JSW], PB[hour][K][ISW + 1][JSW]) - 273.15
                TSE = self.temperature(THETA[hour][K][ISW][JSW + 1], P[hour][K][ISW][JSW + 1], PB[hour][K][ISW][JSW + 1]) - 273.15
                TSW = self.temperature(THETA[hour][K][ISW][JSW], P[hour][K][ISW][JSW], PB[hour][K][ISW][JSW]) - 273.15
                T = self.weighted_mean(lat_station, lon_station, ISW, JSW, LAT, LON, TNE, TNW, TSE, TSW)
                RHNE = self.relative_humidity(QVAPOR[hour][K][ISW + 1][JSW + 1], TNE + 273.15,  P[hour][K][ISW + 1][JSW + 1], PB[hour][K][ISW + 1][JSW + 1])
                RHNW = self.relative_humidity(QVAPOR[hour][K][ISW + 1][JSW], TNW + 273.15,  P[hour][K][ISW + 1][JSW], PB[hour][K][ISW + 1][JSW])
                RHSE = self.relative_humidity(QVAPOR[hour][K][ISW][JSW + 1], TSE + 273.15,  P[hour][K][ISW][JSW + 1], PB[hour][K][ISW][JSW + 1])
                RHSW = self.relative_humidity(QVAPOR[hour][K][ISW][JSW], TSW + 273.15,  P[hour][K][ISW][JSW], PB[hour][K][ISW][JSW])
                RH = self.weighted_mean(lat_station, lon_station, ISW, JSW, LAT, LON, RHNE, RHNW, RHSE, RHSW)
                self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['Year'] == int(data[0])) & (self.dataFrame['Month'] == int(data[1])) & (self.dataFrame['Day'] == int(data[2])) & (self.dataFrame['Hour'] ==  hour), [label+'_T']] = T
                self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['Year'] == int(data[0])) & (self.dataFrame['Month'] == int(data[1])) & (self.dataFrame['Day'] == int(data[2])) & (self.dataFrame['Hour'] ==  hour), [label+'_RH']] = RH
                
    def add_WRF_W(self, date, US, VS, COSALPHA, SINALPHA, LAT, LON, label):
        data = re.split('-', date)
        for code in self.codes:
            ISWU = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["ISWU"].values[0])
            JSWU = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["JSWU"].values[0])
            ISWV = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["ISWV"].values[0])
            JSWV = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["JSWV"].values[0])
            K = int(self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]["VertW"].values[0])
            lat_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lat'].values[0]
            lon_station = self.dataFrame[self.dataFrame['CODI_ESTACIO'] == code]['lon'].values[0]
            for hour in range(24):
                UNE = self.rotate_u(US[hour][K][ISWU + 1][JSWU + 1], VS[hour][K][ISWU + 1][JSWU + 1], COSALPHA[hour][ISWU + 1][JSWU + 1], SINALPHA[hour][ISWU + 1][JSWU + 1])
                UNW = self.rotate_u(US[hour][K][ISWU + 1][JSWU], VS[hour][K][ISWU + 1][JSWU], COSALPHA[hour][ISWU + 1][JSWU], SINALPHA[hour][ISWU + 1][JSWU])
                USE = self.rotate_u(US[hour][K][ISWU][JSWU + 1], VS[hour][K][ISWU][JSWU + 1], COSALPHA[hour][ISWU][JSWU + 1], SINALPHA[hour][ISWU][JSWU + 1])
                USW = self.rotate_u(US[hour][K][ISWU][JSWU], VS[hour][K][ISWU][JSWU], COSALPHA[hour][ISWU][JSWU], SINALPHA[hour][ISWU][JSWU])
                U = self.weighted_mean(lat_station, lon_station, ISWU, JSWU, LAT, LON, UNE, UNW, USE, USW)
                VNE = self.rotate_v(US[hour][K][ISWV + 1][JSWV + 1], VS[hour][K][ISWV + 1][JSWV + 1], COSALPHA[hour][ISWV + 1][JSWV + 1], SINALPHA[hour][ISWV + 1][JSWV + 1])
                VNW = self.rotate_v(US[hour][K][ISWV + 1][JSWV], VS[hour][K][ISWV + 1][JSWV], COSALPHA[hour][ISWV + 1][JSWV], SINALPHA[hour][ISWV + 1][JSWV])
                VSE = self.rotate_v(US[hour][K][ISWV][JSWV + 1], VS[hour][K][ISWV][JSWV + 1], COSALPHA[hour][ISWV][JSWV + 1], SINALPHA[hour][ISWV][JSWV + 1])
                VSW = self.rotate_v(US[hour][K][ISWV][JSWV], VS[hour][K][ISWV][JSWV], COSALPHA[hour][ISWV][JSWV], SINALPHA[hour][ISWV][JSWV])
                V = self.weighted_mean(lat_station, lon_station, ISWV, JSWV, LAT, LON, VNE, VNW, VSE, VSW)
                WS = self.speed(U, V)
                WD = self.direction(U, V)
                self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['Year'] == int(data[0])) & (self.dataFrame['Month'] == int(data[1])) & (self.dataFrame['Day'] == int(data[2])) & (self.dataFrame['Hour'] ==  hour), [label+'_WS']] = WS
                self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['Year'] == int(data[0])) & (self.dataFrame['Month'] == int(data[1])) & (self.dataFrame['Day'] == int(data[2])) & (self.dataFrame['Hour'] ==  hour), [label+'_WD']] = WD            
    
    def weighted_mean(self, lat_station, lon_station, ISW, JSW, LAT, LON, NE, NW, SE, SW):
        lat_NE = LAT[ISW + 1][JSW + 1]
        lon_NE = LON[ISW + 1][JSW + 1]
        lat_NW = LAT[ISW + 1][JSW]
        lon_NW = LON[ISW + 1][JSW]
        lat_SE = LAT[ISW][JSW + 1]
        lon_SE = LON[ISW][JSW + 1]
        lat_SW = LAT[ISW][JSW]
        lon_SW = LON[ISW][JSW]
        Weight_NW = abs((lon_NE - lon_station)/(lon_NE - lon_NW))
        Weight_SW = abs((lon_SE - lon_station)/(lon_SE - lon_SW))
        N = NE*(1 - Weight_NW) +  NW*Weight_NW
        S = SE*(1 - Weight_SW) + SW*Weight_SW
        Nlat = lat_NE*(1 - Weight_NW) + lat_NW*Weight_NW
        Slat = lat_SE*(1 - Weight_SW) + lat_SW*Weight_SW
        Weight_N = abs((Slat - lat_station)/(Nlat - Slat))
        Weighted_value = N*Weight_N + S*(1 - Weight_N)
        return Weighted_value
                
    def temperature(self, theta, p, pb):
        T = (theta + 300)*pow((p + pb)/100000, 2/7)
        return T
    
    def relative_humidity(self, qvapor, temperature, p, pb):
        e0 = 6.112
        b = 17.67
        T1 = 273.15
        T2 = 29.65
        eps = 0.622
        es = float(e0*math.exp(b*(temperature - T1)/(temperature - T2)))
        qs = float(eps*es/(((p + pb)/100.) - (1 - eps)*es))
        rh = 100*qvapor/qs
        if rh > 100:
            rh = 100
        if rh < 0:
            rh = 0
        return rh
            
    def rotate_u(self, u, v, cosa, sina):
        return u*cosa -v*sina
    
    def rotate_v(self, u, v, cosa, sina):
        return u*sina + v*cosa
    
    def speed(self, u, v):
        return math.sqrt(u**2 + v**2)
    
    def direction(self, u, v):
        angle = math.atan2(-u, -v)
        if angle < 0:
            angle = angle + 2*math.pi
        angle = math.degrees(angle)
        return angle
    
    def save_dataFrame(self, dataframe, abrev = ''):
        dataframe.to_csv('./WRFEvaluation' + abrev + '.csv', index = None, header = True)
        
    def reuse_dataFrame(self, file_name):
        self.dataFrame = pd.read_csv(file_name)
        columns = self.dataFrame.columns.values
        for column in columns:
            if '_WS' in column:
                self.labels.append(re.split('_WS', column)[0])
        times = self.dataFrame['DATA'].unique()
        dates = []
        for date in times:
            date = date.split('T')[0]
            if date not in dates:
                dates.append(date)
        self.dates = dates
    
    def plot_results(self, Plot_TS, Plot_metrics, path_out):
        times = np.sort(self.dataFrame['DATA'].unique())
        list_of_temperatures = [[]]
        codesT = [] 
        list_of_humidities = [[]]
        list_of_wind_speed = [[]]
        list_of_wind_direction = [[]]
        codesW = []
        for code in self.codes:
            T = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['T'].notnull()), ['T']].values, axis=1)
            RH = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['HR'].notnull()), ['HR']].values, axis=1)
            if len(T) != 0:
                list_of_temperatures[0].append(T)
                list_of_humidities[0].append(RH)
                codesT.append(code)
                
            WS = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['VV10'].notnull()), ['VV10']].values, axis=1)
            WD = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'] == code) & (self.dataFrame['DV10'].notnull()), ['DV10']].values, axis=1)
            if len(WS) != 0:
                list_of_wind_speed[0].append(WS)
                list_of_wind_direction[0].append(WD)
                codesW.append(code)
        for i, model in enumerate(self.labels):
            list_of_temperatures.append([])
            list_of_humidities.append([])
            list_of_wind_speed.append([])
            list_of_wind_direction.append([])
            for code in self.codes:
                T = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'].isin(codesT)) & (self.dataFrame['CODI_ESTACIO'] == code), [model+'_T']].values, axis=1)
                RH = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'].isin(codesT)) & (self.dataFrame['CODI_ESTACIO'] == code), [model+'_RH']].values, axis=1)
                if len(T) != 0:
                    list_of_temperatures[i+1].append(T)
                    list_of_humidities[i+1].append(RH)
                WS = np.mean(self.dataFrame.loc[(self.dataFrame['CODI_ESTACIO'].isin(codesW)) & (self.dataFrame['CODI_ESTACIO'] == code), [model+'_WS']].values, axis=1)
                WD = np.mean(self.dataFrame.loc[self.dataFrame['CODI_ESTACIO'].isin(codesW) & (self.dataFrame['CODI_ESTACIO'] == code), [model+'_WD']].values, axis=1)
                if len(WS) != 0:
                    list_of_wind_speed[i+1].append(WS)
                    list_of_wind_direction[i+1].append(WD)

        if Plot_TS or Plot_metrics:
            self.plot_variable(times, list_of_temperatures, 'Temperature', codesT, 'Temperature ($^\circ$C)', 'TEMP', Plot_TS, Plot_metrics, path_out, None)
            self.plot_variable(times, list_of_humidities, 'Relative humidity', codesT, 'Relative humidity (%)', 'HUM', Plot_TS, Plot_metrics, path_out, (0,100))
            self.plot_variable(times, list_of_wind_speed, 'Wind speed', codesW, 'Wind speed (m/s)', 'WS', Plot_TS, Plot_metrics, path_out, None)
            self.plot_variable(times, list_of_wind_direction, 'Wind direction', codesW, 'Wind direction ($^\circ$)', 'WD', Plot_TS, Plot_metrics, path_out, (0,360))
            
    def plot_variable(self, times, list_of_variable, title, codes, definition, abrev, Plot_TS, Plot_metrics, path_out, ylim):
        Evaluation = pd.DataFrame()
        Evaluation['Station'] = [self.code_dict[x] for x in codes]
        for i in range(1, len(list_of_variable)):
            Evaluation['RMSE ' + self.labels[i-1]] = np.nan
            Evaluation['MB ' + self.labels[i-1]] = np.nan
            Evaluation['R ' + self.labels[i-1]] = np.nan
        for i in range(len(list_of_variable[0])):
            arrays = pd.DataFrame()
            arrays['times'] = times
            for j in range(len(list_of_variable)):
                if j == 0:
                    arrays['0. Observed'] = list_of_variable[j][i]
                else:
                    arrays[str(j) + '. ' + self.labels[j - 1]] = list_of_variable[j][i] 
                    if Plot_metrics:
                        Evaluation.loc[Evaluation['Station'] == self.code_dict[codes[i]], ['RMSE ' + self.labels[j - 1]]] = math.sqrt(mean_squared_error(arrays['0. Observed'],arrays[str(j) + '. ' + self.labels[j - 1]]))
                        Evaluation.loc[Evaluation['Station'] == self.code_dict[codes[i]], ['MB ' + self.labels[j - 1]]] = (arrays[str(j) + '. ' + self.labels[j - 1]] - arrays['0. Observed']).mean()
                        Evaluation.loc[Evaluation['Station'] == self.code_dict[codes[i]], ['R ' + self.labels[j - 1]]] = arrays[str(j) + '. ' + self.labels[j - 1]].corr(arrays['0. Observed'])
            if Plot_TS:
                self.plot_TS(times, arrays, title, codes, i, definition, abrev, path_out, ylim)
        if Plot_metrics:
            self.save_dataFrame(Evaluation, '_' + abrev)
            self.plot_metrics(Evaluation, title, codes, definition, abrev, path_out)  
      
    def RMSE(self, x, y):
        diff = (x - y)**2
        return math.sqrt(np.mean(diff))
    
    def R(self, x, y):
        mx = np.mean(x)
        my = np.mean(y)
        return np.sum((x - np.full((len(x)), mx))*(y-np.full((len(y)), my)))/(math.sqrt(np.sum((x - np.full((len(x)), mx))**2))*math.sqrt(np.sum((y - np.full((len(y)), my))**2)))

    def MB(self, x, y):
        return np.mean(x - y)

    def plot_TS(self, times, arrays, title, codes, i , definition, abrev, path_out, ylimit):
        dfi = arrays
        dfi.times = pd.to_datetime(dfi.times)
        dfi.set_index('times', inplace=True)
        colors = ['black','blue', 'red', 'yellow', 'magenta', 'green']
        ax = dfi.plot(style = '.-', color = colors, markersize = 2, linewidth = 1, ylim = ylimit)
        ax.xaxis.grid(True, which='both')
        plt.ylabel(definition, fontsize=16)
        plt.legend(loc = 'best', fontsize = 'large')
        plt.grid(axis = 'x')
        plt.xlabel('Time', fontsize = 16)
        plt.title(self.code_dict[codes[i]], fontsize = 18)
        plt.tight_layout()
        plt.savefig(path_out + abrev + '_' + self.code_dict[codes[i]], dpi = 100)
    
    def plot_metrics(self, Evaluation, title, names, definition, abrev, path_out):
        colors = ['blue', 'red', 'yellow', 'magenta', 'green']
        Hrows = [self.code_dict[x] for x in names]
        n = len(self.labels)

        fig, ax = plt.subplots()
        index = np.arange(len(Evaluation['RMSE ' + self.labels[0]]))
        bar_width = 0.25
        opacity = 0.8
        for i in range(n):
            rects = plt.bar(index + (n*i + 1)*bar_width/n , Evaluation['RMSE ' + self.labels[i]].values, bar_width, 
                            alpha = opacity,
                            color = colors[i],
                            label = self.labels[i])
        plt.xticks(index + bar_width, Hrows, rotation = -45, ha = "left", rotation_mode = "anchor")
        plt.ylabel('RMSE ' + definition, fontsize = 16)
        plt.title(title, fontsize = 18)
        plt.legend(loc = 'best', fontsize = 'large')
        plt.grid(axis = 'y')
        plt.axhline(0, color = 'black')
        plt.tight_layout()
        plt.savefig(path_out + 'RMSE_' + abrev)
        
        fig, ax = plt.subplots()
        index = np.arange(len(Evaluation['MB ' + self.labels[0]]))
        bar_width = 0.25
        opacity = 0.8
        for i in range(n):
            rects = plt.bar(index + (n*i + 1)*bar_width/n , Evaluation['MB ' + self.labels[i]].values, bar_width, 
                            alpha = opacity,
                            color = colors[i],
                            label = self.labels[i])
        plt.xticks(index + bar_width, Hrows, rotation = -45, ha = "left", rotation_mode = "anchor")
        plt.ylabel('Mean Bias ' + definition, fontsize = 16)
        plt.title(title, fontsize = 18)
        plt.legend(loc = 'best', fontsize = 'large')
        plt.grid(axis = 'y')
        plt.axhline(0, color = 'black')
        plt.tight_layout()
        plt.savefig(path_out + 'MB_' + abrev)
        
        fig, ax = plt.subplots()
        index = np.arange(len(Evaluation['R ' + self.labels[0]]))
        bar_width = 0.25
        opacity = 0.8
        for i in range(n):
            rects = plt.bar(index + (n*i + 1)*bar_width/n , Evaluation['R ' + self.labels[i]].values, bar_width, 
                            alpha = opacity,
                            color = colors[i],
                            label = self.labels[i])
        plt.ylim(0, 1)
        plt.xticks(index + bar_width, Hrows, rotation = -45, ha = "left", rotation_mode = "anchor")
        plt.ylabel('R ' + title, fontsize = 16)
        plt.title(title, fontsize = 18)
        plt.legend(loc = 'best', fontsize = 'large')
        plt.grid(axis = 'y')
        plt.axhline(0, color = 'black')
        plt.tight_layout()
        plt.savefig(path_out + 'CORR_' + abrev)
        
we = WRFEvaluation_stations()
we.initialize_evaluation(obs_path, initial_date, final_date)
we.extract_WRF_data(path1, label1)
we.extract_WRF_data(path2, label2)
we.save_dataFrame(we.dataFrame)
#we.reuse_dataFrame('C:/Users/1361078/Desktop/TFM-Sergi/WRFEvaluation.csv')  
we.plot_results(True, True, path_out)      
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        