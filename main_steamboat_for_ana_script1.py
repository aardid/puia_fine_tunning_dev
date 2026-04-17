from datetime import timedelta
#from puia.tests import run_tests
from puia.model import ForecastModel,MultiVolcanoForecastModel,MultiDataForecastModel
from puia.data import SeismicData, GeneralData
from puia.utilities import datetimeify, load_dataframe
from glob import glob
from sys import platform
import pandas as pd
import numpy as np
import os, shutil, json, pickle, csv
import matplotlib.pyplot as plt


_MONTH=timedelta(days=365.05/12)
_DAY = timedelta(days=1)

def forecast_test():
    ''' test scale forecast model
    '''
    # define data 
    data={'YNM':['2018-01-04','2019-06-30'],}
    eruptions={'YNM':[i for i in range(0,57)],}
    # define datastreams 
    data_streams = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']
    #
    fm=MultiVolcanoForecastModel(data=data, window=2.0, overlap=0.75, look_forward=2.0, data_streams=data_streams, root='test')
    drop_features=['linear_trend_timewise','agg_linear_trend']
    #
    exclude_dates={}
    for _sta in eruptions.keys():
        exclude_dates[_sta]=None        
    #train
    fm.train(drop_features=drop_features, retrain=True, Ncl=300, n_jobs=6, exclude_dates=exclude_dates)        

    #compute forecast over the an eruption period
    te = fm.data['YNM'].tes[-1]#datetimeify(eruptions['YNM'][-1])
    tf=te+_DAY*1
    ti=te-_DAY*7

    # forecast in high resolution 
    fm.hires_forecast(station='YNM', ti=ti, tf=tf, recalculate=True, n_jobs=6,  threshold=.75, 
        root='test', save='plots'+os.sep+'test'+os.sep+'_fc_eruption_'+'YNM'+'_'+str(eruptions['YNM'][-1])+'.png')    

def main():
    forecast_test()
    pass

if __name__=='__main__':
    main()
