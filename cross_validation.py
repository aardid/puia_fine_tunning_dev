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

_MONTH=timedelta(days=365.25/12)
_DAY = timedelta(days=1)

# set path depending on OS
if platform == "linux" or platform == "linux2":
    root=r'/media/eruption_forecasting/eruptions'
elif platform == "win32":
    root=r'U:'+os.sep+'Research'+os.sep+'EruptionForecasting'+os.sep+'eruptions'
    # root=r'C:\Users\dde62\code\alberto\EruptionForecasting'

DATA_DIR=f'{root}'+os.sep+'data'
FEAT_DIR=f'{root}'+os.sep+'features'
MODEL_DIR=f'{root}'+os.sep+'models'
FORECAST_DIR=f'{root}'+os.sep+'forecasts'

# define pool of volcanoes and record times
test = True
phreatic_pool = False
magmatic_pool = False
camerica_pool = False
full = False
#
if test:
    print('running test pool')
    data={#'PUNZ':['2013-02-15','2022-12-30'], # .5 years
            'PUNZTA2':['2013-03-01','2022-12-30'], # .5 years
            #'YFT':['2024-01-01','2024-07-25'], # .5 years
            #'YNM':['2018-01-04','2019-06-30'],
            #'COP':['2020-03-09','2022-11-19'], # 3 years
            #'COP':['2020-03-10','2022-11-01'],
            #'WIZ':['2011-01-01','2020-01-15'], # 10 years
            }
    eruptions={#'PUNZ':[0,1,2,3],
                'PUNZTA2':[0,1,2,3],
                #'PVV':[0,1,2,3,4,5],'PN7A':[0,1,2,3,4,5]#6],#[0,1,2], 
                #'PVV':[0,1,2],
                #'CRPO3':[0,1,2,3,4],
                #'YFT':[0],
                #'YNM':[i for i in range(0,57)],
                #'COP':[0,1,2,3,4,5],
                #'WIZ':[0,1,2,3,4],
                #'SHW':[0],
                #'ONTA':[0],
                } 
if phreatic_pool:
    print('running phreatic_pool')
    data={
        'WIZ':['2010-01-03','2020-01-31'], # 10 years
            'FWVZ':['2006-01-01','2015-12-31'], # 13 years      
            'KRVZ':['2010-01-01','2019-12-31'],# 10 years               
            'ONTA':['2013-01-10','2014-12-18'],#['2014-07-02','2014-11-19'], 
            #'COP':['2020-03-09','2020-11-30'], # 3 years
            'SHW':['2004-01-02','2005-12-30'], 
            #'BELO':['2007-08-22','2010-07-10'], # 3 years 
            #'MBGH':['2001-05-26','2007-05-28'], 
            #'CETU':['2019-06-13','2019-12-22'],      
            #'CRPO':['2014-02-01','2017-04-22'], # 3 years
            #'CRPO2':['2017-03-01','2017-04-22'],
            #'POS':['2013-01-01','2013-07-28'], # 3 years
            #'CAU':['2011-05-13', '2012-05-31']#'2011-06-05'], # 3 year
            
            }
    eruptions={
        'WIZ':[0,1,2,3,4],
                'FWVZ':[0,1,2],#2,1,0],
                'KRVZ':[0,1],#1,0],
                'ONTA':[0],
                #'COP':[0,1,2],
                'SHW':[0],
                #'BELO':[0,1,2,3,4],#[0,1,2,3,4],#[0,3,4],
                #'MBGH':[0,1],  
                #'CETU':[0,1],
                #'CRPO':[0,1,2] 
                #'CRPO2':[2], 
                #'POS':[0],
                #'CAU':[0], # 3 year
                }  
if magmatic_pool:
    print('running magmatic_pool')
    data={
        'VNSS':['2013-01-01','2019-12-31'], # 6 years
           'BELO':['2007-08-22','2010-07-10'], # 3 years
           'REF':['2009-01-02','2009-04-08'],
           'AUH':['2005-01-02','2006-10-31'],
           'CETU':['2019-06-13','2019-12-22'],
           'GSTR':['2019-07-01','2021-05-27'],
           'PVV':['2014-02-01','2016-06-30'],
           'OKWR':['2008-01-01','2009-12-30'],
           'SHW':['2004-01-02','2005-12-30'],
           
           #'VONK':['2014-07-15','2014-08-30'],# 8 months#['2014-01-02','2015-07-15'], # 1.5 years
           #'GOD':['2010-03-06','2010-05-29'],#  months
           #'MBGH':['2001-05-26','2007-05-28'], 
            }
    eruptions={
        'VNSS':[0,1], 
               'BELO':[0,1,2,3,4],#[0,1,2,3,4],#[0,3,4],
               'REF':[0],
               'AUH':[0],
               'CETU':[0,1],
               'GSTR':[0],
               'PVV':[0,1,2],
               'OKWR':[0],
               'SHW':[0],
               
               #'GOD':[1],
               #'VONK':[0],     
               #'MBGH':[0,1],            
                }  
if camerica_pool:
    print('running camerica_pool')
    data={'VTUN':['2014-08-06','2015-12-28'],
            #'TBTN':['2010-10-02','2013-06-29'],
            #'MEA01':['2013-12-18','2014-07-26'],
            'MBGH':['2001-05-26','2007-05-28'],
            'VRLE':['2014-08-05','2017-10-03'],
            }
    eruptions={'VTUN':[0,1,2,3],#,
                #'TBTN':[0,1,2,3],#, 
                #'MEA01':[0],#,  
                'MBGH':[0,1],
                'VRLE':[0,1,2],
                }  
if full:
    print('running full pool')
    data={'WIZ':['2010-01-03','2020-01-31'], # 10 years
            'FWVZ':['2006-01-01','2015-12-31'], # 13 years      
            'KRVZ':['2010-01-01','2019-12-31'],# 10 years               
            'ONTA':['2013-01-10','2014-12-18'],#['2014-07-02','2014-11-19'], 
            'COP':['2020-03-09','2020-11-30'], # 3 years         
            #'CRPO':['2014-02-01','2017-04-22'], # 3 years
            #'CRPO2':['2017-03-01','2017-04-22'],
            #'POS':['2013-01-01','2013-07-28'], # 3 years

           'VNSS':['2013-01-01','2019-12-31'], # 6 years
           'BELO':['2007-08-22','2010-07-10'], # 3 years
           'REF':['2009-01-02','2009-04-08'],
           #'AUH':['2005-01-02','2006-10-31'],
           'CETU':['2019-06-13','2019-12-22'],
           'GSTR':['2019-07-01','2021-05-27'],
           'PVV':['2014-02-01','2016-06-30'],
           'OKWR':['2008-01-01','2009-12-30'],
           'SHW':['2004-01-02','2005-12-30'],
           
           #'VONK':['2014-07-15','2014-08-30'],# 8 months#['2014-01-02','2015-07-15'], # 1.5 years
           #'GOD':['2010-03-06','2010-05-29'],#  months
            
            'VTUN':['2014-08-06','2015-12-28'],
            #'TBTN':['2010-10-02','2013-06-29'],
            #'MEA01':['2013-12-18','2014-07-26'],
            'MBGH':['2001-05-26','2007-05-28'],
            'VRLE':['2014-08-05','2017-10-03'],
            #'CAU':['2011-05-13', '2011-06-05'], # 3 year
            }
    eruptions={'WIZ':[0,1,2,3,4],
                'FWVZ':[0,1,2],#2,1,0],
                'KRVZ':[0,1],#1,0],
                'ONTA':[0],
                'COP':[0,1,2],  
                #'CRPO':[0,1,2] 
                #'CRPO2':[2], 
                #'POS':[0],

                'VNSS':[0,1], 
                'BELO':[0,1,2,3,4],#[0,1,2,3,4],#[0,3,4],
                'REF':[0],
                #'AUH':[0],
                'CETU':[0,1],
                'GSTR':[0],
                'PVV':[0,1,2],
                'OKWR':[0],
                'SHW':[0],
                
                #'GOD':[1],
                #'VONK':[0], 
               
                'VTUN':[0,1,2,3],#,
                #'TBTN':[0,1,2,3],#, 
                #'MEA01':[0],#,  
                'MBGH':[0,1],
                'VRLE':[0,1,2],
                #'CAU':['0'], # 3 year
                }
                
# model hyper pars
window=2.
overlap=0.75
look_forward=window#2.
data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']#['zsc2_rsam','zsc2_dsar','zsc2_mf','zsc2_hf']
data_streams=['zsc2_rsam', 'zsc2_dsar']#['rsam','mf','hf','dsar']
data_streams=['zsc2_rsam','mw',]#'dsar']
Ncl=300
n_jobs=64

# multi-resolution scales: list of (window_days, resample_minutes) or None for legacy
# Example: scales=[(2, 10), (14, 60), (60, 360), (180, 1440)]
scales=None

print('window '+str(window))
print('overlap '+str(overlap))
print('n_jobs '+str(n_jobs))

# define thresholds for performance metrics 
ths = np.linspace(0, 1, num=101)#101)
#ths = ths[:-1]

def cross_validation_multi_volcano_leave_eruption():
    pass
    #define training model
    model_dir = 'models'+os.sep+'cve_'+'_'.join(data.keys())
    forecast_dir = 'forecasts'+os.sep+'cve_'+'_'.join(data.keys())
    print('model dir: '+model_dir)
    print('forecast dir: '+forecast_dir)
    
    # run training with all eruptions in the pool and forecast over the whole periods
    if True:  
        print('00')
        erup = '00'
        fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward, data_streams=data_streams,
                feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                    root=str(erup),
                        model_dir=model_dir,
                            forecast_dir=forecast_dir, scales=scales)
        #
        drop_features=['linear_trend_timewise','agg_linear_trend']
        exclude_dates={}
        for _sta in eruptions.keys():
            exclude_dates[_sta]=None        
        #train
        #fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
        fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        

        # compute forecast over whole stations period
        for _sta in data.keys():
            tf=datetimeify(data[_sta][1])#-1*_DAY
            ti=datetimeify(data[_sta][0])#+1*_DAY
            fm.hires_forecast(station=_sta, ti=ti, tf=tf, recalculate=False, n_jobs=n_jobs, 
                root='cve_'+'_'.join(data.keys())+os.sep+str(erup)+os.sep+str(_sta), threshold=1.0)  
    
    # loop over eruptions (stations and eruptions)
    if True:
        print(eruptions)
        for sta in data.keys():
            pass
            print(sta)
            for erup in eruptions[sta]:
                pass
                print(erup)
                fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward,
                    data_streams=data_streams,
                        feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                            root=sta+'_'+str(erup),
                                model_dir=model_dir,
                                    forecast_dir=forecast_dir, scales=scales)
                #
                drop_features=['linear_trend_timewise','agg_linear_trend']
                #exclude data from eruption (train a model with the following data excluded)
                te1=fm.data[sta].tes[erup]
                exclude_dates={sta:[[te1-_MONTH, te1+_MONTH]]}
                for _sta in eruptions.keys(): # temporal, need to be fixed (exclude eruptions not considered in eruptions, but present in the .txt file)
                    if _sta != sta:
                        for _i in range(len(fm.data[_sta].tes)):
                            if _i not in eruptions[_sta]:
                                _te=fm.data[_sta].tes[_i]
                                if _sta in exclude_dates:
                                    exclude_dates[_sta].append([_te-_MONTH, _te+_MONTH]) 
                                else:
                                    exclude_dates[_sta]=[[_te-_MONTH, _te+_MONTH]] 
                for _sta in eruptions.keys():
                    if _sta not in exclude_dates:
                        exclude_dates[_sta]=None
                
                #train
                #fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)  
                fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
                
                #compute forecast over the eruptions excluded (for png)
                tf=te1+_DAY*4#MONTH#/12#_MONTH*.1
                ti=te1-_MONTH
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+os.sep+sta+'_'+str(erup),
                        save='forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+'_fc_eruption_'+sta+'_'+str(erup)+'.png') 
                
                #compute forecast over the eruptions excluded (for master consensus)
                tf=te1+_DAY*4
                ti=te1-_MONTH
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+os.sep+sta+'_'+str(erup)) # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+
                    
        # save txt with cross validation info 
        with open('forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+'readme.txt', 'w') as f:
            f.write('data\n')
            #f.write(json.dumps(data)) 
            for sta in data.keys():
                f.write(sta+'\t')
                for dt in data[sta]:
                    f.write(str(dt)+'\t')  
            f.write('\n')
            f.write('eruptions\n')
            #f.write(json.dumps(eruptions))
            for sta in eruptions.keys():
                f.write(sta+'\t')
                for erup in eruptions[sta]:
                    f.write(str(erup)+'\t') 
            f.write('\n')
            f.write('window '+str(window)+'\n')
            f.write('overlap '+str(overlap)+'\n')
            f.write('look_forward '+str(look_forward)+'\n')
            f.write('data_streams '+str(data_streams)+'\n')
            f.write('Ncl '+str(Ncl)+'\n')
            f.write('n_jobs '+str(n_jobs)+'\n')
        
    # construct master consensus
    if True:
        # one time series per volcano where eruptive periods (two days before-to eruption) from out of sample and non-eruptive taken from 
        # consensus were all eruptions used in training
        # (1) import master consensus over the station records from '00' model (all eruptions used in training)
        for sta in data.keys():
            #Check all consensus files and concatenate (file are per year of record) 
            _path = forecast_dir+os.sep+'00'+os.sep+sta
            _con = [x[2] for x in os.walk(_path)][0]

            _con = [x for x in _con if 'consensus' in x]
            _consensus_master = pd.concat([pd.read_pickle(_path+os.sep+x) for x in _con])
            _consensus_master.sort_index(inplace=True)
            # (2) for each eruption in out of sample models, look for consensus during the eruptive period and replace on master consensus
            for erup in eruptions[sta]:
                print(sta)
                print(erup)
                _path = forecast_dir+os.sep+sta+'_'+str(erup)
                print(_path)
                _consensus_erup = pd.read_pickle(glob(_path+os.sep+'*consensus*.pkl')[0])
                # replace erup_consensus into master consensus  
                if True: # need to test
                    # drop rows in _consensus_master dor the range in _consensus_erup
                    l1 = _consensus_master.index.get_loc(_consensus_erup.index[0], method='nearest')
                    l2 = _consensus_master.index.get_loc(_consensus_erup.index[-1], method='nearest')
                    _consensus_master.drop(_consensus_master.index[list(range(l1,l2+1,1))], inplace=True)
                    # concat _consensus_master with _consensus_erup and sort
                    _consensus_master = pd.concat([_consensus_master,_consensus_erup])
                    _consensus_master.sort_index(inplace=True)  
                    #                  
            # (3) save master consensus for each station in main folder 
            _consensus_master.to_pickle(forecast_dir+os.sep+'_consensus_master_'+sta+'.pkl')
            pass

def cross_validation_multi_volcano_leave_volcano():
    pass
    #define training model
    model_dir = 'models'+os.sep+'cvv_'+'_'.join(data.keys())
    forecast_dir = 'forecasts'+os.sep+'cvv_'+'_'.join(data.keys())
    print('model dir: '+model_dir)
    print('forecast dir: '+forecast_dir)
    
    # loop over volcanoes (stations) to leave out on training 
    if True:
        print(eruptions)
        for sta in data.keys():
            pass
            print(sta)
            # get eruption times 
            fl_nm=DATA_DIR+os.sep+sta+'_eruptive_periods.txt'
            with open(fl_nm,'r') as fp:
                tes=[datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #exclude data from eruptions of sta       
            exclude_dates={}
            for _sta in data.keys():
                exclude_dates[_sta]=None
            exclude_dates[sta]=[[te-_MONTH, te+_MONTH] for te in tes]
            #print(exclude_dates)
            #    
            fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward, data_streams=data_streams,
                    feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                        root=sta,
                            model_dir=model_dir,
                               forecast_dir=forecast_dir, scales=scales)
            drop_features=['linear_trend_timewise','agg_linear_trend']
            ##train
            fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)
            
            ##compute forecast over the eruptions excluded
            for k,te in enumerate(tes):
                tf=te+_DAY*1#+_MONTH*.1
                ti=te-_MONTH*1
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=False, n_jobs=n_jobs,  threshold=1., 
                    root='cvv_'+'_'.join(data.keys())+os.sep+sta,
                        save='forecasts'+os.sep+'cvv_'+'_'.join(data.keys())+os.sep+'_fc_eruption_'+sta+'_'+str(k)+'.png') # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+
            #compute forecast over the eruptions excluded
            tf=datetimeify(data[sta][1])-_DAY*1
            ti=datetimeify(data[sta][0])+_DAY*1
            fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                root='cvv_'+'_'.join(data.keys())+os.sep+sta) # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+        
        # save txt with cross validation info 
        with open('forecasts'+os.sep+'cvv_'+'_'.join(data.keys())+os.sep+'readme.txt', 'w') as f:
            f.write('data\n')
            #f.write(json.dumps(data)) 
            for sta in data.keys():
                f.write(sta+'\t')
                for dt in data[sta]:
                    f.write(str(dt)+'\t')  
            f.write('\n')
            f.write('eruptions\n')
            #f.write(json.dumps(eruptions))
            for sta in eruptions.keys():
                f.write(sta+'\t')
                for erup in eruptions[sta]:
                    f.write(str(erup)+'\t') 
            f.write('\n')
            f.write('window '+str(window)+'\n')
            f.write('overlap '+str(overlap)+'\n')
            f.write('look_forward '+str(look_forward)+'\n')
            f.write('data_streams '+str(data_streams)+'\n')
            f.write('Ncl '+str(Ncl)+'\n')
            f.write('n_jobs '+str(n_jobs)+'\n')
        
    # construct master consensus
    if True:
        # (1) for each eruption in out of sample models, look for consensus during the eruptive period and replace on master consensus
        for sta in data.keys():
            #Check all consensus files and concatenate (file are per year of record) 
            _path = forecast_dir+os.sep+sta
            _con = [x[2] for x in os.walk(_path)][0]
            _con = [x for x in _con if 'consensus' in x]
            _consensus_sta = pd.concat([pd.read_pickle(_path+os.sep+x) for x in _con])
            _consensus_sta.sort_index(inplace=True) 
            # (3) save master consensus for each station in main folder 
            _consensus_sta.to_pickle(forecast_dir+os.sep+'_consensus_master_'+sta+'.pkl')

def cross_validation_multi_volcano_leave_eruption_multi_station():
    pass
    #define training model
    model_dir = 'models'+os.sep+'cve_'+'_'.join(data.keys())+'_ms'
    forecast_dir = 'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+'_ms'
    print('model dir: '+model_dir)
    print('forecast dir: '+forecast_dir)
    
    # run training with all eruptions in the pool and forecast over the whole periods
    if True:  
        print('00')
        erup = '00'
        fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward, data_streams=data_streams,
                feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                    root=str(erup),
                        model_dir=model_dir,
                            forecast_dir=forecast_dir, scales=scales)
        #
        drop_features=['linear_trend_timewise','agg_linear_trend']
        exclude_dates={}
        for _sta in eruptions.keys():
            exclude_dates[_sta]=None        
        #train
        #fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
        fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
        #
        # compute forecast over whole stations period
        for _sta in data.keys():
            tf=datetimeify(data[_sta][1])#-1*_DAY
            ti=datetimeify(data[_sta][0])#+1*_DAY
            fm.hires_forecast(station=_sta, ti=ti, tf=tf, recalculate=False, n_jobs=n_jobs, 
                root='cve_'+'_'.join(data.keys())+'_ms'+os.sep+str(erup)+os.sep+str(_sta), threshold=1.0)  
    # loop over eruptions (stations and eruptions)
    if True:
        print(eruptions)
        for sta in data.keys(): # loop station 
            pass
            print(sta)
            for erup in eruptions[sta]:  # loop eruption 
                pass
                print(erup)
                fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward,
                    data_streams=data_streams,
                        feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                            root=sta+'_'+str(erup),
                                model_dir=model_dir,
                                    forecast_dir=forecast_dir, scales=scales)
                #
                drop_features=['linear_trend_timewise','agg_linear_trend']
                #exclude data from eruption (train a model with the following data excluded)
                te1=fm.data[sta].tes[erup]
                exclude_dates={sta:[[te1-_MONTH, te1+_DAY*1]]}# te1+_MONTH]]}
                for _sta in eruptions.keys(): # temporal, need to be fixed (exclude eruptions not considered in eruptions, but present in the .txt file)
                    #if _sta != sta: ## diferent from normal cve (remove section from both stations)
                        for _i in range(len(fm.data[_sta].tes)):
                            if _i not in eruptions[_sta]:
                                _te=fm.data[_sta].tes[_i]
                                if _sta in exclude_dates:
                                    exclude_dates[_sta].append([_te-_MONTH, _te+_MONTH]) 
                                else:
                                    exclude_dates[_sta]=[[_te-_MONTH, _te+_MONTH]] 
                for _sta in eruptions.keys():
                    if _sta not in exclude_dates:
                        exclude_dates[_sta]=None
                #train
                #fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)  
                fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
                #compute forecast over the eruptions excluded (for png)
                tf=te1+_DAY*1#_MONTH#/12#_MONTH*.1
                ti=te1-_MONTH
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+'_ms'+os.sep+sta+'_'+str(erup),
                        save='forecasts'+os.sep+'cve_'+'_'.join(data.keys())+'_ms'+os.sep+'_fc_eruption_'+sta+'_'+str(erup)+'.png') 
                #compute forecast over the eruptions excluded (for master consensus)
                tf=te1+_MONTH
                ti=te1+_DAY#te1-_MONTH
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+'_ms'+os.sep+sta+'_'+str(erup)) # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+
        # save txt with cross validation info 
        with open('forecasts'+os.sep+'cve_'+'_'.join(data.keys())+'_ms'+os.sep+'readme.txt', 'w') as f:
            f.write('data\n')
            #f.write(json.dumps(data)) 
            for sta in data.keys():
                f.write(sta+'\t')
                for dt in data[sta]:
                    f.write(str(dt)+'\t')  
            f.write('\n')
            f.write('eruptions\n')
            #f.write(json.dumps(eruptions))
            for sta in eruptions.keys():
                f.write(sta+'\t')
                for erup in eruptions[sta]:
                    f.write(str(erup)+'\t') 
            f.write('\n')
            f.write('window '+str(window)+'\n')
            f.write('overlap '+str(overlap)+'\n')
            f.write('look_forward '+str(look_forward)+'\n')
            f.write('data_streams '+str(data_streams)+'\n')
            f.write('Ncl '+str(Ncl)+'\n')
            f.write('n_jobs '+str(n_jobs)+'\n')
        
    # construct master consensus
    if True:
        # one time series per volcano where eruptive periods (two days before-to eruption) from out of sample and non-eruptive taken from 
        # consensus were all eruptions used in training
        # (1) import master consensus over the station records from '00' model (all eruptions used in training)
        for sta in ['YFT']:#data.keys():
            #Check all consensus files and concatenate (file are per year of record) 
            _path = forecast_dir+os.sep+'00'+os.sep+sta
            _con = [x[2] for x in os.walk(_path)][0]

            _con = [x for x in _con if 'consensus' in x]
            _consensus_master = pd.concat([pd.read_pickle(_path+os.sep+x) for x in _con])
            _consensus_master.sort_index(inplace=True)
            # (2) for each eruption in out of sample models, look for consensus during the eruptive period and replace on master consensus
            for erup in eruptions[sta]:
                print(sta)
                print(erup)
                _path = forecast_dir+os.sep+sta+'_'+str(erup)
                print(_path)
                _consensus_erup = pd.read_pickle(glob(_path+os.sep+'*consensus*.pkl')[0])
                # replace erup_consensus into master consensus  
                if True: # need to test
                    # drop rows in _consensus_master dor the range in _consensus_erup
                    l1 = _consensus_master.index.get_loc(_consensus_erup.index[0], method='nearest')
                    l2 = _consensus_master.index.get_loc(_consensus_erup.index[-1], method='nearest')
                    _consensus_master.drop(_consensus_master.index[list(range(l1,l2+1,1))], inplace=True)
                    # concat _consensus_master with _consensus_erup and sort
                    _consensus_master = pd.concat([_consensus_master,_consensus_erup])
                    _consensus_master.sort_index(inplace=True)  
                    #                  
            # (3) save master consensus for each station in main folder 
            _consensus_master.to_pickle(forecast_dir+os.sep+'_consensus_master_'+sta+'.pkl')
            pass
  
def ROC_cross_validation(dir_path):
    '''
    '''
    only_for_station = False#'PVV'#'BELO' # None
    rand = False #generate a randomized master consensus
    print('roc curve on '+dir_path)
    # Define path for cross validation results 
    #dir_path='forecasts'+os.sep+'cve_'+'_'.join([sta for sta in data.keys()])
    #dir_path='forecasts'+os.sep+'cve_WIZ_FWVZ_KRVZ'
    #(0) Define vector of thresholds
    l_fpr, l_tpr, l_sen, l_spc, l_tp, l_fn, l_fp, l_tn,l_prec= [],[],[],[],[],[],[],[],[]
    l_dal_non, l_dal = [],[]
    # get stations names (in pool) 
    #stas = dir_path.split(os.sep)[-1].split('_')[1:]
    # (1) loop over thresholds 
    for j, th in enumerate(ths):
        print(j)
        # define total TP FN TN and FP
        c_tp, c_fn, c_tn, c_fp=0,0,0,0
        c_dal, c_dal_non = 0, 0
        # (2) loop over stations
        if only_for_station:
            sta = only_for_station
            # get master consensus 
            _consensus = pd.read_pickle(dir_path+os.sep+'_consensus_master_'+sta+'.pkl')
            _consensus.to_csv(dir_path+os.sep+'_consensus_master_'+sta+'.csv')
            #_consensus=_consensus.rolling(int(window)*24*6).median()
            # get eruption times 
            fl_nm=DATA_DIR+os.sep+sta+'_eruptive_periods.txt'
            with open(fl_nm,'r') as fp:
                tes=[datetimeify(ln.rstrip()) for ln in fp.readlines()]
            # count TP and FN in eruptive record (and add to total)
            for k, te in enumerate(tes):
                #if te in 
                inds = (_consensus.index<te-window*_DAY)|(_consensus.index>=te) 
                _max = _consensus.loc[~inds].quantile(q=0.95)['consensus']
                #_consensus.loc[~inds].plot()
                if _max>=th: 
                    c_tp+=288 
                else: 
                    c_fn+=288 
                _consensus=_consensus.loc[inds]
            # count TN and FP in non-eruptive record (and add to total)
            _idx_bool = _consensus['consensus']<th
            c_tn += len(_consensus[_idx_bool])
            c_fp += len(_consensus[~_idx_bool])
            if True: # count days of alert (and non)
                # resample consensus to two days
                _consensus_aux=_consensus.resample('2D').quantile(q=0.95)#, int(window)*24*6).median()#(q=0.95)                
                _idx_bool = _consensus_aux['consensus']<th
                c_dal_non += len(_consensus_aux[_idx_bool])*2
                c_dal += len(_consensus_aux[~_idx_bool])*2
        else:
            for sta in data.keys():
                # get master consensus 
                _consensus = pd.read_pickle(dir_path+os.sep+'_consensus_master_'+sta+'.pkl')
                #_consensus.to_csv(dir_path+os.sep+'_consensus_master_'+sta+'.csv')
                #_consensus=_consensus.rolling(int(window)*24*6).median()
                if rand: # randomize (shuffle) consensus values 
                    print('running randomized')
                    import random
                    _aux=_consensus.iloc[:,0].values
                    random.shuffle(_aux)
                    _consensus['_consensus_rand'] = _aux
                    _consensus = _consensus.drop('_consensus_rand', axis=1)
                    #_consensus = _consensus.drop('_consensus_rand', axis=1)
                if False:
                    if os.path.isfile(DATA_DIR+os.sep+sta+'_unrest_periods.txt'): # filter unrest
                        with open(DATA_DIR+os.sep+sta+'_unrest_periods.txt','r') as fp:
                            #ln = ln.rstrip().split(',')
                            unr=[[datetimeify(ln.rstrip().split(',')[0]),datetimeify(ln.rstrip().split(',')[1])] for ln in fp.readlines()]
                            for un in enumerate(unr):
                                t1,t2=un[1][0], un[1][1]
                                inds = (_consensus.index < t1 )|(_consensus.index > t2 ) 
                                _consensus = _consensus.loc[inds]
                # get eruption times 
                fl_nm=DATA_DIR+os.sep+sta+'_eruptive_periods.txt'
                with open(fl_nm,'r') as fp:
                    tes=[datetimeify(ln.rstrip()) for ln in fp.readlines()]
                # count TP and FN in eruptive record (and add to total)
                for k, te in enumerate(tes):
                    #if te in 
                    inds = (_consensus.index<te-window*_DAY)|(_consensus.index>=te) 
                    _max = _consensus.loc[~inds].quantile(q=0.95)['consensus']
                    #_consensus.loc[~inds].plot()
                    if _max>=th: 
                        c_tp+=288 
                    else: 
                        c_fn+=288 
                    _consensus=_consensus.loc[inds]
                # count TN and FP in non-eruptive record (and add to total)
                _idx_bool = _consensus['consensus']<th
                c_tn += len(_consensus[_idx_bool])
                c_fp += len(_consensus[~_idx_bool])
                if True: # count days of alert (and non)
                    # resample consensus to two days
                    _consensus_aux=_consensus.resample('2D').quantile(q=0.95)#, int(window)*24*6).median()#(q=0.95)                
                    _idx_bool = _consensus_aux['consensus']<th
                    c_dal_non += len(_consensus_aux[_idx_bool])*2
                    c_dal += len(_consensus_aux[~_idx_bool])*2
        # compute FPR, TPR, SENS and SPEC
        tpr = c_tp/(c_tp+c_fn)
        fpr = c_fp/(c_fp+c_tn)
        sen = c_tp/(c_tp+c_fn) 
        spc = c_tn/(c_tn+c_fp) 
        try:
            prec = c_tp/(c_tp+c_fp)
        except:
            prec = 1.
        #
        # append to lists
        l_tp.append(c_tp)
        l_fn.append(c_fn)
        l_fp.append(c_fp)
        l_tn.append(c_tn)
        l_fpr.append(fpr) 
        l_tpr.append(tpr)
        l_sen.append(sen)
        l_spc.append(spc) 
        l_dal_non.append(c_dal_non)
        l_dal.append(c_dal) 
        l_prec.append(prec)
        
    # rate of days under alert vs rate of eruptions cougth in alert 
    l_r_dal = np.asarray(l_dal)/(np.asarray(l_dal)+np.asarray(l_dal_non)) # x axis 
    l_r_erup_dal = (np.asarray(l_tp))/(np.asarray(l_tp)+np.asarray(l_fn))#(np.asarray(l_tp)/288)/((np.asarray(l_tp)/288)+np.asarray(l_dal))
    #
    if True: # basic stats
        ## compute accuracy:number of correctly classify instances (TP + TN) divided by the number of instances
        ## compute precision: % of correctly labelled positives instances (TP) out all positive labelled instances (TP+FP)
        ## compute recall:% of correctly labelled positives instances (TP) out of all positives instances (TP+FN)
        if rand:
            with open('forecasts'+os.sep+dir_path.split(os.sep)[-1].split('_')[0]+'_'+'_'.join(data.keys())+os.sep+'perf_pars_rand.csv', 'w') as f:
                f.write('threshold,TP,FN,FP,TN,ACCU,PREC,REC,SEN,SPC,FPR,TPR,DAL,NDAL,RDAL,REDAL\n')
                for th,c_tp,c_fn,c_fp,c_tn,sen,spc,fpr,tpr,dal,ndal,rdal,redal,prec in zip(ths,l_tp,l_fn,l_fp,l_tn,l_sen,l_spc,l_fpr,l_tpr,l_dal,l_dal_non,l_r_dal,l_r_erup_dal,l_prec):
                    accu = (c_tp+c_tn) / (c_tp+c_fn+c_fp+c_tn)#len(df) 
                    #prec = c_tp/(c_tp+c_fp)
                    rec = c_tp/(c_tp+c_fn)
                    f.write(str(round(th,3))+','+str(round(c_tp,3))+','+str(round(c_fn,3))+','+str(round(c_fp,2))+','+str(round(c_tn,2))+','+str(round(accu,2))+','+str(round(prec,2))
                        +','+str(round(rec,2))+','+str(round(sen,2))+','+str(round(spc,2))+','+str(round(fpr,2))+','+str(round(tpr,2))+','+str(round(ndal,2))+','+str(round(ndal,2))
                        +','+str(round(rdal,2))+','+str(round(redal,2))+'\n')        
            import sys
            sys.exit() 
        if only_for_station:
            _fl = 'forecasts'+os.sep+dir_path.split(os.sep)[-1].split('_')[0]+'_'+'_'.join(data.keys())+os.sep+'perf_pars_'+only_for_station+'.csv'
        else:
            _fl = 'forecasts'+os.sep+dir_path.split(os.sep)[-1].split('_')[0]+'_'+'_'.join(data.keys())+os.sep+'perf_pars.csv'
        with open(_fl, 'w') as f:
            f.write('threshold,TP,FN,FP,TN,ACCU,PREC,REC,SEN,SPC,FPR,TPR,DAL,NDAL,RDAL,REDAL\n')
            for th,c_tp,c_fn,c_fp,c_tn,sen,spc,fpr,tpr,dal,ndal,rdal,redal,prec in zip(ths,l_tp,l_fn,l_fp,l_tn,l_sen,l_spc,l_fpr,l_tpr,l_dal,l_dal_non,l_r_dal,l_r_erup_dal,l_prec):
                accu = (c_tp+c_tn) / (c_tp+c_fn+c_fp+c_tn)#len(df) 
                #prec = c_tp/(c_tp+c_fp)
                rec = c_tp/(c_tp+c_fn)
                f.write(str(round(th,3))+','+str(round(c_tp,3))+','+str(round(c_fn,3))+','+str(round(c_fp,2))+','+str(round(c_tn,2))+','+str(round(accu,2))+','+str(round(prec,2))
                    +','+str(round(rec,2))+','+str(round(sen,2))+','+str(round(spc,2))+','+str(round(fpr,2))+','+str(round(tpr,2))+','+str(round(dal,3))+','+str(round(ndal,3))
                    +','+str(round(rdal,4))+','+str(round(redal,4))+'\n')    
        #    
    if True: # ROC curve
        plt.plot(l_fpr,l_tpr)
        for i, th in enumerate(ths):
            if th in np.linspace(0, 1, num=11)[5:]:
                plt.text(l_fpr[i], l_tpr[i]+0.01, "{:.2f}".format(th))
                plt.plot(l_fpr[i], l_tpr[i], '.k')
        #
        if True:
            _auc=[]
            for i in range(len(l_fpr)-1):
                _dx, _dy = l_fpr[i]-l_fpr[i+1], l_tpr[i]
                _auc.append([_dx*_dy])
            _auc = np.sum(np.asarray(_auc))
            plt.plot([],[],marker='o',color= 'k', label = 'AUC '+str(round(_auc,2)))#,
        #
        plt.plot([], [], 'ok', label='thresholds')
        plt.title('ROC: Receiver Operating Characteristic')
        plt.xlabel('False positive rate')
        #plt.xlim([0,0.1])
        plt.ylabel('True positive rate')#'True positive rate')
        plt.legend()
        plt.ylim([0,1.1])
        #plt.xlim([0,1])
        if only_for_station:
            plt.savefig(dir_path+os.sep+'roc_curve_'+only_for_station+'.png')
            plt.xscale('log')
            plt.savefig(dir_path+os.sep+'roc_curve_log_'+only_for_station+'.png')
        else:
            plt.savefig(dir_path+os.sep+'roc_curve.png')
            plt.xscale('log')
            plt.savefig(dir_path+os.sep+'roc_curve_log.png')
        plt.close()

    if False: # days of alert vs threshold
        plt.plot(ths,np.asarray(l_dal),label='days of alert')
        plt.plot(ths,np.asarray(l_dal_non),label='days of non-alert')
        plt.title('dal and non-dal ')
        plt.xlabel('threshold')
        plt.ylabel('days')
        plt.legend()
        #plt.show()
        plt.savefig(dir_path+os.sep+'dal_non_dal_curves.png')
        plt.close()

    if True: # Altered ROC curve
        # rate of days under alert vs rate of eruptions cougth in alert 
        _l_r_dal=l_r_dal*100
        _l_r_erup_dal=l_r_erup_dal*100
        plt.plot(_l_r_dal,_l_r_erup_dal)
        for i, th in enumerate(ths[:-1]):
            if th in np.linspace(0, 1, num=11)[5:]:
                plt.text(_l_r_dal[i], _l_r_erup_dal[i], "{:.2f}".format(th))
                plt.plot(_l_r_dal[i], _l_r_erup_dal[i], '.k')
        plt.plot([], [], 'ok', label='thresholds')
        #plt.plot([], [], '+', label='min, max (%): '+str(round(min(_l_r_dal),2))+' , '+str(round(max(_l_r_erup_dal),1)))
        plt.title('ROC2: proportion of eruptions during alerts (DAYS)')
        plt.xlabel('proportion of days under alert [%]')
        #plt.xlim([0,0.1])
        plt.ylabel('proportion of eruptions during alert [%]')#'True positive rate')
        plt.legend()
        #plt.ylim([0,1.1])
        #plt.xlim([0,100])
        if only_for_station:
            plt.savefig(dir_path+os.sep+'roc_curve_days'+only_for_station+'.png')
            plt.xscale('log')
            plt.savefig(dir_path+os.sep+'rroc_curve_days_log_'+only_for_station+'.png')
        else:
            plt.savefig(dir_path+os.sep+'roc_curve_days.png')
            plt.xscale('log')
            plt.savefig(dir_path+os.sep+'roc_curve_days_log.png')
        plt.close()
        ##

    if False: # TP and FN vs threshold
        plt.plot(ths,np.asarray(l_tp),label='true positive')
        plt.plot(ths,np.asarray(l_fn),label='false negative')
        plt.title('tp and fn ')
        plt.xlabel('threshold')
        plt.ylabel('')
        plt.legend()
        #plt.show()
        plt.savefig(dir_path+os.sep+'tp_fn_curves.png')
        plt.close()
    
    if False: # Sensitivity and Specificity vs threshold 
        plt.plot(ths,np.asarray(l_sen)*100,label='Sensitivity')
        plt.plot(ths,np.asarray(l_spc)*100,label='Specificity')
        plt.title('Sens. and Spec. vs Thresholds')
        plt.xlabel('threshold')
        plt.ylabel('%')
        plt.legend()
        # crossing point
        d_min,i_min, th_min = 1e14,0,0
        for i, th in enumerate(ths):
            _d = np.abs(l_sen[i]-l_spc[i])
            if _d < d_min:
                d_min,i_min,th_min=_d,i,th
        #print('optimal threshold: '+str(round(th_min,2)))
        #plt.show()
        plt.savefig(dir_path+os.sep+'spc_sens_curves.png')
        plt.close()

def ROC_compare_multi_cross_validation(dir_list):
    '''
    '''
    #
    if len(dir_list) > 4:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=300)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    import random
    random.seed(0)
    colors = []
    for i in range(10):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    rand=True

    # load list of fpr and tpr for every directory 
    if True: # plot roc and roc log with AUC (axs[0, 0], axs[0, 1])
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_fpr = df['FPR'].values
            l_tpr = df['TPR'].values
            #
            # ROC curve
            axs[0, 0].plot(l_fpr,l_tpr, color= col,label=dir.split(os.sep)[-1])
            axs[0, 1].plot(l_fpr,l_tpr,color= col)#, label=dir.split(os.sep)[-1])
            #for i, th in enumerate(ths):
            #    if th in np.linspace(0, 1, num=11)[5:]:
            #        plt.text(l_fpr[i], l_tpr[i]+0.01, "{:.2f}".format(th))
            #        plt.plot(l_fpr[i], l_tpr[i], '.k')   
            # AUC
            _auc=[]
            for i in range(len(l_fpr)-1):
                _dx, _dy = l_fpr[i]-l_fpr[i+1], l_tpr[i]
                _auc.append([_dx*_dy])
            _auc = np.sum(np.asarray(_auc))
            axs[0, 1].plot([],[],marker='o',color= col, label = 'AUC '+str(round(_auc,2)))#,
        
        if rand:
            dx = np.linspace(0,1,len(l_fpr)) 
            dy = np.linspace(0,1,len(l_fpr))
            axs[0, 0].plot(dx,dy, color= 'gray', linestyle='dashed',label='random',alpha=0.8)
            axs[0, 1].plot(dx,dy, color= 'gray', linestyle='dashed',alpha=0.8)  
            _auc= []        
            for i in range(len(dx)-1):
                _dx, _dy = dx[i+1]-dx[i], dy[i]
                _auc.append([_dx*_dy])
            _auc = np.sum(np.asarray(_auc)) +0.01
            axs[0, 1].plot([],[],marker='o',color= 'gray', label = 'AUC '+str(round(_auc,2)))#,    
        ##
        axs[0, 0].set_title('ROC: Receiver Operating Characteristic')
        axs[0, 0].set_xlabel('False alarms rate (false positive rate)')
        #plt.xlim([0,0.1])
        axs[0, 0].set_ylabel('Eruptions forecasted rate (true positive rate)')
        axs[0, 0].legend(loc=4)
        axs[0, 0].set_ylim([0,1.1])
        axs[0, 0].set_xlim([0,1])
        #
        axs[0, 1].set_title('ROC: Receiver Operating Characteristic')
        axs[0, 1].set_xlabel('False alarms rate (false positive rate)')
        #plt.xlim([0,0.1])
        axs[0, 1].set_ylabel('Eruptions forecasted rate (true positive rate)')
        axs[0, 1].legend(loc=4)
        axs[0, 1].set_ylim([0,1.1])
        axs[0, 1].set_xscale('log')
        #axs[0, 1].set_xlim([1e-10,1e-0])
        pass
    if False: # Accuracy
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_accu = df['ACCU'].values
            #    
            # ACCU curve
            axs[0, 1].plot(ths,l_accu, label=dir.split(os.sep)[-1])
        axs[0, 1].set_title('Accuracy')
        axs[0, 1].set_xlabel('Thresholds')
        #plt.xlim([0,0.1])
        #axs[0, 1].set_ylabel('%')
        #plt.ylim([0,1.1])
        #plt.xlim([0,1])
        #plt.savefig('forecasts'+os.sep+'accuracy_compare.png')
        #plt.close()
        axs[0, 1].legend()
    if False: # Precision
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_tp = df['TP'].values
            l_prec = df['PREC'].values
            #    
            # ACCU curve
            axs[1, 0].plot(ths,l_prec, color= col,label=dir.split(os.sep)[-1])
        axs[1, 0].set_title('Precision')
        axs[1, 0].set_xlabel('Thresholds')
        #plt.xlim([0,0.1])
        #axs[1, 0].set_ylabel('%')
        #axs[1, 0].legend()
        #axs[1, 0].set_yscale('log')
        #plt.ylim([0,1.1])
        #plt.xlim([0,1])
        #plt.savefig('forecasts'+os.sep+'precision_compare.png')
        #plt.close()
        pass
    if False: # Specificity
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_tp = df['TP'].values
            l_spc = df['SPC'].values
            #    
            # ACCU curve
            axs[1, 0].plot(ths,l_spc, color= col,label=dir.split(os.sep)[-1])
        axs[1, 0].set_title('Specificity')
        axs[1, 0].set_xlabel('Thresholds')
        #plt.xlim([0,0.1])
        axs[1, 0].set_ylabel('%')
        #axs[1, 0].legend()
        #axs[1, 0].set_yscale('log')
        #plt.ylim([0,1.1])
        #plt.xlim([0,1])
        #plt.savefig('forecasts'+os.sep+'precision_compare.png')
        #plt.close()
        pass
    if True: # Sensitivity
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_tp = df['TP'].values
            l_rec = df['REC'].values
            #    
            # ACCU curve
            plt.plot(ths,l_rec, color= col,label=dir.split(os.sep)[-1])
        axs[1, 1].set_title('Sensitivity')
        axs[1, 1].set_xlabel('Thresholds')
        #plt.xlim([0,0.1])
        axs[1, 1].set_ylabel('%')
        axs[1, 1].legend(loc=3)
        #plt.ylim([0,1.1])
        #plt.xlim([0,1])
        pass
    if False: # alt roc (proportion of days under alert)
        _max = 0
        _min = 0
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            l_rdal = df['RDAL'].values
            l_redal = df['REDAL'].values
            if max(l_redal) > _max:
                _max = max(l_redal)
            if min(l_redal) < _min:
                _min = min(l_redal)
            #
            # ROC curve
            axs[1, 0].plot(l_rdal*100,l_redal*100, color= col,label=dir.split(os.sep)[-1])
            #for i, th in enumerate(ths):
            #    if th in np.linspace(0, 1, num=11)[5:]:
            #        plt.text(l_fpr[i], l_tpr[i]+0.01, "{:.2f}".format(th))
            #        plt.plot(l_fpr[i], l_tpr[i], '.k')   
        ##
        axs[1, 0].set_title('ROC2')
        axs[1, 0].set_xlabel('proportion of days under alert [%]')
        #plt.xlim([0,0.1])
        axs[1, 0].set_ylabel('proportion of eruptions during alert [%]')
        #axs[1, 0].legend(loc=1)
        #axs[1, 0].set_ylim([_min,_max])
        #axs[1, 0].set_xlim([0,1])
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_yscale('log')
        #
        pass
    if False: # alt roc (proportion of days under alert)
        ##
        for dir,col in zip(dir_list,colors):
            #
            df = pd.read_csv(dir+os.sep+'perf_pars.csv')
            ths = df['threshold'].values
            
            l_rdal = df['RDAL'].values
            l_redal = df['REDAL'].values
            idx = (np.abs(l_redal - 0.99)).argmin()
            l_rdal=l_rdal[:idx]*100
            l_redal=l_redal[:idx]*100
            #  curve
            axs[1, 0].plot(l_rdal,l_redal, color= col)#,label=dir.split(os.sep)[-1])
            axs[1, 0].plot(l_rdal[-2], l_redal[-2], 'o', color= col, label=dir.split(os.sep)[-1]+' max: ' +str(round(max(l_redal),1))+' %')
            #for i, th in enumerate(ths):
            #    if th in np.linspace(0, 1, num=11)[5:]:
            #        plt.text(l_fpr[i], l_tpr[i]+0.01, "{:.2f}".format(th))
            #        plt.plot(l_fpr[i], l_tpr[i], '.k')   
        ##
        axs[1, 0].set_title('ROC2: proportion of eruptions during alert [%]')
        axs[1, 0].set_xlabel('proportion of days under alert [%]')
        #plt.xlim([0,0.1])
        axs[1, 0].set_ylabel('proportion of eruptions during alert [%]')
        axs[1, 0].legend(loc=1)
        #axs[1, 0].set_xscale('log')
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_ylim([0,100])
        axs[1, 0].set_xlim([-2,100])
        #
        pass
    ##
    plt.savefig('forecasts'+os.sep+'performance_compare.png')
    plt.close()

def sensitivity_dAUC_dV():
    pass

def main():

    #cross_validation_multi_volcano_leave_volcano()
    #ROC_cross_validation(dir_path='forecasts'+os.sep+'cvv_'+'_'.join([sta for sta in data.keys()])) 
    cross_validation_multi_volcano_leave_eruption()
    ROC_cross_validation(dir_path='forecasts'+os.sep+'cve_'+'_'.join([sta for sta in data.keys()]))
    #cross_validation_multi_volcano_leave_eruption_multi_station()
    #ROC_cross_validation(dir_path='forecasts'+os.sep+'cve_'+'_'.join([sta for sta in data.keys()])+'_ms')
    pass

if __name__=='__main__':
    main()
