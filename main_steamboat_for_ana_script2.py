from datetime import timedelta
# from puia.tests import run_tests
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


###########################################################################################################
# define data to run 
if False: # run small example #change this to False to run all
    data={'YNM':['2018-01-04','2018-06-28'],    
                }
    eruptions={'YNM':[5,4,3,2,1,0],
                }  
else: # run full 
    #data={'YNM':['2018-04-01','2020-12-31'],#,'2019-06-30'],
    #            }
    #eruptions={'YNM':[i for i in range(0,103)],#range(0,57)],
    #            } 
    data={'YNM':['2021-01-01','2023-05-30'],#,'2019-06-30'],
                 }
    eruptions={'YNM':[i for i in range(103,135)],#range(0,57)],
                 } 

# define directories
root=r'.'

DATA_DIR=f'{root}'+os.sep+'data'
FEAT_DIR=f'{root}'+os.sep+'features'
MODEL_DIR=f'{root}'+os.sep+'models'
FORECAST_DIR=f'{root}'+os.sep+'forecasts'
          
# model hyper pars
window=1.  # Length of data window in days.
overlap=0.75 # 0.75 low resolution. Change to 1.0 for high resolution. Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
look_forward= window #2. 
data_streams=['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF']
Ncl=200 #300 # Number of classifier models to train.
n_jobs=64 # CPUs to use when training classifiers in parallel.

# multi-resolution scales: list of (window_days, resample_minutes) or None for legacy
# Example: scales=[(2, 10), (14, 60), (60, 360), (180, 1440)]
scales=None

# define thresholds for performance metrics
ths = np.linspace(0, 1, num=101)#101)

def cross_validation_multi_volcano_leave_eruption():
    pass
    #define training model
    model_dir = 'models'+os.sep+'cve_'+'_'.join(data.keys())
    forecast_dir = 'forecasts'+os.sep+'cve_'+'_'.join(data.keys())
    print('model dir: '+model_dir)
    print('forecast dir: '+forecast_dir)
    
    # run training with all eruptions in the pool and forecast over the whole periods
    if False:  
        print('00')
        erup = '00'
        fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward, data_streams=data_streams, 
                feature_dir=FEAT_DIR, data_dir=DATA_DIR, 
                    root=str(erup),
                        model_dir=model_dir, 
                            forecast_dir=forecast_dir)
        drop_features=['linear_trend_timewise','agg_linear_trend']
        exclude_dates={}
        for _sta in eruptions.keys():
            exclude_dates[_sta]=None        
        #train
        fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        

        # compute forecast over whole stations period
        for _sta in data.keys():
            tf=datetimeify(data[_sta][1])#-1*_DAY
            ti=datetimeify(data[_sta][0])#+1*_DAY
            fm.hires_forecast(station=_sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs, 
                root='cve_'+'_'.join(data.keys())+os.sep+str(erup)+os.sep+str(_sta), threshold=1.0)  
    
    # loop over eruptions (stations and eruptions)
    if False:
        print(eruptions)
        for sta in data.keys():
            pass
            print(sta)
            for erup in eruptions[sta]:
                pass
                print(erup)
                fm=MultiVolcanoForecastModel(data=data, window=window, overlap=overlap, look_forward=look_forward, data_streams=data_streams, 
                        feature_dir=FEAT_DIR, data_dir=DATA_DIR, 
                            root=sta+'_'+str(erup),
                                model_dir=model_dir, 
                                    forecast_dir=forecast_dir)
                drop_features=['linear_trend_timewise','agg_linear_trend']
                #exclude data from eruption (train a model with the following data excluded)
                te1=fm.data[sta].tes[erup]
                exclude_dates={sta:[[te1-_DAY*1, te1+_DAY*1]]}
                for _sta in eruptions.keys(): # temporal, need to be fixed (exclude eruptions not considered in eruptions, but present in the .txt file)
                    if _sta != sta:
                        for _i in range(len(fm.data[_sta].tes)):
                            if _i not in eruptions[_sta]:
                                _te=fm.data[_sta].tes[_i]
                                if _sta in exclude_dates:
                                    exclude_dates[_sta].append([_te-_DAY*2, _te+_DAY*2]) 
                                else:
                                    exclude_dates[_sta]=[[_te-_DAY*2, _te+_DAY*2]] 
                for _sta in eruptions.keys():
                    if _sta not in exclude_dates:
                        exclude_dates[_sta]=None
                #train
                fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs, exclude_dates=exclude_dates)        
                #compute forecast over the eruptions excluded (for png)
                tf=te1+_DAY*window#/12#_DAY*2*.1
                ti=te1-_DAY*5 #*window
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+os.sep+sta+'_'+str(erup),
                        save='forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+'_fc_eruption_'+sta+'_'+str(erup)+'.png') # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+
                #compute forecast over the eruptions excluded (for master consensus)
                tf=te1+_DAY*window
                ti=te1-_DAY*window
                fm.hires_forecast(station=sta, ti=ti, tf=tf, recalculate=False, n_jobs=n_jobs,  threshold=1., 
                    root='cve_'+'_'.join(data.keys())+os.sep+sta+'_'+str(erup)) # '.'+os.sep+'forecasts'+os.sep+'cve_'+'_'.join(data.keys())+os.sep+
                    
        # save txt with cross validation info 
        if True:
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
            print(sta)
            
            #Check all consensus files and concatenate (file are per year of record) 
            _path = forecast_dir+os.sep+'00'+os.sep+sta
            _con = [x[2] for x in os.walk(_path)][0]

            _con = [x for x in _con if 'consensus' in x]
            _consensus_master = pd.concat([pd.read_pickle(_path+os.sep+x) for x in _con])#pd.read_pickle(_path+os.sep+'consensus_2018.pkl') #pd.concat([pd.read_pickle(_path+os.sep+x) for x in _con])
            _consensus_master.sort_index(inplace=True)

            # (2) for each eruption in out of sample models, look for consensus during the eruptive period and replace on master consensus
            for erup in eruptions[sta]:
                try:
                    _path = forecast_dir+os.sep+sta+'_'+str(erup)
                    _consensus_erup = pd.read_pickle(glob(_path+os.sep+'*consensus*.pkl')[0])
                    # drop rows in _consensus_master dor the range in _consensus_erup
                    l1 = _consensus_master.index.get_loc(_consensus_erup.index.tolist()[0], method='nearest')
                    l2 = _consensus_master.index.get_loc(_consensus_erup.index.tolist()[-1], method='nearest')                
                    _consensus_master=_consensus_master.drop(_consensus_master.index[list(range(l1,l2+1))]) #range is not inclusive, need this plus one to cut out end date too
                    # _consensus_master[list(range(l1,l2,1))]
                    # concat _consensus_master with _consensus_erup and sort
                    _consensus_master = pd.concat([_consensus_master,_consensus_erup])
                    _consensus_master.sort_index(inplace=True)                
                except:
                    pass
            # (3) save master consensus for each station in main folder 
            _consensus_master.to_pickle(forecast_dir+os.sep+'_consensus_master_'+sta+'.pkl')
            pass
        _consensus_master.plot()

def plot_consensus(sta = 'YNM'):
    # Load consensus data
    consensus_data = pd.read_pickle(FORECAST_DIR + os.sep  + 'cve_' + sta + os.sep + '_consensus_master_' + sta + '.pkl')

    # Load eruptive periods dates
    eruptive_periods = pd.read_csv(DATA_DIR + os.sep + sta + '_eruptive_periods.txt', sep=' ', header=None, names=['year', 'month', 'day', 'hour', 'minute', 'second'])

    # Convert eruptive periods to datetime
    eruptive_periods['date'] = pd.to_datetime(eruptive_periods[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Filter eruptive periods before the last date in consensus data
    eruptive_periods = eruptive_periods[eruptive_periods['date'] < consensus_data.index[-1]]

    if False:
        # Divide the consensus data into three parts
        consensus_length = len(consensus_data)
        part_size = consensus_length // 3
        part1 = consensus_data.iloc[:part_size]
        part2 = consensus_data.iloc[part_size:2*part_size]
        part3 = consensus_data.iloc[2*part_size:]

        # Plot whole consensus with rolling mean
        plt.figure(figsize=(8, 10))

        # Plotting part 1
        plt.subplot(3, 1, 1)
        plt.plot(part1['consensus'], label='Consensus')
        plt.plot(part1['consensus'].rolling(window='.75D').quantile(.9), label='Rolling Median (.75 Day)')
        
        plt.xlabel('Time')
        plt.ylabel('Consensus')
        plt.title('Consensus')
        plt.xticks(rotation=25)
        for date in eruptive_periods['date']:
            if part1.index[0] <= date <= part1.index[-1]:  # Only plot eruptive periods within the time range of part 1
                plt.axvline(x=date, color='r', linestyle='--')
        plt.axvline(x=eruptive_periods['date'][0], color='r', linestyle='--', label='Eruption')
        plt.legend(loc=2)

        # Plotting part 2
        plt.subplot(3, 1, 2)
        plt.plot(part2['consensus'], label='Consensus')
        plt.plot(part2['consensus'].rolling(window='.75D').quantile(.9), label='Rolling Median (.75 Day)')
        plt.xlabel('Time')
        plt.ylabel('Consensus')
        #plt.title('Consensus with Rolling Mean (Part 2)')
        #plt.legend(loc=2)
        plt.xticks(rotation=45)
        for date in eruptive_periods['date']:
            if part2.index[0] <= date <= part2.index[-1]:  # Only plot eruptive periods within the time range of part 2
                plt.axvline(x=date, color='r', linestyle='--')

        # Plotting part 3
        plt.subplot(3, 1, 3)
        plt.plot(part3['consensus'], label='Consensus')
        plt.plot(part3['consensus'].rolling(window='.75D').quantile(.6), label='Rolling Median (.75 Day)')
        plt.ylabel('Consensus')
        #plt.title('Consensus with Rolling Mean (Part 3)')
        #plt.legend(loc=2)
        plt.xticks(rotation=45)
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:  # Only plot eruptive periods within the time range of part 3
                plt.axvline(x=date, color='r', linestyle='--')
        plt.axvline(x=date, color='r', linestyle='--', label='Eruption')
        plt.legend(loc=3)
        plt.title('Consensus (out of sample)')
        plt.tight_layout()
        plt.savefig(FORECAST_DIR + os.sep + 'cve_' + sta + os.sep + '_consensus_plot.png')
        plt.show()
        plt.close()

    # Plot whole consensus with rolling mean
    if False:
        plt.figure(figsize=(18, 5))
        plt.plot(consensus_data['consensus'], label='Consensus')
        plt.plot(consensus_data['consensus'].rolling(window='.75D').mean(), label='Rolling Mean (.75 Day)')
        plt.xlabel('Time')
        plt.ylabel('Consensus')
        plt.title('Consensus with Rolling Mean')
        #plt.legend(loc=2)
        plt.xticks(rotation=45)
        for date in eruptive_periods['date']:
            plt.axvline(x=date, color='r', linestyle='--')
        plt.show()
        asdf
        plt.savefig(FORECAST_DIR + os.sep + 'cve_' + sta + os.sep + '_consensus_plot.png')
        plt.close()

    # Plot individual eruptive events
    if False:
        for idx, date in enumerate(eruptive_periods['date']):
            start_date = date - pd.Timedelta(days=4)
            end_date = date + pd.Timedelta(days=0.5)
            event_data = consensus_data[(consensus_data.index >= start_date) & (consensus_data.index <= end_date)]
            
            plt.figure(figsize=(10, 6))
            plt.plot(event_data['consensus'], label='Consensus')
            plt.xlabel('Time')
            plt.ylabel('Consensus')
            plt.title(f'Eruptive Event {idx+1}')
            plt.legend()
            plt.xticks(rotation=45)
            for event_date in eruptive_periods['date']:
                if start_date <= event_date <= end_date:
                    plt.axvline(x=event_date, color='g', linestyle='-.')  # Plotting all events within the period
            plt.savefig(FORECAST_DIR + os.sep +  'cve_' + sta + os.sep + f'{sta}_event_{idx+1}_plot.png')
            plt.close()

    if True: # Plot individual eruptive events
        # Load consensus data
        consensus_data = pd.read_pickle(FORECAST_DIR + os.sep  + 'cve_' + sta + os.sep + '_consensus_master_' + sta + '.pkl')

        # Load consensus data
        consensus_data = pd.read_pickle(FORECAST_DIR + os.sep + 'cve_' + sta + os.sep + '_consensus_master_' + sta + '.pkl')

        # Load eruptive periods dates
        eruptive_periods = pd.read_csv(DATA_DIR + os.sep + sta + '_eruptive_periods.txt', sep=' ', header=None, names=['year', 'month', 'day', 'hour', 'minute', 'second'])

        # Convert eruptive periods to datetime
        eruptive_periods['date'] = pd.to_datetime(eruptive_periods[['year', 'month', 'day', 'hour', 'minute', 'second']])

        # Filter eruptive periods before the last date in consensus data
        eruptive_periods = eruptive_periods[eruptive_periods['date'] < consensus_data.index[-1]]

        # Get the last 12 eruptive events
        last_12_events = eruptive_periods.iloc[-13:]

        # Plot the last 12 eruptive events as subplots
        fig, axes = plt.subplots(7, 2, figsize=(8, 12))
        axes = axes.flatten()  # Flatten axes array to simplify indexing

        for i, (index, event) in enumerate(last_12_events.iterrows()):
            start_date = event['date'] - pd.Timedelta(days=2)
            end_date = event['date'] + pd.Timedelta(days=1)
            event_data = consensus_data.loc[start_date:end_date]

            ax = axes[i]
            ax.plot(event_data.index, event_data['consensus'], label='Consensus')
            ax.axvline(x=event['date'], color='r', linestyle='--', label='Event')
            ax.axhline(y=.6, color='gray', linestyle='--', label='ref. threshold')
            #ax.set_xlabel('Time')
            ax.set_ylabel('Consensus')
            ax.set_ylim([.1,.99])
            date1 = start_date
            date2=end_date
            ax.set_xticks([date1,date2])
            ax.set_xticklabels([date1.strftime('%Y-%m-%d'), date2.strftime('%Y-%m-%d')])

            #ax.set_title(f'Event {i+1}')
            ax.set_title(f'Event Date: {event["date"].strftime("%Y-%m-%d")}')
            if i ==1:
                ax.legend(loc=1)

            # Plot transparent vertical section between the event and 0.75 days before the event
            start_section = event['date'] - pd.Timedelta(hours=18)  # 0.75 days before the event
            end_section = event['date']
            ax.axvspan(start_section, end_section, color='orange', alpha=0.2)

            plt.tight_layout()
            #plt.savefig(FORECAST_DIR + os.sep + 'cve_' + sta + os.sep + 'consensus_around_last_12_events.png')
            plt.show()
            plt.close()

def ROC_cross_validation(dir_path):
    '''
    Performs ROC cross-validation on the provided directory path.
    '''
    window = 0.75
    rand = False  # Generate a randomized master consensus
    print('ROC curve on ' + dir_path)

    # Initialize lists for storing results
    l_fpr, l_tpr, l_sen, l_spc, l_tp, l_fn, l_fp, l_tn, l_prec = [], [], [], [], [], [], [], [], []
    l_dal_non, l_dal = [], []

    # Loop over thresholds
    for th in ths:
        c_tp, c_fn, c_tn, c_fp = 0, 0, 0, 0
        c_dal, c_dal_non = 0, 0

        # Loop over stations
        for sta in data.keys():
            _consensus = pd.read_pickle(dir_path + os.sep + '_consensus_master_' + sta + '.pkl')

            # Filter period to the last third of the data
            #consensus_length = len(_consensus)
            #part_size = consensus_length // 3
            #_consensus = _consensus.iloc[2 * part_size:]

            if rand:  # Randomize (shuffle) consensus values
                print('Running randomized')
                import random
                _aux = _consensus.iloc[:, 0].values
                random.shuffle(_aux)
                _consensus['_consensus_rand'] = _aux
                _consensus = _consensus.drop('_consensus_rand', axis=1)

            # Get eruption times
            fl_nm = DATA_DIR + os.sep + sta + '_eruptive_periods.txt'
            with open(fl_nm, 'r') as fp:
                tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            tes = [date for date in tes if _consensus.index[0] <= date <= _consensus.index[-1]]

            # Count TP and FN in the eruptive record
            for k, te in enumerate(tes):
                if k in eruptions[sta]:
                    inds = (_consensus.index < te - window * _DAY) | (_consensus.index >= te)
                    window_data = _consensus.loc[~inds]

                    threshold_exceeded = False
                    for idx, row in window_data.iterrows():
                        if row['consensus'] >= th:
                            threshold_exceeded = True
                            tp_start_idx = window_data.index.get_loc(idx)
                            break

                    if threshold_exceeded:
                        fn_window = window_data.iloc[:tp_start_idx]
                        tp_window = window_data.iloc[tp_start_idx:]
                        c_tp += len(tp_window)
                        c_fn += len(fn_window[fn_window['consensus'] < th])
                    else:
                        c_fn += len(window_data)

            # Count TN and FP in the non-eruptive record
            _idx_bool = _consensus['consensus'] < th
            c_tn += len(_consensus[_idx_bool])
            c_fp += len(_consensus[~_idx_bool])

            # Count days of alert (and non-alert)
            _consensus_aux = _consensus.resample('2D').quantile(q=0.95)
            _idx_bool = _consensus_aux['consensus'] < th
            c_dal_non += len(_consensus_aux[_idx_bool]) * 2
            c_dal += len(_consensus_aux[~_idx_bool]) * 2

        # Compute FPR, TPR, SENS, and SPEC
        tpr = c_tp / (c_tp + c_fn)
        fpr = c_fp / (c_fp + c_tn)
        sen = c_tp / (c_tp + c_fn)
        spc = c_tn / (c_tn + c_fp)
        try:
            prec = c_tp / (c_tp + c_fp)
        except ZeroDivisionError:
            prec = 1.0

        # Append results to lists
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

    # Rate of days under alert vs. rate of eruptions caught in alert
    l_r_dal = np.asarray(l_dal) / (np.asarray(l_dal) + np.asarray(l_dal_non))  # x-axis
    l_r_erup_dal = np.asarray(l_tp) / (np.asarray(l_tp) + np.asarray(l_fn))  # y-axis
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
        with open('forecasts'+os.sep+dir_path.split(os.sep)[-1].split('_')[0]+'_'+'_'.join(data.keys())+os.sep+'perf_pars.csv', 'w') as f:
            f.write('threshold,TP,FN,FP,TN,ACCU,PREC,REC,SEN,SPC,FPR,TPR,DAL,NDAL,RDAL,REDAL\n')
            for th,c_tp,c_fn,c_fp,c_tn,sen,spc,fpr,tpr,dal,ndal,rdal,redal,prec in zip(ths,l_tp,l_fn,l_fp,l_tn,l_sen,l_spc,l_fpr,l_tpr,l_dal,l_dal_non,l_r_dal,l_r_erup_dal,l_prec):
                accu = (c_tp+c_tn) / (c_tp+c_fn+c_fp+c_tn)#len(df) 
                #prec = c_tp/(c_tp+c_fp)
                rec = c_tp/(c_tp+c_fn)
                f.write(str(round(th,3))+','+str(round(c_tp,3))+','+str(round(c_fn,3))+','+str(round(c_fp,2))+','+str(round(c_tn,2))+','+str(round(accu,2))+','+str(round(prec,2))
                    +','+str(round(rec,2))+','+str(round(sen,2))+','+str(round(spc,2))+','+str(round(fpr,2))+','+str(round(tpr,2))+','+str(round(dal,3))+','+str(round(ndal,3))
                    +','+str(round(rdal,4))+','+str(round(redal,4))+'\n')    
    
    if True: # ROC curve
        plt.figure(figsize=(4, 3)) 
        plt.plot(l_fpr,l_tpr, 'b', label = 'RF model')
        for i, th in enumerate(ths):
            if th == .6:#in np.linspace(0, 1, num=11)[5:]:
                plt.text(l_fpr[i], l_tpr[i]+0.01, "{:.2f}".format(th))
                plt.plot(l_fpr[i], l_tpr[i], '.k')
        # AUC
        #_auc=[]
        #for i in range(len(l_fpr)-1):
        #    _dx, _dy = l_fpr[i]-l_fpr[i+1], l_tpr[i]
        #    _auc.append([_dx*_dy])
        #_auc = np.sum(np.asarray(_auc))
        _auc = -1*np.trapz(l_tpr, l_fpr)
        # ROC curve
        plt.plot([],[], color= 'w', linestyle='-', linewidth=2, label='('+'AUC:'+str(round(_auc,2))+') ', zorder = 5)
            #
        plt.plot([], [], 'ok', label='thresholds')
        plt.plot([0, 1], [0, 1], 'k--', label='random')
        plt.title('ROC: Receiver Operating Characteristic')
        plt.xlabel('False positive rate')
        #plt.xlim([0,0.1])
        plt.ylabel('True positive rate')#'True positive rate')
        plt.legend()
        plt.ylim([0,1.1])
        plt.tight_layout()
        plt.show()
        #plt.xlim([0,1])
        plt.savefig(dir_path+os.sep+'roc_curve.png')
        plt.xscale('log')
        plt.show()   
        #plt.savefig(dir_path+os.sep+'roc_curve_log.png')
        plt.close()

    if True: # days of alert vs threshold
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
        #plt.xlim([0,1])
        plt.savefig(dir_path+os.sep+'roc_curve_days.png')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim([0,1])
        #plt.ylim([0,100])
        #plt.show()   
        plt.savefig(dir_path+os.sep+'roc_curve_log_days.png')
        plt.close()
        ##

    if True: # TP and FN vs threshold
        plt.plot(ths,np.asarray(l_tp),label='true positive')
        plt.plot(ths,np.asarray(l_fn),label='false negative')
        plt.title('tp and fn ')
        plt.xlabel('threshold')
        plt.ylabel('')
        plt.legend()
        #plt.show()
        plt.savefig(dir_path+os.sep+'tp_fn_curves.png')
        plt.close()
    
    if True: # Sensitivity and Specificity vs threshold 
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

def plot_features(sta = 'YNM'):

    # Load consensus data
    consensus_data = pd.read_pickle(FORECAST_DIR + os.sep  + 'cve_' + sta + os.sep + '_consensus_master_' + sta + '.pkl')

    # Load eruptive periods dates
    eruptive_periods = pd.read_csv(DATA_DIR + os.sep + sta + '_eruptive_periods.txt', sep=' ', header=None, names=['year', 'month', 'day', 'hour', 'minute', 'second'])

    # Convert eruptive periods to datetime
    eruptive_periods['date'] = pd.to_datetime(eruptive_periods[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Filter eruptive periods before the last date in consensus data
    eruptive_periods = eruptive_periods[eruptive_periods['date'] < consensus_data.index[-1]]

    if True: # whole period
        # Load the pickle file containing the time series data
        time_series_data = pd.read_pickle(FEAT_DIR + os.sep + 'fm_0.75w_zsc2_mfF_YNM_2018.pkl')

        # Divide the time series data into three parts
        time_series_length = len(time_series_data)
        part_size = time_series_length // 3
        part1 = time_series_data.iloc[:part_size]
        part2 = time_series_data.iloc[part_size:2*part_size]
        part3 = time_series_data.iloc[2*part_size:]
        if False:
            time_series_length = len(part3)
            part_size = time_series_length // 5
            part3 = time_series_data.iloc[3*part_size:5*part_size]
        # Plot the time series parts
        plt.figure(figsize=(10, 10))

        # Plotting part 2
        plt.subplot(5, 1, 1)
        plt.plot(part3['zsc2_mfF__autocorrelation__lag_7'], label='zsc2_mfF__autocorrelation__lag_7')
        #plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('MF autocorrelation 7')
        #plt.xticks(rotation=25)
        # Plot vertical lines for eruptive periods within the time range of part 2
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:
                plt.axvline(x=date, color='r', linestyle='--')
        plt.legend(loc = 3)

        # Plotting part 3
        plt.subplot(5, 1, 2)
        plt.plot(part3['zsc2_mfF__longest_strike_above_mean'], label='zsc2_mfF__longest_strike_above_mean')
        #plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('MF longest strike above mean')
        #plt.xticks(rotation=25)
        # Plot vertical lines for eruptive periods within the time range of part 3
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:
                plt.axvline(x=date, color='r', linestyle='--')
        plt.legend(loc = 3)


        # Plotting part1
        plt.subplot(5, 1, 3)
        plt.plot(np.abs(part3['zsc2_mfF__agg_autocorrelation__f_agg_"median"__maxlag_40']), label='zsc2_mfF__agg_autocorrelation__f_agg_"median"__maxlag_40')
        #plt.xlabel('Time')
        plt.ylabel('Value')
        #plt.yscale('log')
        plt.title('MF agg autocorrelation median')
        #plt.xticks(rotation=25)
        # Plot vertical lines for eruptive periods within the time range of part 1
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:
                plt.axvline(x=date, color='r', linestyle='--')
        plt.legend(loc = 3)


        # Plotting part 3
        plt.subplot(5, 1, 4)
        plt.plot(part3['zsc2_mfF__autocorrelation__lag_3'], label='zsc2_mfF__autocorrelation__lag_3')
        #plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('MF autocorrelation 3')
        #plt.xticks(rotation=25)
        # Plot vertical lines for eruptive periods within the time range of part 3
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:
                plt.axvline(x=date, color='r', linestyle='--')
        plt.legend(loc = 3)

        # Plotting part 3
        plt.subplot(5, 1, 5)
        plt.plot(part3['zsc2_mfF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0'], label='zsc2_mfF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0')
        #plt.xlabel('Time')
        plt.ylabel('Value')
        plt.yscale('log')
        plt.title('MF change quantiles 0.-.2')
        #plt.xticks(rotation=25)
        # Plot vertical lines for eruptive periods within the time range of part 3
        for date in eruptive_periods['date']:
            if part3.index[0] <= date <= part3.index[-1]:
                plt.axvline(x=date, color='r', linestyle='--')
        plt.legend(loc = 3)

        plt.tight_layout()
        #plt.savefig(FORECAST_DIR + os.sep + 'cve_' + sta + os.sep + 'time_series_plot.png')
        plt.show()
        plt.close()
        asdfd


def main():
    # forecast_test()
    cross_validation_multi_volcano_leave_eruption()
    ROC_cross_validation(dir_path='forecasts'+os.sep+'cve_'+'_'.join([sta for sta in data.keys()]))
    #plot_consensus()
    #plot_features()

    pass

if __name__=='__main__':
    main()
