'''
Similarity based FOSELM ensemble model
Each weak model responsible for each template wave-form
Scaled input (xScaled) is compared with template wave-form and similarity vector is calculated
models (templates) with high similarity value are used to forecast and increment learn
'''

import numpy as np
import pandas as pd
import charnonThesis as cnc
import matplotlib.pyplot as plt

##--------------Setting parameter--------------------------------------####
timeLag = 24         #number of data used for time-lag in time series
hiddenNode = 50      #number of hidden node in weak model
hiddenNodeEns = 1   #number of hidden node in ensemble model
batchSize = 1        #number of data to predict at once
runData = 24*5   #number of data used to run model 
numModel = 50        #number of weak model
threshold = 0   #minimum similarity index to be considered  
similarMetrics = 'rbf' #metrics used for calculate similarity value
limitMAPE = 100      #limit of MAPE for each model to reset
minNumModel = 1     #minimum number of model that have similarity > threshold 
##---------------------------------------------------------------------###

#----Load data set, set index, set to dateTime, sorting ------------------
rawData =  pd.read_csv('Load Dataset/AEP.csv')
rawData = rawData.set_index('Datetime')     #Set dataTime column as index
rawData.index = pd.to_datetime(rawData.index, format='%d-%m-%y %H:%M') #Change index data to datetime type 
rawData = rawData.sort_index() #sorting according to date time

#-------------------------create time-lag data----------------------------
[x,y] = cnc.buildTimeLagData(rawData['MW'][:runData+timeLag],timeLag,batchSize)


#-------divide by max of each sample-------------------------------------
scaler = np.zeros(runData)
xScaled = np.zeros([runData,timeLag])
yScaled = np.zeros([runData,1])
for j in range(runData):
    scaler[j] = x[j].max()
    xScaled[j] = x[j]/scaler[j]
    yScaled[j] = y[j]/scaler[j]
    
#---------template waveform initial by zero----------------------------------------
template = np.zeros([numModel,timeLag])

    
#---------create weak models---------------------------------------------
foselm = []
for i in range(numModel):
    foselm.append(cnc.FOSELM(hiddenNode=hiddenNode,activation='sigmoid'))
#foselm = np.array(foselm) #convert to numpy array

#------weak model initial learn from zero-------------------------------
xIni = np.zeros([1,timeLag])
yIni = np.zeros([1,batchSize])

for i in range(numModel):
    foselm[i].incrementLearn(xIni,yIni)


#------variable for store the results----------------------------------

yHat = np.empty([numModel,runData,batchSize])       #store yHat each time each weak model
yHat[:] = np.nan                                    #initialized by NaN
mapeTrend = np.empty([numModel,runData,batchSize])  #store MAPE each time each model
mapeTrend[:] = np.nan                               #initialzed by NaN
yHatEnsemble = np.zeros([runData,batchSize])        #used for store final prediction in each time
maeEnsembleTrend = np.empty([runData,batchSize])    #used for store final MAE in each time
maeEnsembleTrend[:] = np.nan                        #initialzed by NaN
mapeEnsembleTrend = np.empty([runData,batchSize])   #used for store final MAPE in each time  
mapeEnsembleTrend[:] = np.nan                       #initialzed by NaN

simVec = np.zeros([runData,template.shape[0]])   #used for store similarity index of each template (model) in each time
waveType = np.zeros([runData,1])        #used for store wavetype vaule in each time (max similarity index)

#---------loop through sample-----------------------------------------
for hour in range(runData):
    #-----0)initial template by some input------------------------------ 
    if hour < template.shape[0]: #if hour less or equal than number of template waveform  
        template[hour] = xScaled[hour] #xScaled are assigned to be template  
        for i in range(numModel):
            yHat[i,hour] = foselm[i].predict(xScaled[[hour]])*scaler[hour]
        foselm[hour].incrementLearn(xScaled[[hour]],yScaled[[hour]])#initial learn from first part data
        
    #----1)determine similarity and use weak model to predict-------- 
    else:
        #for i in range(numModel):
            #yHat[i,hour] = foselm[i].predict(xScaled[[hour]])*scaler[hour]
        
        
        
        simVec[hour] = cnc.similarityVector(template, xScaled[[hour]],metrics=similarMetrics)  
        waveType[hour] = np.argmax(simVec[hour])    
        
        if np.count_nonzero(simVec[hour][simVec[hour]>threshold]) < minNumModel:
        #have few model with high similarity use all model to forecast
            print('hour ',hour,' have few model with high similarity')
            for i in range(numModel):
                yHat[i,hour] = foselm[i].predict(xScaled[[hour]])*scaler[hour]

        else: #have more model with high similarity 
            for i in range(numModel):
                if simVec[hour,i] > threshold:#select only high similarity model
                    yHat[i,hour] = foselm[i].predict(xScaled[[hour]])*scaler[hour]
     
        
        
        #----2)ensemble result--------------------------------------
        #yHatEnsemble[hour] = np.nanmean(yHat[:,hour])    #average only non-NaN element
        yHatEnsemble[hour] = cnc.clumpMean(yHat[:,hour,:]) #use clumpMean to average only clump data (exclude NaN)


        #----3)performance assesment for weak model and ensemble model-----
        for i in range(numModel):
            mapeTrend[i,hour] = cnc.MAPE(y[hour],yHat[i,hour])
        
        mapeEnsembleTrend[hour] = cnc.MAPE(y[hour],yHatEnsemble[hour])
        maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])


        #----4)increment learn and adjust template------------------------
        for i in range(numModel):
            ff = cnc.forgettingFactor(mapeTrend[i,hour], limitMAPE)
            foselm[i].incrementLearn(xScaled[[hour]],yScaled[hour])
                                                     
#----------print and plot result------------------------------------------
print("Average MAPE = ",np.mean(mapeEnsembleTrend[numModel:]),"%")
#print("Average MAE = ",np.mean(maeEnsembleTrend[numModel:]))
#fig, ax = plt.subplots()
#graph=ax.boxplot(maeEnsembleTrend[numModel:])
#plt.ylim(-1,500)
#plt.show()
