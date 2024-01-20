'''
Similarity based FOSELM ensemble model
like V1 but use ensemble model instead of one model
for resposible for each template wave-from
Scaled input (xScaled) is compared with template wave-form and similarity vector is calculated
ensemble models (templates) with high similarity value are used to forecast and increment learn
'''

import numpy as np
import pandas as pd
import charnonThesis as cnc
import matplotlib.pyplot as plt

##--------------Setting parameter--------------------------------------####
timeLag = 24         #number of data used for time-lag in time series
hiddenNode = 50      #number of hidden node in weak model
hiddenNodeEns = 10   #number of hidden node in ensemble model
batchSize = 1        #number of data to predict at once
runData = 24*300    #number of data used to run model 
numModel = 1        #number of weak model
numTemplate = 50     #number of template used to compared
threshold = 1      #minimum similarity index to be considered  
similarMetrics = 'rbf' #metrics used for calculate similarity value
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
template = np.zeros([numTemplate,timeLag]) #record template waveform 

    
#---------create weak models---------------------------------------------
foselm = []
for j in range(numTemplate):
    for i in range(numModel):
        foselm.append(cnc.FOSELM(hiddenNode=hiddenNode,activation='sigmoid'))
foselm = np.array(foselm) #convert to numpy array of object
foselm = foselm.reshape(numTemplate,-1) #make 2D array of object



#------weak model initial learn from zero-------------------------------
xIni = np.zeros([1,timeLag])
yIni = np.zeros([1,batchSize])

for i in range(numTemplate):
    for j in range(numModel):
        foselm[i,j].incrementLearn(xIni,yIni)


#------variable for store the results----------------------------------
yHat = []
mapeTrend = []
for i in range(numModel):
    yHat.append(np.zeros([runData,batchSize]))
    mapeTrend.append(np.zeros([runData,batchSize]))    
yHat = np.array(yHat)           #used for store yHat in each time of each model
mapeTrend = np.array(mapeTrend)   #used for store MAE in each time of each model

yHatEnsemble = np.zeros([runData,batchSize])     #used for store final prediction in each time
maeEnsembleTrend = np.zeros([runData,batchSize]) #used for store final MAE in each time
simVec = np.zeros([runData,template.shape[0]])   #used for store similarity index of each template (model) in each time
waveType = np.zeros([runData,1])        #used for store wavetype vaule in each time (max similarity index)

#---------loop through sample-----------------------------------------
for hour in range(runData):
    
    #-----initial template by some input----------------------------# 
    if hour < template.shape[0]: #if hour less or equal than number of template waveform  
        template[hour] = xScaled[hour] #xScaled are assigned to be template  
        for i in range(numModel):
            foselm[hour,i].incrementLearn(xScaled[[hour]],yScaled[hour])#initial learn from first part data
    #---------------------------------------------------------------#
    
    else:#after initial, update template and make prediction
        simVec[hour] = cnc.similarityVector(template, xScaled[[hour]],metrics=similarMetrics)  
        waveType[hour] = np.argmax(simVec[hour])    
        template[np.argmax(simVec[hour])] = xScaled[[hour]] #update template
        

    #---select one ensemble model with highest simVec value for predict and learn
        for i in range(numModel): 
            yHat[i,hour] = foselm[int(waveType[hour]),i].predict(xScaled[[hour]])*scaler[hour]
            foselm[int(waveType[hour]),i].incrementLearn(xScaled[[hour]],yScaled[[hour]])
 

    yHatEnsemble[hour] = np.mean(yHat[:,hour])
    
    #for j in range(numTemplate):
     #   for i in range(numModel):
      #      foselm[j,i].incrementLearn(xScaled[[hour]],yScaled[[hour]])




#----------print and plot result------------------------------------------
print("MAPE of Ensemble model = ",cnc.MAPE(y[template.shape[0]:],yHatEnsemble[template.shape[0]:]))
plt.plot(y[template.shape[0]:])
plt.plot(yHatEnsemble[template.shape[0]:])
plt.show()