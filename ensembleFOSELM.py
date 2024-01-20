'''
ensemble and Stack of FOSELM Model

'''

import numpy as np
import pandas as pd
import charnonThesis as cnc
import matplotlib.pyplot as plt

##--------------Setting parameter--------------------------------------####
timeLag = 24         #number of data used for time-lag in time series
hiddenNode = 50      #number of hidden node in weak model
batchSize = 1        #number of data to predict at once
runData = 24*30        #number of data used to run model 
numModel = 10         #number of weak model
ensembleFunction = 'simpleMean' #how to ensemble output of each weak model
firstSample = 50    #first sample in dataset used to run 

if (ensembleFunction == 'rls'):
    rls = cnc.rlsEnsemble(numModel, batchSize)


##---------------------------------------------------------------------###

#----Load data set, set index, set to dateTime, sorting ------------------
rawData =  pd.read_csv('Load Dataset/NI.csv')
rawData = rawData.set_index('Datetime')     #Set dataTime column as index
rawData.index = pd.to_datetime(rawData.index, format='%d-%m-%y %H:%M') #Change index data to datetime type 
rawData = rawData.sort_index() #sorting according to date time

#-------------------------create time-lag data----------------------------
#[x,y] = cnc.buildTimeLagData(rawData['MW'][:runData+timeLag],timeLag,batchSize)
[x,y] = cnc.buildTimeLagData(rawData['MW'][firstSample:firstSample+runData+timeLag],timeLag,batchSize)


#-------divide by max of each sample-------------------------------------
scaler = np.zeros(runData)
xScaled = np.zeros([runData,timeLag])
yScaled = np.zeros([runData,1])
for j in range(runData):
    scaler[j] = x[j].max()
    xScaled[j] = x[j]/scaler[j]
    yScaled[j] = y[j]/scaler[j]
    
#---------load template waveform----------------------------------------
#template = pd.read_csv('template.csv',header=None)
#template = np.array(template)
template = np.zeros([numModel,timeLag])

    
#---------create weak models---------------------------------------------
foselm = []
for i in range(numModel):
    foselm.append(cnc.FOSELM(hiddenNode=hiddenNode,activation='sigmoid'))


#--------create ensemble model-------------------------------------------
#foselm_ens = cnc.FOSELM(hiddenNode=hiddenNodeEns,activation='purelin')


#------weak model initial learn from zero or 1st data-------------------------------
#xIni = np.zeros([1,timeLag])
xIni = xScaled[[0]]
#yIni = np.zeros([1,batchSize])
yIni = yScaled[[0]]

for i in range(numModel):
    foselm[i].incrementLearn(xIni,yIni)
    

#-----ensemble model initial learn from zero---------------------------
#if similarity not used
#foselm_ens.incrementLearn(np.zeros([1,numModel*batchSize]),np.zeros([1,batchSize]))

#if similarity vector used
#foselm_ens.incrementLearn(np.zeros([1,numModel*batchSize+template.shape[0]]),np.zeros([1,batchSize]))


#------variable for store the results----------------------------------
yHat = []
maeTrend = []
mapeTrend = []
trainingError = []

for i in range(numModel):
    yHat.append(np.zeros([runData,batchSize]))
    maeTrend.append(np.zeros([runData,batchSize]))
    mapeTrend.append(np.zeros([runData,batchSize]))
    trainingError.append(np.zeros([runData,batchSize]))
    
yHat = np.array(yHat)
maeTrend = np.array(maeTrend)
mapeTrend = np.array(mapeTrend)
trainingError = np.array(trainingError)
yHatEnsemble = np.zeros([runData,batchSize])
maeEnsembleTrend = np.zeros([runData,batchSize])
mapeEnsembleTrend = np.zeros([runData,batchSize])
simVec = np.zeros([runData,template.shape[0]])
waveType = np.zeros([runData,1])

#---------loop through sample-----------------------------------------
for hour in range(1,runData):
    
    for i in range(numModel): #predict and increment learn for weak model
        yHat[i,hour] = foselm[i].predict(xScaled[[hour]])*scaler[hour]
        maeTrend[i,hour] = cnc.MAE(y[hour],yHat[i,hour])
        mapeTrend[i,hour] = cnc.MAPE(y[hour],yHat[i,hour])
        foselm[i].incrementLearn(xScaled[[hour]],yScaled[hour])
        #foselm[i].incrementLearn(xScaled[[hour]],yScaled[hour])
        #foselm[i].incrementLearn(xScaled[[hour]],yScaled[hour])
        trainingError[i,hour] = cnc.MAPE(y[hour],foselm[i].predict(xScaled[[hour]])*scaler[hour])

                
    '''
    if(hour%5==0):
        plt.plot(foselm[0].beta)
        plt.title(f'FOSELM {hour} data')
        plt.show()
    '''
    
    if(ensembleFunction=='simpleMean'):
        #----------Simple average Ensemble--------------------------------
        yHatEnsemble[hour] = np.mean(yHat[:,hour,:])    
        maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])
        mapeEnsembleTrend[hour] = cnc.MAPE(y[hour],yHatEnsemble[hour])
    
    elif(ensembleFunction=='median'):    
    #---------ensemble by median value-----------------------------
        yHatEnsemble[hour] = np.median(yHat[:,hour,:])    
        maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])
        mapeEnsembleTrend[hour] = cnc.MAPE(y[hour],yHatEnsemble[hour])

    elif(ensembleFunction=='clumpMean'):
    #-----------ensemble by clump mean--------------------------
        yHatEnsemble[hour] = cnc.clumpMean(yHat[:,hour,:])
        maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])
        mapeEnsembleTrend[hour] = cnc.MAPE(y[hour],yHatEnsemble[hour])

    elif(ensembleFunction == 'rls'):    
        yHatEnsemble[hour] = rls.predict(yHat[:,hour,:].T)
        maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])        
        mapeEnsembleTrend[hour] = cnc.MAPE(y[hour],yHatEnsemble[hour])
        rls.learn(yHat[:,hour,:].T, y[[hour]],forget=1)
        '''
        plt.plot(rls.W)
        plt.ylim(-2,2)
        plt.title(hour)
        plt.show()
        '''
    
if __name__ == '__main__':
    #----------print and plot result------------------------------------------
    print("Average MAPE = ",np.mean(mapeEnsembleTrend[1:]),'%')
    print("Average MAE = ",np.mean(maeEnsembleTrend[1:]))


    #for i in range(numModel):
        #   print("MAE of Model ",i, " = ",np.mean(maeTrend[i,1:,:]))

'''
    #---------box plot--------------------------------------------------------
    fig, ax = plt.subplots()
    graph=ax.boxplot(mapeEnsembleTrend[numModel:])
    plt.ylim(-1,10)
    plt.show()
'''

