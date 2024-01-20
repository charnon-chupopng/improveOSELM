'''
compare 
OSELM initialed by noise added data (Proposed) 
VS
OSELM initialed by real training data (baseline) 
'''


import numpy as np
import pandas as pd
import charnonThesis as cnc
import matplotlib.pyplot as plt

##--------------Setting parameter--------------------------------------####
timeLag = 24         #number of data (hour) used for time-lag in time series
hiddenNode = 50      #number of hidden node in weak model
batchSize = 1        #number of data (hour) to predict at once
runData = 8712       #number of data (hour) used to run model 
numModel = 10       #number of weak model
percentNoise = 5   #noise add to 1st data when use addNoise
iniMethod = 'addNoise'
##---------------------------------------------------------------------###

#----Load data set, set index, set to dateTime, sorting ------------------
rawData =  pd.read_csv('Load Dataset/netload_2016.csv')
rawData = rawData.set_index('time')     #Set dataTime column as index
rawData.index = pd.to_datetime(rawData.index, format='%Y-%m-%dT%H:%M:%Sz') #Change index data to datetime type 
rawData = rawData.sort_index() #sorting according to date time

#-------------------------create time-lag data----------------------------
[x,y] = cnc.buildTimeLagData(rawData['netLoad_lowDrift'][:runData+timeLag+hiddenNode],timeLag,batchSize)


#-------divide by max of each sample-------------------------------------
scaler = np.zeros(runData+hiddenNode)
xScaled = np.zeros([runData+hiddenNode,timeLag])
yScaled = np.zeros([runData+hiddenNode,batchSize])
for j in range(runData+hiddenNode):
    scaler[j] = x[j].max()
    #scaler[j] = 10000
    xScaled[j] = x[j]/scaler[j]
    yScaled[j] = y[j]/scaler[j]
    
xScaledIni = xScaled[:hiddenNode]
yScaledIni = yScaled[:hiddenNode]
xScaeldRun = xScaled[hiddenNode:]
yScaeldRun = yScaled[hiddenNode:]


#---------create weak models---------------------------------------------
baseline = []
for i in range(numModel):
    baseline.append(cnc.OSELM(hiddenNode=hiddenNode,activation='sigmoid'))

propose = []
for i in range(numModel):
    propose.append(cnc.OSELM(hiddenNode=hiddenNode,activation='sigmoid'))



#------ baseline model initial learn-------------------------------------
for i in range(numModel):
    baseline[i].initialLearn(xScaledIni,yScaledIni)
    #baseline[i].incrementLearn(xScaledIni,yScaledIni)

#------propose model initial learn---------------------------------------
if iniMethod == 'addNoise':
    for i in range(numModel):
        xIni = np.zeros([hiddenNode,timeLag])
        yIni = np.zeros([hiddenNode,batchSize])
        
        for j in range(hiddenNode):
            xIni[j] = cnc.addNoise(xScaled[hiddenNode],percentNoise=percentNoise)
            xIni[j] = xIni[j]/xIni[j].max()
            yIni[j] = yScaled[hiddenNode]
            
        propose[i].initialLearn(xIni, yIni)

elif iniMethod == 'phaseShift':
        for i in range(numModel):
            temp = np.concatenate((xScaled[hiddenNode],yScaled[hiddenNode]))
            temp = cnc.addNoise(temp,percentNoise=percentNoise)
            temp = cnc.shiftData(temp, hiddenNode)
            xIni = temp[:,:24]      
            yIni = temp[:,-1]
            propose[i].initialLearn(xIni, yIni)

#------variable for store the results----------------------------------
yHatBaseline = []
yHatPropose =[]
maeBaseline = []
maePropose = []
mapeBaseline = []
mapePropose = []
trainingErrorBaseline = []
trainingErrorPropose = []

for i in range(numModel):
    yHatBaseline.append(np.zeros([runData,batchSize]))
    yHatPropose.append(np.zeros([runData,batchSize]))
    maeBaseline.append(np.zeros([runData,batchSize]))
    maePropose.append(np.zeros([runData,batchSize]))
    mapeBaseline.append(np.zeros([runData,batchSize]))
    mapePropose.append(np.zeros([runData,batchSize]))
    trainingErrorBaseline.append(np.zeros([runData,batchSize]))
    trainingErrorPropose.append(np.zeros([runData,batchSize]))
    
yHatBaseline = np.array(yHatBaseline)
yHatPropose = np.array(yHatPropose)

maeBaseline = np.array(maeBaseline)
maePropose = np.array(maePropose)

mapeBaseline = np.array(mapeBaseline)
mapePropose = np.array(mapePropose)

trainingErrorBaseline = np.array(trainingErrorBaseline)
trainingErrorPropose = np.array(trainingErrorPropose)

yHatBaselineEnsemble = np.zeros([runData,batchSize])
yHatProposeEnsemble = np.zeros([runData,batchSize])

maeBaselineEnsemble = np.zeros([runData,batchSize])
maeProposeEnsemble = np.zeros([runData,batchSize])

mapeBaselineEnsemble = np.zeros([runData,batchSize])
mapeProposeEnsemble = np.zeros([runData,batchSize])


#---------loop through sample-----------------------------------------
for hour in range(1,runData):
    
    for i in range(numModel): #predict and increment learn for weak model
        yHatBaseline[i,hour] = baseline[i].predict(xScaled[[hour+hiddenNode]])*scaler[hour+hiddenNode]
        yHatPropose[i,hour] = propose[i].predict(xScaled[[hour+hiddenNode]])*scaler[hour+hiddenNode]

        maeBaseline[i,hour] = cnc.MAE(y[hour+hiddenNode],yHatBaseline[i,hour])
        maePropose[i,hour] = cnc.MAE(y[hour+hiddenNode],yHatPropose[i,hour])
        
        mapeBaseline[i,hour] = cnc.MAPE(y[hour+hiddenNode],yHatBaseline[i,hour])
        mapePropose[i,hour] = cnc.MAPE(y[hour+hiddenNode],yHatPropose[i,hour])
        
        baseline[i].incrementLearn(xScaled[[hour+hiddenNode]],yScaled[hour+hiddenNode])
        propose[i].incrementLearn(xScaled[[hour+hiddenNode]],yScaled[hour+hiddenNode])

        trainingErrorBaseline[i,hour] = cnc.MAPE(y[hour+hiddenNode],baseline[i].predict(xScaled[[hour+hiddenNode]])*scaler[hour+hiddenNode])
        trainingErrorPropose[i,hour]  = cnc.MAPE(y[hour+hiddenNode],propose[i].predict(xScaled[[hour+hiddenNode]])*scaler[hour+hiddenNode])
    
  
    #----------Simple average Ensemble--------------------------------
    #yHatEnsemble[hour] = np.mean(yHat[:,hour,:])    
    #maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])

    #---------ensemble by median value-----------------------------
    #yHatEnsemble[hour] = np.median(yHat[:,hour,:])    
    #maeEnsembleTrend[hour] = cnc.MAE(y[hour],yHatEnsemble[hour])


    #-----------ensemble by clump mean--------------------------
    yHatBaselineEnsemble[hour] = cnc.clumpMean(yHatBaseline[:,hour,:])
    yHatProposeEnsemble[hour] = cnc.clumpMean(yHatPropose[:,hour,:])
    
    maeBaselineEnsemble[hour] = cnc.MAE(y[hour+hiddenNode],yHatBaselineEnsemble[hour])
    maeProposeEnsemble[hour] = cnc.MAE(y[hour+hiddenNode],yHatProposeEnsemble[hour])
    
    mapeBaselineEnsemble[hour] = cnc.MAPE(y[hour+hiddenNode],yHatBaselineEnsemble[hour])
    mapeProposeEnsemble[hour] = cnc.MAPE(y[hour+hiddenNode],yHatProposeEnsemble[hour])


    
if __name__ == '__main__':
    #----------print and plot result------------------------------------------
    print("Average Baseline MAPE = ",np.mean(mapeBaselineEnsemble[1:]),'%')
    print("Average Propose MAPE = ",np.mean(mapeProposeEnsemble[1:]),'%')

   #--------line plot---------------------------------------------------------
    plt.plot(mapeBaselineEnsemble,label="baseline")
    plt.plot(mapeProposeEnsemble,label='proposed')
    plt.ylabel('MAPE[%]')
    plt.xlabel('DATA')
    plt.legend()
    plt.ylim(0,12)
    plt.show()



'''
    #---------box plot--------------------------------------------------------
    fig, ax = plt.subplots()
    graph=ax.boxplot(maeEnsembleTrend[1:])
    plt.ylim(-1,1000)
    plt.show()
'''



