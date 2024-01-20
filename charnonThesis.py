import numpy as np
       
if __name__  == '__main__':
    print('this is charnonThesis\'s module')
else:
    print('### Load charnonThesis Module ###')

class ELM:
    def __init__(self,hiddenNode=100,activation='sigmoid'):
        self.hiddenNode = hiddenNode
        self.activation = activation
        
    def relu(self,x): # Rectified Linear Unit (Relu) activation function
        return np.maximum(0,x)
    
    def pureLin(self,x):
        return x
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    
    def learn(self,x,y):
        #each row of data represent each instance(record) of data      
        if x.shape[0]<self.hiddenNode:
            print("Cant learn, sample less than hidden node")
            return None
        
        featureNum = x.shape[1]
        self.w = np.random.uniform(-1,1,[self.hiddenNode,featureNum]) #random weight range 0-1 
        
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T))
        
        elif self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T))
            
        elif self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T))
                    
        elif self.activation=='tanh':
            H = self.tanh(np.dot(x,self.w.T))
        
        else: 
            print("unknow activation function")
        
        self.H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        Hinv = np.linalg.pinv(self.H)
        self.beta = np.dot(Hinv,y)

        
    def predict(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T))
                    
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T))
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T))
            
        if self.activation=='tanh':
            H = self.tanh(np.dot(x,self.w.T))
     
        H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        return np.dot(H,self.beta)            
        
        

class  AE_ELM:
    def __init__(self,hiddenNode=100,activation='sigmoid'):
        self.hiddenNode = hiddenNode
        self.activation = activation
        self.ae = self.autoencoder(hiddenNode)
        
    def relu(self,x): # Rectified Linear Unit (Relu) activation function
        return np.maximum(0,x)
    
    def pureLin(self,x):
        return x
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def learn(self,x,y):
        #each row of data represent each instance(record) of data      
        if x.shape[0]<self.hiddenNode:
            print("Cant learn, sample less than hidden node")
            return None
        
        self.w = self.ae.calWeight(x) 
        
        if self.activation=='relu':
            self.H = self.relu(np.dot(x,self.w.T))
        
        elif self.activation=='sigmoid':   
            self.H = self.sigmoid(np.dot(x,self.w.T))
            
        elif self.activation=='pureLin':
            self.H = self.pureLin(np.dot(x,self.w.T))
                    
        elif self.activation=='tanh':
            self.H = self.tanh(np.dot(x,self.w.T))
        
        else: 
            print("unknow activation function")
        
        Hinv = np.linalg.pinv(self.H)
        self.beta = np.dot(Hinv,y)
        
    def predict(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T))
                    
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T))
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T))
            
        if self.activation=='tanh':
            H = self.tanh(np.dot(x,self.w.T))
     
        return np.dot(H,self.beta)     
        
    class autoencoder:
      def __init__(self,hiddenNode):
            self.hiddenNode = hiddenNode
            
      def randomOrthonormal(self,m,n):
         H = np.random.rand(m,n)
         u,s,vh = np.linalg.svd(H,full_matrices=False)
         return np.matmul(u,vh)
              
      def calWeight(self,x):
         featureNum = x.shape[1]
         self.a = self.randomOrthonormal(self.hiddenNode,featureNum)
         self.b = self.randomOrthonormal(1,self.hiddenNode)
         #h = 1/(1+np.exp(x@(self.a.T)+self.b))
         h = (x@self.a.T)+self.b
         beta = np.linalg.pinv(h)@x
         return beta
            

class OSELM:
        
    def __init__(self,hiddenNode=100,activation='sigmoid',forget=1):
        self.hiddenNode = hiddenNode
        self.activation = activation
        self.forget = forget
        
    
    def relu(self,x): # Rectified Linear Unit (Relu) activation function
        return np.maximum(0,x)
    
    def pureLin(self,x):
        return x
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
      
    def initialLearn(self,x,y):
        #each row of data represent each instance(record) of data      
        
        if x.shape[0]<self.hiddenNode:
            print("Cant boost, sample less than hidden node")
            return None
        
        
        featureNum = x.shape[1] # number of input data feature (number of column) 
        self.w = np.random.uniform(-1,1,[self.hiddenNode,featureNum]) #random weight range 0-1 
        self.b = np.random.uniform(-1,1,self.hiddenNode)
        
        if self.activation=='relu':
            self.H = self.relu(np.dot(x,self.w.T)+self.b)
        
        elif self.activation=='sigmoid':   
            self.H = self.sigmoid(np.dot(x,self.w.T)+self.b)
            
        elif self.activation=='pureLin':
            self.H = self.pureLin(np.dot(x,self.w.T)+self.b)
                    
        elif self.activation=='tanh':
            self.H = self.tanh(np.dot(x,self.w.T)+self.b)
        
        else: 
            print("unknow activation function")
        
        Hinv = np.linalg.pinv(self.H)
        self.beta = np.dot(Hinv,y)
        self.K = np.dot(self.H.T,self.H) #used for increment learn
        
 
        
        
        
    def incrementLearn(self,x,y):
        if  self.activation=='relu':
            Hk = self.relu(np.dot(x,self.w.T)+self.b) 
        
        elif self.activation=='sigmoid':   
             Hk = self.sigmoid(np.dot(x,self.w.T)+self.b)
            
        elif self.activation=='pureLin':
             Hk = self.pureLin(np.dot(x,self.w.T)+self.b)
             
        elif self.activation=='tanh':
             Hk = self.tanh(np.dot(x,self.w.T)+self.b)
        
        self.K = (self.forget)*self.K + np.dot(Hk.T,Hk) #update value of K that increase H value of each incoming sample
        K_inv = np.linalg.pinv(self.K)
        self.beta = self.beta + np.dot(np.dot(K_inv,Hk.T),y-np.dot(Hk,self.beta))
    
    
    def predict(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T)+self.b)
                    
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T)+self.b)
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T)+self.b)
            
        if self.activation=='tanh':
            H = self.tanh(np.dot(x,self.w.T)+self.b)
     
        return np.dot(H,self.beta)


class FOSELM:
    
    def __init__(self,hiddenNode=100,activation='sigmoid',forget=1,reg=0.001):
        self.hiddenNode = hiddenNode
        self.activation = activation
        self.forget = forget
        self.reg = reg
        self.firstTime = True
    
    def relu(self,x): # Rectified Linear Unit (Relu) activation function
        return np.maximum(0,x)
    
    def pureLin(self,x):
        return x
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def incrementLearn(self,x,y,forget=1):        
        self.forget = forget
        
        if self.firstTime == True:
            #print('this is first time')
            featureNum = x.shape[1] # number of input data feature (number of column) 
            self.w = np.random.uniform(-1,1,[self.hiddenNode,featureNum]) #random weight range 0-1 
            self.b = np.random.uniform(-1,1,self.hiddenNode)
            self.K = np.zeros([self.hiddenNode,self.hiddenNode])+(self.reg*np.identity(self.hiddenNode))
            self.beta = np.zeros([self.hiddenNode,y.shape[1]])
            self.firstTime = False
            
        if  self.activation=='relu':
            Hk = self.relu(np.dot(x,self.w.T)+self.b) 
        
        elif self.activation=='sigmoid':   
             Hk = self.sigmoid(np.dot(x,self.w.T)+self.b)
            
        elif self.activation=='pureLin':
             Hk = self.pureLin(np.dot(x,self.w.T)+self.b)
             
        elif self.activation == 'tanh':    
            Hk = self.tanh(np.dot(x,self.w.T)+self.b)
         
        else:
            print("unknow activation function")
        
        self.forget=forget
        self.K = (self.forget)*self.K + np.dot(Hk.T,Hk) #update value of K that increase H value of each incoming sample
        K_inv = np.linalg.inv(self.K)
        self.beta = self.beta + np.dot(np.dot(K_inv,Hk.T),y-np.dot(Hk,self.beta))
        
    
    def predict(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T)+self.b)
        
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T)+self.b)
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T)+self.b)
            
        elif self.activation == 'tanh':    
            H = self.tanh(np.dot(x,self.w.T)+self.b)
     
        return np.dot(H,self.beta) 
        
    
    def incrementLearn2(self,x,y): #use bias in hidden layer not in input layer     
        if self.firstTime == True:
            #print('this is first time')
            featureNum = x.shape[1] # number of input data feature (number of column) 
            self.w = np.random.uniform(-1,1,[self.hiddenNode,featureNum]) #random weight range 0-1 
            #self.w = np.random.randn(self.hiddenNode,featureNum)
            #self.w = np.random.normal(loc=0,size=(self.hiddenNode,featureNum))
            self.K = np.zeros([self.hiddenNode+1,self.hiddenNode+1])
            self.beta = np.zeros([self.hiddenNode+1,y.shape[1]])
            self.firstTime = False
            
        if  self.activation=='relu':
            Hk = self.relu(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
        
        elif self.activation=='sigmoid':   
             Hk = self.sigmoid(np.dot(x,self.w.T))
             Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
                            
        elif self.activation=='pureLin':
             Hk = self.pureLin(np.dot(x,self.w.T))
             Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)    
                            
        elif self.activation == 'tanh':    
            Hk = self.tanh(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
         
        else:
            print("unknow activation function")
        
        self.K = (self.forget)*self.K + np.dot(Hk.T,Hk) #update value of K that increase H value of each incoming sample
        K_inv = np.linalg.pinv(self.K)
        self.beta = self.beta + np.dot(np.dot(K_inv,Hk.T),y-np.dot(Hk,self.beta))

    
    def predict2(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
                          
        elif self.activation == 'tanh':    
            H = self.tanh(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        
        return np.dot(H,self.beta) 


class AE_FOSELM:
   
   def __init__(self,hiddenNode=50,activation='sigmoid',forget=1):
       self.hiddenNode = hiddenNode
       self.activation ='sigmoid'
       self.forget = forget
       self.autoEncoder = self.AUTO_ENCODER(hiddenNode)
       self.firstTime = True
                       
   def relu(self,x): # Rectified Linear Unit (Relu) activation function
        return np.maximum(0,x)
    
   def pureLin(self,x):
        return x
    
   def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
   def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


   def incrementLearn(self,x,y):
       if self.firstTime == True:
          self.w = self.autoEncoder.calWeight(x)
          #print('x = ')
          #print(x)
          #print('In first time w = ')
          #print(self.w)
          self.K = np.zeros([self.hiddenNode+1,self.hiddenNode+1])
          self.beta = np.zeros([self.hiddenNode+1,y.shape[1]])
          self.firstTime = False        
       
        
       if self.activation=='relu':
            Hk = self.relu(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
        
       elif self.activation=='sigmoid':   
            Hk = self.sigmoid(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
                            
       elif self.activation=='pureLin':
            Hk = self.pureLin(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)    
                            
       elif self.activation == 'tanh':    
            Hk = self.tanh(np.dot(x,self.w.T))
            Hk = np.append(Hk,np.ones([Hk.shape[0],1]),axis=1)
         
       else:
            print("unknow activation function")
            
       
       self.K = (self.forget)*self.K + np.dot(Hk.T,Hk) #update value of K that increase H value of each incoming sample
       K_inv = np.linalg.pinv(self.K)
       self.beta = self.beta + np.dot(np.dot(K_inv,Hk.T),y-np.dot(Hk,self.beta))
        
        
   def predict(self,x):
        if self.activation=='relu':
            H = self.relu(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        
        if self.activation=='sigmoid':   
            H = self.sigmoid(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
            
        if self.activation=='pureLin':
            H = self.pureLin(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
                          
        elif self.activation == 'tanh':    
            H = self.tanh(np.dot(x,self.w.T))
            H = np.append(H,np.ones([H.shape[0],1]),axis=1)
        
        return np.dot(H,self.beta) 
       
   
  
   class AUTO_ENCODER:
      def __init__(self,hiddenNode):
         self.firstTime = True
         self.hiddenNode = hiddenNode
     
      def randomOrthonormal(self,m,n):
         H = np.random.rand(m,n)
         u,s,vh = np.linalg.svd(H,full_matrices=False)
         return np.matmul(u,vh)
              
      def calWeight(self,x):
         if self.firstTime == True:
             featureNum = x.shape[1]
             self.a = self.randomOrthonormal(self.hiddenNode,featureNum)
             self.b = self.randomOrthonormal(1,self.hiddenNode)
             self.K = np.zeros([self.hiddenNode,self.hiddenNode])
             self.beta = np.zeros([self.hiddenNode,featureNum])
             self.firstTime = False
         
         
         Hk = 1/(1+np.exp(-(np.dot(x,self.a.T)+self.b)))
         #print('Hk in auteoncoder =')
         #print(Hk)
         self.K = self.K + np.dot(Hk.T,Hk) #update value of K that increase H value of each incoming sample
         K_inv = np.linalg.pinv(self.K)
         self.beta = self.beta + np.dot(np.dot(K_inv,Hk.T),x-np.dot(Hk,self.beta))
         #print('autoencoder a = ')
         #print(self.a)
         #print('autoencoder b = ')
         #print(self.b)
         #print('autoencoder beta = ')
         #print(self.beta)
         return self.beta


class OSELM_Layer:
    
    def __init__(self,init_beta):
        self.beta = init_beta
        self.K = 0
    
    
    def predict(self,H):
        return np.dot(H,self.beta)


    def incrementLearn(self,H,Y):
        self.K = self.K+np.dot(H.T,H)
        K_inv = np.linalg.pinv(self.K) 
        candidateBeta = self.beta + np.dot(np.dot(K_inv,H.T),Y-np.dot(H,self.beta))
        s = self.cosSimilar(self.beta,candidateBeta)
        if s>0.7:
            self.beta = candidateBeta
        
        #self.beta = s*candidateBeta + (1-s)*self.beta
    
    
    def cosSimilar(self,A,B):
        AdotB = np.dot(A.T,B)
        Anorm = np.linalg.norm(A)
        Bnorm = np.linalg.norm(B)
        return AdotB/(Anorm*Bnorm)       


def MAPE(y_actual,y_hat):
    y_actual_num = y_actual.shape[0]
    y_hat_num = y_hat.shape[0]
    
    if y_actual_num!=y_hat_num:
        print("dimension difference")
    
    if y_actual.all() == 0:
        return 999
    
    return np.mean(np.absolute((y_actual-y_hat)/y_actual))*100            


def RMSE(y_actual,y_hat):
    y_actual_num = y_actual.shape[0]
    y_hat_num = y_hat.shape[0]
   
    if y_actual_num!=y_hat_num:
        print("dimension difference")
        
    return np.sqrt(np.mean((y_actual-y_hat)**2))


def MAE(y_actual,y_hat):
    y_actual_num = y_actual.shape[0]
    y_hat_num = y_hat.shape[0]
    
    if y_actual_num!=y_hat_num:
        print("dimension difference")
        
    return np.mean(np.absolute((y_actual-y_hat)))            

    
def buildTimeLagData(rawData,timeLag,batchSize):
    '''
    This function use sliding window techique, window size=timeLag 
    and step size=batch to create temp matix and seperate to x and y         
    '''
    dataLen = rawData.size #calculate number of rawData
    '''
    index of last element of last row of temp matrix should less than dataLen
    (maxRowIndex*batchSize)+timeLag+batchSize <= dataLen then
    '''
    maxRowIndex = int(np.floor((dataLen-timeLag-batchSize)/batchSize)) 
    temp = np.zeros([maxRowIndex+1,timeLag+batchSize])
    
    #Create temp matrix
    for i in range(maxRowIndex+1):
        temp[i,:] = rawData[i*batchSize:i*batchSize+(timeLag+batchSize)]
    
    x=temp[:,0:timeLag]
    y=temp[:,timeLag:timeLag+batchSize]    
    return x,y    

def movingAvg(x,window):
    xSum = np.cumsum(x)
    return (xSum[window:]-xSum[:-window])/float(window)

def groupAvg(x,member):
    groupNum = np.int(x.shape[0]/member)
    xAvg = np.zeros(groupNum)
    for i in range(groupNum):
        xAvg[i]=x[i*member:i*member+member].mean()       
    return xAvg
        

def diff(x): #cal the differrnt between data and previous data, x is numpy array
    diffVal = []
    for i in range(x.shape[1]-1): #x.shape[1] = number of time lag
        diffVal.append(x[:,i+1]-x[:,i])
    return np.array(diffVal).T

def cosSimilar(A,B):
    AdotB = np.dot(A.T,B)
    Anorm = np.linalg.norm(A)
    Bnorm = np.linalg.norm(B)
    return AdotB/(Anorm*Bnorm)


def eculideanDist(A,B):
    return np.linalg.norm(A-B)

def radialBasis(A,B):
    dist = np.linalg.norm(A-B)
    return np.exp(-dist)


def DTW(s, t):
    s = s.reshape(-1)
    t = t.reshape(-1)
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    #calculate dtw matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    
    #calculate path and w
    i = dtw_matrix.shape[0]-1
    j = dtw_matrix.shape[1]-1
    w = dtw_matrix[i,j]
    count = 1
    
    while(i>0 and j>0):
        if(dtw_matrix[i-1,j-1]<=dtw_matrix[i,j-1] and dtw_matrix[i-1,j-1]<=dtw_matrix[i-1,j]):
            w = w+dtw_matrix[i-1,j-1]
            i = i-1
            j = j-1
            count = count+1
 
        elif(dtw_matrix[i-1,j]<=dtw_matrix[i-1,j-1] and dtw_matrix[i-1,j]<=dtw_matrix[i,j-1]):
            w = w+dtw_matrix[i-1,j]
            i = i-1
            count = count+1

        else:
            w = w+dtw_matrix[i,j-1]
            j = j-1
            count = count+1
   
    
    return w/(count-1)


def CID_ECU(q,c):
    
    def diff(x): #cal the differrnt between data and previous data, x is numpy array
        diffVal = []
        for i in range(x.shape[1]-1): #x.shape[1] = number of time lag
            diffVal.append(x[:,i+1]-x[:,i])
        return np.array(diffVal).T
       
    ed = np.linalg.norm(q-c)
        
    ce_q = np.sqrt(np.sum(np.square(diff(q))))
    ce_c = np.sqrt(np.sum(np.square(diff(c))))
    
    cf = max(ce_q,ce_c)/min(ce_q,ce_c)
    
    return ed*cf


def CID_DTW(q,c):
    
    def diff(x): #cal the differrnt between data and previous data, x is numpy array
        diffVal = []
        for i in range(x.shape[1]-1): #x.shape[1] = number of time lag
            diffVal.append(x[:,i+1]-x[:,i])
        return np.array(diffVal).T  
    
    def DTW(s, t):  
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
    
    #calculate dtw matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s[i-1] - t[j-1])
                # take last min from a square box
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
    
    #calculate path and w
        i = dtw_matrix.shape[0]-1
        j = dtw_matrix.shape[1]-1
        w = dtw_matrix[i,j]
        count = 1
    
        while(i>0 and j>0):
            if(dtw_matrix[i-1,j-1]<=dtw_matrix[i,j-1] and dtw_matrix[i-1,j-1]<=dtw_matrix[i-1,j]):
                w = w+dtw_matrix[i-1,j-1]
                i = i-1
                j = j-1
                count = count+1
 
            elif(dtw_matrix[i-1,j]<=dtw_matrix[i-1,j-1] and dtw_matrix[i-1,j]<=dtw_matrix[i,j-1]):
                w = w+dtw_matrix[i-1,j]
                i = i-1
                count = count+1

            else:
                w = w+dtw_matrix[i,j-1]
                j = j-1
                count = count+1
   
    
        return w/(count-1)
      
    
    ed = DTW(q[0],c[0])
        
    ce_q = np.sqrt(np.sum(np.square(diff(q.reshape(len(q),-1)))))
    ce_c = np.sqrt(np.sum(np.square(diff(c.reshape(len(c),-1)))))
    
    cf = max(ce_q,ce_c)/min(ce_q,ce_c)
    
    return ed*cf


def complexity_estimate(s):
    n = len(s)
    diff = []
    for i in range(1,n):
        diff.append(s[i]-s[i-1])
    
    diff = np.array(diff)
    
    return np.sqrt(np.sum(np.square(diff))) #root sum square of diff
    
    
def randomOrthonormalMatrix(m,n):
    H = np.random.rand(m,n)
    u,s,vh = np.linalg.svd(H,full_matrices=False)
    return np.matmul(u,vh)
    

def similarityVector(template,x,metrics='cosSimilarity'):
    numWave = template.shape[0]
    simVec = np.zeros([1,numWave])
    
    for i in range(numWave):
        
        if metrics == 'cosSimilarity':
            simVec[0,i] = cosSimilar(template[i], x[0])
        elif metrics == 'eculidean':
            simVec[0,i] = eculideanDist(template[i], x[0])
        elif metrics == 'dtw':
            simVec[0,i] = DTW(template[i], x[0])
        elif metrics == 'cid_ecu':
            simVec[0,i] = CID_ECU(template[[i]], x[[0]])
        elif metrics == 'cid_dtw':
            simVec[0,i] = CID_DTW(template[i], x[0])
        elif metrics == 'rbf':
            simVec[0,i] = radialBasis(template[i],x[0])
        elif metrics == 'hybrid':
            simVec[0,i] = radialBasis(template[i],x[0])*cosSimilar(template[i],x[0])    
        else:
            print('unknow similarity function')
    
    return simVec        

class kalmanRegression:
#single variable linear regression

    def __init__(self,featureNum,p0,w0,r):
        self.featureNum = featureNum #number of input feature
        self.p = p0  #covariance of estimate [featureNum,featureNum]
        self.w = w0  #state in estimate or it's weight of regression [1,featureNum]
        self.r = r  #varianve of measurement in this it's used for tune kalman [1,1]
        
    def learn(self,x,y): #x have dim as [1,input feature] and y -> [1,1]
        res = y-np.matmul(x,self.w.T)
        s = np.matmul(np.matmul(x,self.p),x.T)+self.r
        self.k = np.matmul(self.p,x.T)/s
        self.w = self.w+self.k.T*res
        self.p = (1-np.matmul(self.k,x))*self.p
        return self.w

def clumpMean (data,method='iqr'): #data should array
    
    data = data[~np.isnan(data)] #eliminate NaN value
    
    if data.shape[0] <=2:
        return(np.mean(data))
    
    if method == 'iqr':
        q1 = np.percentile(data,25)
        q3 = np.percentile(data,75)
        iqr = q3-q1
        upper = q3+(1.5*iqr)
        lower = q1-(1.5*iqr)
        clump = data[(data>lower)&(data<upper)]
        if np.any(clump): #have data
            return np.mean(clump)
        else: #no have any data
            return 0
        
    if method == 'sigma':
        sigma = np.std(data)
        mean = np.mean(data)
        upper = mean+sigma
        lower = mean-sigma
        clump = data[(data>lower)&(data<upper)]
        if np.any(clump): #have data
            return np.mean(clump)
        else: #no have any data
            return 0

def forgettingFactor (mape,limitMAPE):
    if np.isnan(mape) or mape<limitMAPE:
        return 1
    else:
        return np.exp(1-(mape/limitMAPE))
    
    
def syntLoad (dataNum,period=24,meanLoad=0.8,percentNoise=30,angle=0):
    timeRange = np.arange(dataNum)
    syntData = np.zeros(dataNum)
    noisePeak = percentNoise/100
    phase = angle*np.pi/180

    for t in timeRange:
        syntData[t] = ((1-meanLoad)*(1-noisePeak))*np.sin(2*np.pi*(1/period)*t+phase)+meanLoad+((1-meanLoad)*noisePeak*np.random.rand())
    
    scaler = syntData.max()
    syntData = syntData/scaler
    
    return syntData

def addNoise(data,percentNoise=30,pdf='uniform',rate=1):
    try:
        len(data)  #data have size  (dimension)
        data = np.array(data) #convert to array or make sure it's an array
    except:
        data = np.array([data]) #it's single value with no dimension, give it dimension    
    
    if pdf == 'uniform':
        data = data+(data*percentNoise/100)*np.random.rand(len(data)) 
        return data
    
    elif pdf == 'gaussian':
        #data = np.random.normal(loc=data,scale=percentNoise/100)
        #data = data+(data*percentNoise/100)*np.random.normal()
        data = data+(data*percentNoise/100)*np.random.normal(loc=0,scale=0.1,size=len(data))
        return data
    
    elif pdf == 'exponential':
        data = data+(data*percentNoise/100)*np.random.exponential(scale=1/rate,size=len(data))
        return data

def shiftData(data,number):
    outData = []
    outData.append(data)
    for i in range(number-1):
        data = np.roll(data,-1)
        outData.append(data)
    outData = np.array(outData)
    return outData


class rlsEnsemble():
   def __init__(self,inDim,outDim,reg=0.001):
       self.inDim = inDim   #dimentsion of input vector ->[1,inDim]
       self.outDim = outDim #dimension of putput vector ->[1,outDim]
       #initial weight as simple average of all input
       self.W = (1/inDim)+np.zeros([inDim,outDim])  #weight for calculate output
       self.K = np.zeros([inDim,inDim])+reg*(np.identity(inDim)) 
   
    
   def learn(self,x,y,forget=1):
       if (x.shape != (1,self.inDim)):
           print('wrong dimension for input data')
           
       if (y.shape != (1,self.outDim)):
           print('wrong dimension for output data')
           
       
       self.K = forget*self.K+np.dot(x.T,x)
       K_inv = np.linalg.inv(self.K)
       self.W = self.W+np.dot(K_inv,np.dot(x.T,(y-np.dot(x,self.W))))
       sumW = np.sum(self.W)
       self.W = self.W/sumW
    
       
   def predict(self,x):
       return np.dot(x,self.W)
   
    

   
          