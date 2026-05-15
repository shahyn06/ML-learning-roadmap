import numpy as np 
import matplotlib.pyplot as plt

class Pipeline :
    def __init__(self , weights ,label , bias = 0 ):
        self.weights=weights
        self.bias =bias
        self.label=label

    def z_scoring(self , data):
        result = (data - np.mean(data , axis = 0) )  / np.std(data , axis = 0)
        return result 

    def prediction(self , data):
        normalized_data = self.z_scoring(data)
        predict = normalized_data @ self.weights +  self.bias
        return predict

    def MSE(self , data ):
        pred_data = self.prediction(data)
        mse = np.mean((pred_data - self.label)**2)
        return mse 

    def visual(self , data  ):
        plt.style.use('dark_background')
        fig ,axes =plt.subplots(2,2 , figsize = (14 , 10))
        normalize = self.z_scoring(data)
        #######plot(0,0) , Original Data ########
        
        axes[0,0].scatter(data[:15,0] ,
                          data[:15,1] ,
                          s = 100 , 
                          alpha = 0.5  ,
                          color = 'cyan' ,
                          linewidths = 1.5 ,
                          edgecolors ='white',
                          label = 'Class A')
        
        axes[0,0].scatter(data[15:,0] ,
                          data[15:,1] ,
                          s = 100 , 
                          alpha = 0.5  ,
                          color = 'red' ,
                          linewidths = 1.5,
                          edgecolors ='white',
                          label = 'Class B')
        
        axes[0,0].grid(True , alpha = 0.3)
        axes[0,0].set_title('Original Data' , fontsize = 16)
        axes[0,0].set_xlabel('Class A' , fontsize = 12)
        axes[0,0].set_ylabel('Class B' , fontsize = 12)
        axes[0,0].axhline(y=0 , alpha = 0.4 , linestyle = '--')
        axes[0,0].axvline(x=0 , alpha = 0.4 , linestyle = '--')
		
		#######plot(0,1) , Normalized Data ########
		
        axes[0,1].scatter(normalize[:15,0] ,
                          normalize[:15,1] ,
                          s = 100 ,
                          alpha = 0.5  ,
                          color = 'cyan' ,
                          linewidths = 1.5 ,
                          edgecolors ='white',
                          label = 'Class A')
        
        axes[0,1].scatter(normalize[15:,0] ,
                          normalize[15:,1] ,
                          s = 100 , 
                          alpha = 0.5  ,
                          color = 'red' ,
                          linewidths = 1.5,
                          edgecolors ='white',
                          label = 'Class B')
        
        axes[0,1].grid(True , alpha = 0.3)
        axes[0,1].set_title('Normalized Data' , fontsize = 16)
        axes[0,1].set_xlabel('Class A' , fontsize = 12)
        axes[0,1].set_ylabel('Class B' , fontsize = 12)
        axes[0,1].axhline(y=0 , alpha = 0.4 , linestyle = '--')
        axes[0,1].axvline(x=0 , alpha = 0.4 , linestyle = '--')
        
        #######plot(1,0) , Predictions VS Y-True ########
        
        pre_data = self.prediction(data)
        axes[1,0].scatter(range(len(pre_data)) ,
                          pre_data ,
                          s = 100,
                          color = 'green' ,
                          alpha = 0.5 ,
                          edgecolor = 'white',
                          linewidths = 1.5 ,
                          label = 'Predictions')
        
        axes[1,0].scatter(range(len(pre_data)) ,
                  self.label ,
                  s = 100,
                  color = 'red' ,
                  alpha = 0.5 ,
                  edgecolor = 'white',
                  linewidths = 1.5 ,
                  label = 'Y True')
        axes[1,0].grid(True , alpha = 0.3)
        axes[1,0].set_title('Predictions VS Y-True', fontsize = 16)
        axes[1,0].set_xlabel('sample index' , fontsize =12)
        axes[1,0].set_ylabel('pred , Y-True' , fontsize =12)
        axes[1,0].axhline(y=0 ,linestyle = '--' ,alpha = 0.4)
        
        #######plot(1,1) , Histogram #######
        
        axes[1,1].hist(data[:,0] , alpha = 0.5 , edgecolor = 'white' , color = 'cyan' , bins = 30)
        axes[1,1].hist(data[:,1] , alpha = 0.5 , edgecolor = 'white' , color = 'red' , bins = 30)
        axes[1,1].grid(True , axis = 'y' , alpha = 0.3)
        axes[1,1].set_title('Feature Distribution', fontsize=16)
        axes[1,1].set_xlabel('Feature Values', fontsize=12)
        axes[1,1].set_ylabel('Frequency', fontsize=12)
        
        plt.tight_layout()
        plt.show()

##############################
#Data
##############################
np.random.seed(42)

#Label
label_A = np.zeros(15)
label_B = np.ones(30)
label = np.hstack((label_A,label_B))

#Class
classA = np.random.randn(15,2)+1
classB = np.random.randn(30,2)+3
dataset = np.vstack((classA,classB))

#weights 
weights = np.array([1.2,0.6])

model = Pipeline(weights, label)
#Normalized
Normalized_data = model.z_scoring(dataset)
print (F'Normalized Data : \n{Normalized_data}')

#prediction
predictions = model.prediction(dataset)
print (f'predictions : {predictions}')

#mse
mse =model.MSE(dataset)
print (f'MSE : {mse:.4f}')

#visual
model.visual(dataset)
