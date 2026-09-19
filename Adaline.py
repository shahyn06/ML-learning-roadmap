import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.figure(figsize=(14, 10))
np.random.seed(42)
n_samples = 50

#Cats 

cats_x1 = np.random.uniform(1 , 5 , n_samples)
cats_x2 = np.random.uniform(1 , 5 , n_samples)
cats = np.column_stack([cats_x1 , cats_x2])
cats_label = -np.ones(n_samples)


#Dogs 

dogs_x1 = np.random.uniform(5 , 9 , n_samples)
dogs_x2 = np.random.uniform(5 , 9 , n_samples)
dogs = np.column_stack([dogs_x1 , dogs_x2])
dogs_label = np.ones(n_samples)


#data

data = np.concatenate((cats , dogs) , axis = 0) # shape(100 , 2)
bias = np.ones((1 , len(data)))   # shape(1 , 100)
labels = np.concatenate((cats_label , dogs_label ) , axis = 0)
Data_with_bais = np.column_stack((bias.T , data))

plt.subplot(1 , 2 , 1)
plt.scatter(cats[: , 0] , cats[: , 1] , marker = 'o' , label = 'Cats (-1)' , color = 'yellow' , s = 8  )
plt.scatter(dogs[: , 0] , dogs[: , 1] , marker = 'x' , label = 'Dogs (1)' , color = 'red' , s = 8  )

#
np.random.seed(42)
w = np.random.randn(3)

epochs = 1000
eta = 0.01
cost_value = np.zeros(epochs)

#
for i in range(epochs) :
    z = Data_with_bais @ w 
    e = labels - z  
    w = w + eta *  (Data_with_bais.T @ e ) / len(data)
    
    cost_value[i] = 0.5 * np.mean(e**2)
    if cost_value[i] < 1e-5 :
        break 


b =  w[0]
w1 = w[1]
w2 = w[2]


X = np.linspace(0, 10 ,100)
Y = -(w1 * X + b) / w2

plt.plot( X , Y , linewidth = 3 , label = 'Decision boundry' , color = 'white')
plt.xlabel('Feature 1' , color = 'yellow' , fontsize = 10)
plt.ylabel('Feature 2' , color = 'yellow' , fontsize = 10)
plt.title('Adaline Decision Boundary' , color = 'yellow' , fontsize = 15)
plt.grid(True  , alpha = 0.3)
plt.legend()


plt.subplot( 1 ,2 ,2)
plt.plot( range(epochs) , cost_value[:i+1] , linewidth = 3 , color = 'white')
plt.xlabel('epochs' , color = 'yellow' , fontsize = 10)
plt.ylabel('cost value' , color = 'yellow' , fontsize = 10)
plt.title('Training' , color = 'yellow' , fontsize = 15)
plt.grid(True  , alpha = 0.3)

plt.tight_layout()
plt.show()


print(f'Final weights w1: {w1}, w2: {w2}, b: {b}')
print(f'Training stopped after {i+1} epochs with final cost value: {cost_value[i]}')

preddictions = np.where(z >= 0 , 1 , -1)
accuracy = np.mean(preddictions == labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
