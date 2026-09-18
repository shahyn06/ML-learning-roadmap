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

data = np.concatenate((cats , dogs) , axis = 0)
labels = np.concatenate((cats_label , dogs_label ) , axis = 0)


plt.subplot(1 , 2 , 1)
plt.scatter(cats[: , 0] , cats[: , 1] , marker = 'o' , label = 'Cats (-1)' , color = 'yellow' , s = 8  )
plt.scatter(dogs[: , 0] , dogs[: , 1] , marker = 'x' , label = 'Dogs (1)' , color = 'red' , s = 8  )

#
np.random.seed(42)
w1 = np.random.randn(1)[0]
w2 = np.random.randn(1)[0]
b = np.random.randn(1)[0]

epochs = 100
eta = 0.01
errors = np.zeros(epochs)

#

for i in range(epochs):
    error_count = 0
    for j in range(len(data)):
        z = (w1 * data[ j  , 0] )+(w2 * data[ j , 1] ) + b
        #step activition
        yhat = 1 if z >= 0 else  -1
        
        if yhat != labels[j]:
            w1 = w1 + eta * (labels[j] - yhat) * data[ j  , 0]
            w2 = w2 + eta * (labels[j] - yhat) * data[ j  , 1]
            b = b + eta * (labels[j] - yhat)
            error_count +=1 
    
    errors[i] = error_count
    if error_count == 0 :
        break

X = np.linspace(0 , 10 , 100)
Y = (-w1 * X - b) / w2

plt.plot(X , Y , color = 'white' , linewidth = 2 , label = 'Decision boundary')
plt.xlabel('Feature 1' , color = 'yellow')
plt.ylabel('Feature 2' , color = 'yellow')
plt.title('Perceptron Decision Boundary', color = 'yellow' ,fontsize = 14)
plt.grid(alpha = 0.3)
plt.legend()

plt.subplot(1 , 2, 2 )

plt.plot(range(0 , i+1) , errors[0 : i+1 ] )
plt.xlabel('epochs' , color = 'yellow')
plt.ylabel('Number of errors' , color = 'yellow')
plt.title('Training errors vs Epochs', color = 'yellow' ,fontsize = 14)
plt.grid(alpha = 0.3)

plt.tight_layout()
plt.show()

print(f'Final w1 : {w1} ')
print(f'Final w2 : {w2} ')
print(f'Final bias : {b} ')

predictions = np.array([1 if (w1 * data[i , 0] + w2 * data[i , 1] + b) >= 0 else -1 for i in range(len(data))])
accuracy = np.mean(predictions == labels)* 100
print(f'Final accuracy : {accuracy} %')
