import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,11)
y = np.arange(1,11)

plt.style.use('dark_background')
fig , axes = plt.subplots(1,2 ,figsize = (20,9) ,dpi =100)
axes[0].plot(x, y , 'oy' ,label = 'Data Point')

np.random.seed(42)
w = np.random.rand()
b = np.random.rand()

eta = 0.01
epochs = 1000
costvalue = np.zeros(1000)

for i in range(epochs):
    yhat = (w * x) + b 
    w = w  - eta * (np.mean((yhat - y) * x)) # or .  w = w  - eta * ((yhat - y) @ x) / len (x)
    b = b - eta * (np.mean(yhat - y))
    mse =  0.5 * np.mean((yhat - y)**2)
    costvalue[i] = mse 
    
    if costvalue[i] < 1e-5 : # 1e-5 = 1x10^-5
        break


xvalue = np.arange(-10,11)
yvalue = (w * xvalue ) + b

axes[0].plot(xvalue, yvalue , '-r' ,label = 'Learned Line')
axes[0].grid(True , alpha = 0.3)
axes[0].set_title('Y = x' , fontsize = 16)
axes[0].set_xlabel('x', fontsize =12)
axes[0].set_ylabel('y', fontsize =12)
axes[0].axhline(y = 0 , alpha = 0.5 ,linestyle = '--')
axes[0].axvline(x = 0 , alpha = 0.5 ,linestyle = '--')
axes[0].legend()

axes[1].plot(range(i+1) , costvalue[:i+1] )
axes[1].grid(True , alpha = 0.3)
axes[1].set_title('training cost vs epochs' , fontsize = 16)
axes[1].set_xlabel('epochs', fontsize =12)
axes[1].set_ylabel('cost', fontsize =12)
axes[1].axhline(y = 0 , alpha = 0.5 ,linestyle = '--')
axes[1].axvline(x = 0 , alpha = 0.5 ,linestyle = '--')

plt.tight_layout()
plt.show()

print(f'Final W : {w:0.4f}')
print(f'Final b : {b:0.4f}')
print(f'Final cost: {costvalue[i]: 0.4f}')
print(f'Training stopped at epoch: {i+1}')
