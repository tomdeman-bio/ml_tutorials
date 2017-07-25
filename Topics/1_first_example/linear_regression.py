import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

a = 0.5
b = 1.0

# x from 0 to 10
x = 30 * np.random.random(1000)
print('x')
print(x)
pause()
# y = a*x + b with noise
y = a * x + b + np.random.normal(size=x.shape)
print('y')
print(y)
pause()

# create a linear regression
reg = LinearRegression()
reg.fit(x[:, None], y)
print('model')
print(reg)
pause()
print('Estimated a:', reg.coef_)
print('Estimated b:', reg.intercept_)
pause()

# predict y from the data
x_new = np.linspace(0, 30, 10)
y_new = reg.predict(x_new[:, None])
print('x_new')
print(x_new)
pause()

print('y_new')
print(y_new)
pause()

# plot the results
ax = plt.axes()
ax.scatter(x, y, color='green')
ax.scatter(x_new, y_new, color='blue')
ax.plot(x_new, y_new, color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('tight')
plt.savefig('linear_regression.png', bbox_inches='tight')
