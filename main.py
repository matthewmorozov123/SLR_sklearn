from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


Age = np.array([43,21,25,42,57,59,35,15,55,50,65,10,45,35])
Glucose = np.array([99,65,79,75,87,81,80,80,90,70,95,67,90,82])
AgeReshaped = Age.T.reshape(-1,1)   #T=transpose, -1=unknown rows, 1=col
GlucoseReshaped = Glucose.T.reshape(-1,1)

def simpleLRcoeffsSkLearn(X, Y):
  regr= LinearRegression()
  regr.fit(X, Y)
  yhat= regr.predict(X)

  return regr.coef_, regr.intercept_, yhat

plt.scatter(Age, Glucose)
coef, intercept, yhat= simpleLRcoeffsSkLearn(AgeReshaped, GlucoseReshaped)

plt.plot(Age, yhat, color= "red")
plt.title("Age vs Glucose")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.show()


def simpleLRscoreSkLearn(y, yhat):
  r2= r2_score(y, yhat)
  return r2