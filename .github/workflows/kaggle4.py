# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:18:05 2020

@author: harsh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:34:56 2020

@author: harsh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import tree
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC

data= pd.read_csv('C:/Users/harsh/Desktop/KaggleComp/train.csv')
#X=data[data.columns[0:86]]
features=['Id','V45','V56','V58','V65','V77','V83','V85']
X=data.loc[:,features].values

Y=data[data.columns[86:87]]
#print(Y)

counter = Y.Buy.value_counts()
print(counter)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3,random_state=44)



scl=StandardScaler()
scl.fit(X_train)

X_train=scl.transform(X_train)
X_test=scl.transform(X_test)

sm = SMOTE(random_state=27)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
#X_test, Y_test = sm.fit_sample(X_test, Y_test)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

counter = Y_train.Buy.value_counts()
print(counter)

counter = Y_test.Buy.value_counts()
print(counter)

model=tree.DecisionTreeClassifier(random_state=0, max_depth=10,criterion='entropy')

model.fit(X_train,Y_train)

pred=model.predict(X_test)



print(metrics.confusion_matrix(Y_test,pred))
print(metrics.classification_report(Y_test,pred))

print("Accuracy:",metrics.accuracy_score(Y_test, pred))

tree.plot_tree(model)

AdaBoost = AdaBoostClassifier(n_estimators=400,learning_rate=1.6,algorithm='SAMME',base_estimator=model,random_state=9)

AdaBoost.fit(X_train,Y_train)

Bag=BaggingClassifier(base_estimator=model,n_estimators=100,random_state=10).fit(X_train,Y_train)
BagPred=Bag.score(X_test,Y_test)
BagPrediction=Bag.predict(X_test)
print(metrics.confusion_matrix(Y_test,BagPrediction))
print('Bag Accuracy',BagPred)

'''
predictions = cross_validate(AdaBoost,X_train,Y_train,cv=10)
pred_per=np.mean(predictions['test_score'])
print(predictions)


print('The accuracy is: ',pred_per*100,'%')
'''
AdaPred=AdaBoost.predict(X_test)
print(AdaPred)
print(metrics.confusion_matrix(Y_test,AdaPred))

prediction = AdaBoost.score(X_test,Y_test)
print('The boosting accuracy is: ',prediction*100,'%')

'''

test_data=pd.read_csv('C:/Users/harsh/Desktop/KaggleComp/test.csv')
print(test_data.shape)
print(test_data.columns)
test_data=test_data.loc[:,features].values
#test_data=test_data[test_data.columns[1:86]]
print(test_data.shape)
test_predict=AdaBoost.predict(test_data)
test_predict=np.reshape(test_predict,(1715,1))
print(test_predict.shape)

Test_Res = open("C:/Users/harsh/Desktop/KaggleComp/Boost.csv", "w")
for row in test_predict:
    np.savetxt(Test_Res, row)

Test_Res.close()

'''
