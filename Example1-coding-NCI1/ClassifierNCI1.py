## load adjmatrix
import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Dense, Activation, Dropout, LSTM,Bidirectional,GRU
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences


from keras.regularizers import l2#

fold=10
tunelength=30
onetimelogacc=[]
onetimesvmacc=[]
onetimeDNNacc=[]



cvtimes=1
cvnum=1


load_fn=str(".\\NCI1\\NCI1cv{}traintimes{}.mat".format(cvnum,cvtimes) )

data = sio.loadmat(load_fn)
NCIlabelandsequence=data['trainlabelandsequence']
NCItrainsequence=NCIlabelandsequence[:,1:112]
NCItrainlabel=NCIlabelandsequence[:,0]
NCItrainlabel=np.asarray(NCItrainlabel,'int64')
#load test data

load_fn2=str(".\\NCI1\\NCI1cv{}testtimes{}.mat".format(cvnum,cvtimes) )
data2 = sio.loadmat(load_fn2)
NCIlabelandsequence2=data2['testlabelandsequence']
NCItestsequence=NCIlabelandsequence2[:,1:112]
NCItestlabel=NCIlabelandsequence2[:,0]
NCItestlabel=np.asarray(NCItestlabel,'int64')



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# padding operation
x_train=NCItrainsequence
x_test=NCItestsequence
y_train=NCItrainlabel
y_test=NCItestlabel

x_train=pad_sequences(x_train,truncating='post',maxlen=tunelength)  
x_test=pad_sequences(x_test,truncating='post',maxlen=tunelength)

# Standardized operation

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

"""
model1:logisticRegression
"""
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='liblinear',multi_class='ovr',max_iter=1000,verbose=1, n_jobs=1)
classifier.fit(x_train_std,y_train)
y_predict=classifier.predict(x_test_std)
score=classifier.score(x_test_std,y_test)
print("Accuracy", score)
onetimelogacc.append(score)
"""
model2:svm
"""

from sklearn import svm
model=svm.SVC(kernel='rbf',max_iter=1000)
model.fit(x_train_std,y_train)
score2=model.score(x_test_std,y_test)
print("Accuracy", score2)
onetimesvmacc.append(score2)

##########################

# Deep neural network on keras platform 
###########################

model=Sequential()
model.add(layers.Dense(64,input_dim=tunelength,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1,activation='sigmoid'))

opt=Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1,factor=0.1, min_lr=0.000000001,mode='auto') #learning rate decrease

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
model.summary()
history=model.fit(x_train_std,y_train,
				  epochs=300,
				  verbose=True,
#                 validation_split=0.1,
				  callbacks=[reduce_lr],
				  validation_data=(x_test,y_test),
				  batch_size=50)
loss,accuracy1=model.evaluate(x_train_std,y_train,verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss,accuracy2=model.evaluate(x_test_std,y_test,verbose=0)
print("Test Accuracy: {:.4f}".format(accuracy2))
onetimeDNNacc.append(accuracy2)










