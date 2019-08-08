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
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


from keras.regularizers import l2#

fold=10
tunelength=13
onetimelogacc=[]
onetimesvmacc=[]



cvtimes=1
cvnum=1



load_fn=str(".\\ENZYMES\\ENZYMEScv{}traintimes{}.mat".format(cvnum,cvtimes) )
data = sio.loadmat(load_fn)
NCIlabelandsequence=data['trainlabelandsequence']
NCItrainsequence=NCIlabelandsequence[:,1:127]
NCItrainlabel=NCIlabelandsequence[:,0]
NCItrainlabel=np.asarray(NCItrainlabel,'int64')
labelonehot=LabelEncoder()
labelonehot.fit(NCItrainlabel)
encodeY=labelonehot.transform(NCItrainlabel)
dummylabel=np_utils.to_categorical(encodeY)


#load test data

load_fn2=str(".\\ENZYMES\\ENZYMEScv{}testtimes{}.mat".format(cvnum,cvtimes) )

data2 = sio.loadmat(load_fn2)
NCIlabelandsequence2=data2['testlabelandsequence']
NCItestsequence=NCIlabelandsequence2[:,1:127]
NCItestlabel=NCIlabelandsequence2[:,0]
NCItestlabel=np.asarray(NCItestlabel,'int64')
labelonehottest=LabelEncoder()
labelonehottest.fit(NCItestlabel)
encodeYtest=labelonehottest.transform(NCItestlabel)
dummylabeltest=np_utils.to_categorical(encodeYtest)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


x_train=NCItrainsequence
x_test=NCItestsequence
y_train=NCItrainlabel
y_test=NCItestlabel

# Padding operation

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
classifier=LogisticRegression(solver='lbfgs',multi_class='ovr',max_iter=1000,verbose=1, n_jobs=1)
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
y_predict2=model.predict(x_test_std)
score2=model.score(x_test_std,y_test)
print("Accuracy", score2)
onetimesvmacc.append(score2)








