import numpy as np
import pandas as pd
import scipy.io as io
import os
def sampling(tmp):
    t  = np.zeros(250)
    id = np.argmax(tmp)
    t[100] =tmp[id]
    idx = np.random.random_sample(100)*id
    idx = [int(a) for a in idx]
    idx.sort()
    t[:100]=[tmp[a] for a in idx]

    idx = np.random.random_sample(150-1)*id
    idx = [int(a) for a in idx]
    idx.sort()
    t[101:]=[tmp[a] for a in idx]
    return t
class_dir = ['Electrical axis left side','Left bundle branch block beat',
             'Left ventricular hypertrophy','Normal','Right bundle branch block beat'
             ,'Sinus-bradycardia']
def mat2dict():
    DATA={}
    for cl in class_dir:
        
        print 'AT class_dir=%s'%cl
        files =  os.listdir('./DataECG-Train-EN/%s/'%cl)
        newdata=[]
        for f in files:
            t=io.loadmat('./DataECG-Train-EN/%s/%s'%(cl,f))
            t=t['Beats'][0][0]
            #label =t[1]
            data=t[3][0]
            for i in range(len(data)):
                tmp=data[i][:,0]
                tmp=sampling(tmp)
                newdata.append(list(tmp))
        DATA[cl]=newdata
    return DATA
data = mat2dict()
##transfer dictionary to pandas
data_p=[]
for cl in data.keys():
    if len(data_p)==0:
        data_p=pd.DataFrame (data[cl])
    else:
        data_p=pd.concat([data_p,pd.DataFrame (data[cl])])


        
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
pca.fit(data_p)
data_trans = pca.transform(data_p)
data_trans =pd.DataFrame (data_trans )
print(pca.explained_variance_ratio_)

label = []
for cl in data.keys():
    [label.append(cl) for a in range(len(data[cl]))]
        
data_trans['label']=label

flag  = np.random.random(len(data_p))<0.7
train =data_trans[flag]
test  =data_trans[~flag]
train.groupby (['label']).count()[0]
test.groupby (['label']).count()[0]
# learning
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
gnb = GaussianNB()
svc = SVC(C=50.0,gamma=3,kernel='rbf')
rfc = RandomForestClassifier(n_estimators=100)

svc.fit(train.drop(['label'],axis=1),train['label'])
predicted = svc.predict (test.drop (['label'],axis=1))
test_target =test['label']
from sklearn import metrics
print(metrics.classification_report(test_target, predicted) )
print metrics.confusion_matrix(test_target, predicted)



from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data_trans = data_trans.values
def visual2D():
    '''
    for i in range(10 ):
       plt.plot(pca.components_[i,:])
       plt.show()
'''
    
    fig = plt.figure()
    color=['g','b','k','r','y','c']
    i=0;start=0
    def update(frame):
        s=50*np.random.random(2072)
        for scat in root:
          scat.set_sizes(s)
        return root
    root=[]
    for cl in data.keys():
      leng = len(data[cl])
      scat=plt.scatter (data_trans[start:start+leng,0],data_trans[start:start+leng,1],c=color[i],s=50*np.random.random(leng),alpha=0.5)
      root.append(scat)
      i+=1;start=start+leng
    animation = FuncAnimation(fig, update, interval=10)
    plt.show()
    animation.save('../rain.gif', writer='imagemagick', fps=30, dpi=72)
def visualPair():
    
    
    color=['g','b','k','r','y','c']
    i=0;start=0
    Normal = len(data[class_dir[1]])+len(data[class_dir[2]])+len(data[class_dir[3]])
    len_N  = len(data[class_dir[4]])
    for cl in data.keys():
      leng = len(data[cl])
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')  
      scat1=ax.scatter (data_trans[start:start+leng,0],data_trans[start:start+leng,1],data_trans[start:start+leng,2],c=color[i])
      scat2=ax.scatter (data_trans[Normal:Normal+len_N,0],data_trans[Normal:Normal+len_N,1],data_trans[Normal:Normal+len_N,2],c='r')
      s1=50*np.random.random(1000);s2=50*np.random.random(1000);
      scat1.set_sizes(s1)
      scat2.set_sizes(s2)
      i+=1;start=start+leng
      plt.show()

def visual3D():
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color=['g','b','k','r','y','c']
    i=0;start=0
    def update(frame):
        s=50*np.random.random(2072)
        for scat in root:
          scat.set_sizes(s)
        return root
    root=[]
    for cl in data.keys():
      leng = len(data[cl])
      scat=ax.scatter (data_trans[start:start+leng,0],data_trans[start:start+leng,1],data_trans[start:start+leng,2],c=color[i])
      root.append(scat)
      i+=1;start=start+leng
    animation = FuncAnimation(fig, update, interval=10)
    plt.show()
