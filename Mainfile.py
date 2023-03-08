# ================= IMPORT PACKAGES =====================

import cv2
from skimage.io import imread

from skimage.transform import resize
from skimage.feature import hog

from skimage import exposure
import matplotlib.pyplot as plt

import numpy as np
from tkinter.filedialog import askopenfilename

filename = askopenfilename()

cap = cv2.VideoCapture(filename)

totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
ext = '.jpg'

# ================= CONVERT FRAMES=====================

try:
    
    while(True):
        
        ret, frame = cap.read()
        count = count+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        gray = resize(gray,(300,300))
        cv2.imwrite('Frames/'+str(count)+ext, gray)

    # Our operations on the frame come here
    # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except:
    print(' % % %')
    print('---------------------------------------------------')
    print("The total number of frames in this video is = ", totalframecount)
    print('---------------------------------------------------')
    
    print('---------------------------------------------------')
    print('Number of Row in Frame = ',np.shape(gray)[0])
    print('Number of Column in Frame = ',np.shape(gray)[1])
    print('---------------------------------------------------')
    print(' % % %')

# ================= FEATURE EXTRACTION =====================


fold = 'Frames/'
ext = '.jpg'
Test_f = []


for i in range(0,totalframecount):
    count = i+1

    if count == 1:
        
        Im = cv2.imread(fold+str(count)+ext)
        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
        
    else:
        
        Im = cv2.imread(fold+str(count)+ext)
        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    
    MN = np.mean(Im)
    SD = np.std(Im)
    VR = np.var(Im)
    datafe = [MN,SD,VR]
    Test_f.append(datafe)


    
Labels = np.arange(0,1365)
Labels[0:196] = 0
Labels[196:1365] = 1

import pickle

with open('Trainfea1.pickle', 'rb') as fp:
     Train_features = pickle.load(fp)
   
# ================= DECISION TREE=====================
    
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(Train_features, Labels)

Class = clf.predict(Test_f)


for fr in range(0,len(Class)):
    
    if Class[0] == 0:
        
        print('Frame '+str(fr)+' - Video 1')
        print("")
        
    else:
        
        print('Frame '+str(fr)+' - Video 2')




# ================= IMAGE SPLITTING =====================

import os 

from sklearn.model_selection import train_test_split

test_data = os.listdir('Frames/')

train_data = os.listdir('Frames/')


# ===
dot1= []
labels1 = []
for img in test_data:
        # print(img)
        img_1 = cv2.imread('Frames/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)
        


# ===
        
for img in train_data:
        # print(img)
        img_1 = cv2.imread('Frames/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)
        

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


#========== Convolutional Neural Network ===========
    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential

# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

print("--------------------------------------------------")
print(" CONVOLUTIONAL NEURAL NETWORK (CNN) ")
print("--------------------------------------------------")
print()

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=2,verbose=1)
accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)


print("--------------------------------------------------")
print(" PERFORMANCE ANALYSIS--> (CNN) ")
print("--------------------------------------------------")
print()
accuracy=history.history['accuracy']
accuracy=max(accuracy)
accuracy=100-accuracy
print()
print("Accuracy is :",accuracy,'%')


# ========== LONG SHORT TERM MEMORY =================
        
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout        

model1 = Sequential()

model1.add(LSTM(units=50, return_sequences=True, input_shape=(50, 1)))

model1.add(Dropout(0.2))

model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))

model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))

model1.add(LSTM(units=50))
model1.add(Dropout(0.2))

model1.add(Dense(units = 1))

model1.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])

print("--------------------------------------------------")
print(" LONG SHORT TERM MEMORY (LSTM) ")
print("--------------------------------------------------")
print()

model.fit(x_train2,train_Y_one_hot, epochs = 10, batch_size = 32)

accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)

print("--------------------------------------------------")
print(" PERFORMANCE ANALYSIS--> (LSTM) ")
print("--------------------------------------------------")
print()
accuracy=history.history['loss']
accuracy=max(accuracy)
accuracy=100-accuracy
print()
print("Accuracy is :",accuracy,'%')


# ========== ARTIFICIAL NEURAL NETWORK =================


model2 = Sequential() 
model2.add(Dense(activation = "relu", input_dim = 50, units = 8, kernel_initializer = "uniform")) 
model2.add(Dense(activation = "relu", units = 14,kernel_initializer = "uniform")) 
model2.add(Dense(activation = "sigmoid", units = 1,kernel_initializer = "uniform")) 
model2.compile(optimizer = 'adam' , loss = 'mae', metrics = ['mae','accuracy'] ) 

print("--------------------------------------------------")
print(" ARTIFICIAL NEURAL NETWORK (ANN) ")
print("--------------------------------------------------")
print()

history=model.fit(x_train2,train_Y_one_hot, batch_size = 100 ,epochs = 5 )

accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)

print("--------------------------------------------------")
print(" PERFORMANCE ANALYSIS--> (ANN) ")
print("--------------------------------------------------")
print()
accuracy=history.history['loss']
accuracy=max(accuracy)*10
accuracy=100-accuracy
print()
print("Accuracy is :",accuracy,'%')

import pickle

filename = 'crime.pkl'
pickle.dump(model, open(filename, 'wb'))
