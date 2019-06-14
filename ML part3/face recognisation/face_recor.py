import cv2
import numpy as np
import os

def distance(v1,v2):
	return np.sum((v2-v1)**2)**.5

def knn(train,test,k=5):
    dist = []

    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(test,ix)
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x: x[0])[:k]
    labels =np.array(dk)[:,-1]
    output = np.unique(labels,return_counts=True)
    index=np.argmax(output[1])
    return output[0][index]

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

dataset_path = "./faceData/"
labels=[]
class_id=0
names={}
face_data=[]


for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id] = fx[:-4]
        print("loading file",fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        print(data_item)
        #create labels
        target = class_id*np.ones((data_item.shape[0]))
        labels.append(target)
        class_id+=1


#cv2.puttext() to print text 
X= np.concatenate(face_data,axis=0)
Y= np.concatenate(labels,axis=0).reshape((-1,1))


print(X.shape)
print(Y.shape)

trainset = np.concatenate((X,Y),axis=1)
print(trainset.shape)
#while True:
# #predection
while True:
	ret,frame = cam.read()
	if ret ==False:
		print("try again!! something went wrong")
		continue
	
	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		cv2.imshow("video",frame)
		continue
        #cv2.imshow('Face Recorgnisation',frame)
	for face in faces:
		x,y,w,h=face
		face_section=frame[y-10:y+h+10,x-10:x+w+10]
		face_section= cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		face_section=face_section.reshape((30000,))
		#face_section=face_section.reshape((face_section.shape[0],-1))
		#face_section=face_section.reshape((face_section.shape[0],-1,))

		user_name = int(knn(trainset,face_section))
		user_name=names[user_name]
		print(user_name)
		"""if user_name =='0':
			user_name='arnav'
		elif user_name =='1':
			user_name='Ayushi'
		elif user_name =='2':
			user_name='chirag'
		elif user_name =='3':
			user_name='Rishabh'
		else:
			user_name='Splasher'"""
		#print(user_name)
		cv2.putText(frame,user_name,(x,y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,255,255),fontScale=2,thickness=2)
	cv2.imshow('video',frame)
cam.release()
cv2.destroyAllWindows()