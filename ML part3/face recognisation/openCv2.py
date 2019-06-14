import cv2
import numpy as np

#readvideo steam and display it


cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data=[]
#cnt=0

user_name = input('enter your name')

while True:
    ret,frame = cam.read()

    if ret ==False:
        print("wrong")
        continue

    key_pressed =cv2.waitKey(1)&0xFF                #Bitmasking to get last 8 bits 
    if key_pressed == ord('q'):                     # ord-->ASCII Value(8bit)
        break

    #cv2.imshow('Video Title',frame[::,::,0])
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    print(faces)
    if(len(faces)==0):
        cv2.imshow('Video Title',frame)
        continue
    for face in faces:
        x,y,w,h =face

        face_section = frame[y-10:y+h+10,x-10:x+w+10]
        face_section= cv2.resize(face_section,(100,100))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        #if cnt%2==0:
        print('taking picture')
        face_data.append(face_section)



    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    """new_img = np.zeros((*gray.shape,3))
    new_img[::,::,0]=gray
    new_img[::,::,1]=gray
    new_img[::,::,2]=gray
    cv2.imshow('gray Title',gray)
    cv2.imshow('Video Title',frame)

    combined = np.hstack((frame,new_img))
    cv2.imshow('combined',combined)"""

    #bright_image = frame+30

    #harcascate model

    cv2.imshow('Video Title',frame)
    cv2.imshow('Video ',face_section)
    

#save the data to numpy file
print("total faces",len(face_data))
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))

np.save("faceData/"+user_name+".npy",face_data)

print(face_data.shape)

cam.release()
cv2.destroyAllWindows()