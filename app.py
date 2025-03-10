from flask import Flask,redirect,url_for,render_template,request,Response
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
app=Flask(__name__)
cap = cv2.VideoCapture(0)

alert_message = False

def generate_frames():
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    sound2 = mixer.Sound('alert_message.wav')

    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    # model = load_model('models/new_model2.h5')
    model=load_model('modelnew.h5')
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
        
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                count=count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = np.argmax(model.predict(r_eye), axis=-1)
                if(rpred[0]==1):
                    lbl='Open' 
                    print('r-eye is open')
                if(rpred[0]==0):
                    lbl='Closed'
                    print('r-eye is closed')
                break

            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                count=count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = np.argmax(model.predict(l_eye), axis=-1)
                if(lpred[0]==1):
                    lbl='Open'   
                    print('l-eye is open')
                if(lpred[0]==0):
                    lbl='Closed'
                    print('l-eye is closed')
                break
            print(f'left-eye: {left_eye}, right-eye: {right_eye}')
            if(rpred[0]==0 and lpred[0]==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
                
            if(score<0):
                score=0   
            if(score>15):
                try:
                    if score % 5==0:
                        sound.play()
                except:
                    pass
                if(thicc<16):
                    thicc= thicc+2
                else:
                    thicc=thicc-2
                    if(thicc<2):
                        thicc=2
                
                if score>15 and score%30==0:
                    sound2.play()

                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
            
        else:
            cv2.putText(frame,"NoFace",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    return render_template('new.html')

@app.route('/driver')
def driver():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=='__main__':
    app.run()

