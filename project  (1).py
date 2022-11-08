import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
def face_detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame,'face',(x,y),font,1,(255,20,0),2)
    return frame
def eye_detect(gray, frame):
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.putText(frame,'eye',(ex,ey),font,1,(255,20,0),2)
    return frame
def body_detect(gray, frame):
    body = body_cascade.detectMultiScale(gray, 1.4, 4)
    for (bx, by, bw, bh) in body:
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        cv2.putText(frame,'body',(bx,by),font,1,(255,20,0),2)
    return frame
def smile_detect(gray, frame):
    smile = smile_cascade.detectMultiScale(gray, 1.8, 5)
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        cv2.putText(frame,'smile',(sx,sy),font,1,(255,20,0),2)
    return frame
video_capture = cv2.VideoCapture(0)
n=int(input("Enter your choice : \n--> Press 1 if you want to detect FACE.\n--> Press 2 if you want to detect EYES.\n--> Press 3 if you want to detect FULL BODY.\n--> Press 4 if you want to detect SMILE."))
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if n==1:
        canvas = face_detect(gray, frame)
        cv2.imshow('Video', canvas)
    elif n==2:
        canvas = eye_detect(gray, frame)
        cv2.imshow('Video', canvas)
    elif n==3:
        canvas = body_detect(gray, frame)
        cv2.imshow('Video', canvas)
    elif n==4:
        canvas = smile_detect(gray, frame)
        cv2.imshow('Video', canvas)
           
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()






