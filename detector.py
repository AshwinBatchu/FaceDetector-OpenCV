import cv2 as cv 

guy = cv.imread('Photos/faker.jpg')
cv.imshow('Person', guy)

grey = cv.cvtColor(guy, cv.COLOR_BGR2GRAY )
cv.imshow('Grey Image', grey)

haar = cv.CascadeClassifier('\\haar_face\\')
faces = haar.detectMultiScale(grey,1.1,3)
for(x,y,w,h) in faces:
    cv.rectangle(guy,(x,y),(x+w,y+h),(0.255,0),2)
cv.imshow("Final Detected Product With Green Box",guy)
cv.waitKey(0)
