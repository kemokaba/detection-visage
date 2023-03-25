#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv

# charger les classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# charger les images
img = cv.imread('obama.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# exécution de la détection de visage
# detectMultiScale(image, scale factor, number of neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)

# affichage des visages
for face in faces:
    x, y, w, h = face
    
    # dessiner le rectangle sur l'image principale
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
# exécution de la détection des yeux
eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)    
    
# affichage des yeux
for (ex, ey, ew, eh) in eyes:
    # dessiner le rectangle autour des yeux sur l'image principale
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255,0,0), 2)
    

# affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()
