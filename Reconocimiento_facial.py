import cv2
import face_recognition as fr


# cargo imagenes para reconocimiento

foto_control= fr.load_image_file('fotoB.jpg')
foto_prueba= fr.load_image_file('fotoA.jpg')


# paso imagenes a rgb
foto_control=cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba=cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#mostrar imagenes
cv2.imshow('Foto Control',foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

#mantener programa abierto

cv2.waitKey(0)
