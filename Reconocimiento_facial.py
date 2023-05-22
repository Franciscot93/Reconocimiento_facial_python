import cv2
import face_recognition as fr


# cargo imagenes para reconocimiento

foto_control= fr.load_image_file('fotoB.jpg')
foto_prueba= fr.load_image_file('fotoA.jpg')


# paso imagenes a rgb
foto_control=cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba=cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)


# rotacion y redimencion de imagenes

foto_control1=cv2.rotate(cv2.resize(foto_control,(600,480)),cv2.ROTATE_90_COUNTERCLOCKWISE)
foto_prueba1=cv2.rotate(cv2.resize(foto_prueba,(600,480)),cv2.ROTATE_90_COUNTERCLOCKWISE)

#localizacion cara
lugar_cara_A=fr.face_locations(foto_control1)[0]
cara_codificada_A= fr.face_encodings(foto_control1)[0]

lugar_cara_B=fr.face_locations(foto_prueba1)[0]
cara_codificada_B= fr.face_encodings(foto_prueba1)[0]



#mostrar rectangulo de control
cv2.rectangle(foto_control1,
              (lugar_cara_A[3],lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2])
              ,(0,255,0),
              2)

cv2.rectangle(foto_prueba1,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2])
              , (0, 255, 0),
              2
              )

#realizar comparacion
resultado=fr.compare_faces([cara_codificada_A],cara_codificada_B,0.45)



#medida de la distancia
distancia=fr.face_distance([cara_codificada_A],cara_codificada_B)


#mostrar resultado

cv2.putText(foto_prueba1,
            f'{resultado} {distancia.round(2)}',
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2)


#mostrar imagenes
cv2.imshow('Foto Control',foto_control1)
cv2.imshow('Foto Prueba', foto_prueba1)


#mantener programa abierto

cv2.waitKey(0)
