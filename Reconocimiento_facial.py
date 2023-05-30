import cv2
import face_recognition as fr
import os
import numpy

# creo DB

ruta='Empleados'
mis_imagenes=[]
nombres_empleados=[]
lista_empleados=os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual= cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])


# pasar a rgb las imagenes hardcodeadas, posterior mente codifico con 'fr'
#finalmente pusheo las imagenes a la lista que se retorna al final de la funcion
def codificar_imagenes(imagenes):

    lista_imagenes_codificadas=[]

    for imagen in imagenes:
        imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen_codificada=fr.face_encodings(imagen)[0]
        lista_imagenes_codificadas.append(imagen_codificada)

    return lista_imagenes_codificadas


lista_empleados_codificada= codificar_imagenes(mis_imagenes)

#usar la camara web para capturar mi la imagen a comparar

captura= cv2.VideoCapture(0,cv2.CAP_DSHOW)

# leer la imagen de la camara web

lectura_exitosa, imagen=captura.read()

if not lectura_exitosa:
    print('No se ha podido tomar la captura')
else:
    cara_captura=fr.face_locations(imagen)

    cara_captura_codificada=fr.face_encodings(imagen,cara_captura)

    #Buscar coincidencias

    for cara_codif, cara_location in zip(cara_captura_codificada,cara_captura):
        coincidencias=fr.compare_faces(lista_empleados_codificada,cara_codif)
        distancias=fr.face_distance(lista_empleados_codificada,cara_codif)



        indice_coincidencia=numpy.argmin(distancias)
        print(indice_coincidencia)
        #muestro coincidencias si existe la misma

        if distancias[indice_coincidencia]>0.55:
            print('No coincide con ningun empleado')

        else:
            print(f'Bienvenido {nombres_empleados[indice_coincidencia]}')