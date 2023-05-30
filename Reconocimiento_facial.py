import cv2
import face_recognition as fr
import os

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

print(len(lista_empleados_codificada))