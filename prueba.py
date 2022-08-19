import tensorflow as tf
import numpy as np


#Modelo de aprendizaje automatico

#Declaramos los arreglos que tendrán los datos predeterminados para el aprendizaje del sistema
#Celsius: entradas - Fahrenheit: resultados
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array ([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Implementamos keras para identificar la capa de salida, la cual solo tiene una neurona (capa de tipo densa)
#Anexo a esto, presentamos una capa de entrada la cual tiene una neurona "input_shape=[1]"
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

#Modelo secuencial para redes basicas
modelo = tf.keras.models.Sequential([capa])
#Preparación del modelo para ser entrenado, para el optimizador utilizamos Adam, el cual permite a la red ajustar
#pesos y sesgos de manera eficiente
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
print("Comenzando entrenamiento...")
#Para entrenar la red utilizamos la función fit, las dos variables array que creamos con los datos
#y epoch significa la cantidad de vueltas que el sistema dará a los datos para realizar el aprendizaje
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("¡Entrenamiento completado!")

#Grafica la cantidad de errores con respecto a las vueltas del sistema en su aprendizaje
# import matplotlib.pyplot as plt
# plt.xlabel('Epoch')
# plt.ylabel("Magnitud de error")
# plt.plot(historial.history['loss'])

print("Hagamos una predicción")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " grados Fahrenheit")

print("Variables internas del modelo")
print(capa.get_weights())
