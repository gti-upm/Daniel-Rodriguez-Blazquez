# Desarrollo de un sistema de detección del nivel de música sobre voz en señales de audio basado en redes neuronales

</br>

## Resumen

La distinción entre voz y audio lleva siendo durante muchos años un objeto de estudio en el campo de la Inteligencia Artificial. Este tema ha llevado a diversos trabajos centrados en la distinción de géneros musicales, detección del pitch o el reconocimiento de instrumentos entre otras cosas. Además, existen otros trabajos como la recomendación de música y la distinción de locutores que empiezan a jugar un papel importante en un mundo cada vez más digital y autónomo. Sin embargo, en este campo existen una serie de limitaciones ya que el hecho de distinguir automáticamente música y habla no es una tarea fácil de realizar por las similitudes que ambas señales comparten. Por lo tanto, para abordar la mayoría de los trabajos son necesarias técnicas de desarrollo avanzadas como pueden ser el Machine Learning o el Deep Learning.  

En este trabajo se va a proponer una solución para distinguir en una señal de audio que combina habla y música, cual es el nivel relativo de música respecto a la voz. Una posible aplicación de este trabajo puede ser el reconocimiento de segmentos de música con derechos de autor en transmisiones de televisión y radio, donde puede haber música de fondo en una conversación, entrevista, debate, etc.

Para lograr lo recién mencionado se va a diseñar y entrenar una red neuronal para distinguir el nivel de musica sobre voz que hay en una señal de audio. Para ello se creará una base de datos con segmentos de voz y música mezclados con distintos niveles de música. Esta base de datos se pre-procesará para obtener espectrogramas, los cuales son señales bidimensionales susceptibles de ser analizadas por redes convolucionales. Finalmente, se realizarán experimentos para comprobar la capacidad de aprendizaje y de éxito que dicha red es capaz de lograr.  

</br>

## Sistema operativo y librerías

  * Linux

    

  * Tensorflow

  * Keras

  * matplotlib.pyplot

  * librosa

  * numpy

  * pandas

</br>

## Base de datos y estructura

Para llevar a cabo la distinción del nivel de música con respecto al de voz en un segmento de audio (voz y música), se necesita transformar las bases de datos de voz y música a un pre-procesado de forma que cree una nueva base de datos en forma de espectrogramas para la tarea que se va a realizar.

Para la creación de esta base de datos se siguieron una serie de pasos mostrados en el siguiente diagrama de bloques:

![1](/home/drb/code/Daniel-Rodriguez-Blazquez/figs/Diagrama bloques pre-procesado.png)



* Base de datos de voz

Esta base de datos es [ZeroSpeech](https://download.zerospeech.com). En esta página web, se encuentran disponibles tres archivos: dos para la parte de entrenamiento (training) y una para la parte de test.  Las correspondientes a la parte de entrenamiento son dos: la primera es la empleada como base de datos de voz en nuestro modelo y la segunda es una muestra reducida de la base de datos anterior. El tercer archivo dedicado a la parte de test fue descartado ya que contenía segmentos de voz en un idioma distinto al inglés.  

En este conjunto de muestras de voz existen un total de 23.576 canciones con un tamaño total del archivo de 2.5 GB. En la base de datos escogida existen una serie de subcarpetas a la hora de descargarse el archivo, las cuales en mi caso han sido mezcladas para crear una única carpeta con 23.576 canciones.



* Base de datos de música

Se ha escogido la base de datos [fma](https://github.com/mdeff/fma) debido a que es una de las fuentes de datos más conocidas a la hora de realizar proyectos con técnicas de Machine Learning y música. Esta base de datos se encuentra disponible en.  En este caso, al ser un trabajo de carácter exploratorio, se ha decidido hacer uso de una de las versiones intermedias, fma_medium, con 25.000 canciones de 30 segundos y un tamaño de 22 GB, de forma que sea coherente con magnitud de la fuente de datos de voz (23.576 canciones).

</br>

## Desarrollo

</br>

Tras la creación de la base de datos y haciendo uso de una arquitectura InceptionV3 para la red neuronal, se procede al entrenamiento para que la red sea capaz de extrapolar los ‘conocimientos’ necesarios para que el sistema pueda realizar predicciones correctas sobre el valor en decibelios de la diferencia de volumen entre música y voz. Un ejemplo del proceso general del sistema es el mostrado en la siguiente figura:

![2](/home/drb/code/Daniel-Rodriguez-Blazquez/figs/Captura de pantalla 2019-06-04 a las 16.19.34.png)

A la salida del sistema y gracias a la función de activación softmax, se obtienen 10 valores junto con sus correspondientes probabilidades, generando como único valor de salida finalmente, el valor cuya probabilidad sea mayor (entre 1 y 10 dB).

</br>

# Pasos a seguir para la ejecución

<br>

Una vez descargadas y organizadas las bases de datos y los archivos disponibles en el repositorio, el proceso a seguir es el siguiente:

\* Se procede a crear la base de datos de mel-espectrogramas -> ejecutar el script dataset_preprocess.py  

\* Se lleva a cabo el entrenamiento -> ejecutar el script train.py

\* Se lleva a cabo el test -> ejecutar el script test.py

</br>