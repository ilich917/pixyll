---
layout:     post
title:      Tarea 1
date:       2021-06-01 12:31:19
summary:    Tarea 1 del curso Deep Learning, con videos y material
categories: jekyll pixyll
---

#### Tarea_1_CC6204_2020

Archivo original en:
    [https://colab.research.google.com/drive/1aeuSRjj_kQ_uFEBSJ9bRuyr4G4MY4FAi](#)

Tarea 1: Activaciones y pasada hacia adelante en una red neuronal CC6204 Deep Learning, Universidad de Chile 

[Hoja de Respuestas (con tests automáticos)](https://drive.google.com/file/d/1Xj_0rvpf3zXV69A9xWlohrL8068IrbqZ/view?usp=sharing)

En las primeras tareas del curso **progamarás a mano** varios aspectos de redes neuronales Feed Forward. La idea es familiarizarse con tensores, funciones de activación, derivadas, el algoritmo de backpropagation, algoritmos de optimización, regularización, entrenamiento, y búsqueda de hiperparámetros. No se espera obtener excelentes resultados en problemas de clasificación reales, si no más bien aplicar los conceptos teóricos aprendidos en clases y así entenderlos de manera más precisa.  En esta primera tarea sólo nos preocupará la función `forward` de una red neuronal y usaremos una red pre-entrenada para el conjunto de datos MNIST. 

Te recomendamos que comiences por familiarizarte un poco con [tensores de pytorch](https://pytorch.org/docs/stable/tensors.html) y sus operaciones, que son el objeto básico que usaremos en esta y las siguientes tareas. El material necesario para resolver esta tarea es el siguiente:
* [Video: Perceptrón, funciones de activación, y representación matricial](https://www.youtube.com/watch?v=mDCxK2Pu0mA) 
* [Video: MLP, redes feed-forward, y función de salida](https://www.youtube.com/watch?v=eV-N1ozcZrk&t=1710) (desde el minuto 28:30)
* [Apuntes de Redes Feed Forward](https://github.com/jorgeperezrojas/cc6204-DeepLearning-DCCUChile/raw/master/2019/clases/apuntes/1_FFNN.pdf)

IMPORTANTE: A menos que se exprese lo contrario, sólo podrás utilizar las clases y funciones en el módulo [`torch`](https://pytorch.org/docs/stable/torch.html). Hay excepciones explicadas en el enunciado de la tarea más adelante. 

(por Jorge Pérez, https://github.com/jorgeperezrojas, [@perez](https://twitter.com/perez))
"""

Este notebook está pensado para correr en CoLaboratory. 
Lo único imprescindible por importar es torch 
import torch

Posiblemenete quieras instalar e importar ipdb para debuggear.
Si es así, descomenta lo siguiente:

{% highlight python lineanchors %}
# !pip install -q ipdb
# import ipdb
{% endhighlight %}

## Parte 1: Funciones de activación y función de salida

En esta parte programarás varias funciones que serán de utilidad cuando construyas tu red neuronal. Una cosa **muy importante en esta y las siguientes partes**: evita los loops (`for`, `while`, etc.) a toda costa! todo lo que se pueda hacer con operaciones de tensores sin iterar será muy eficiente (en CPU y GPU).

## 1a) Funciones de activación

En esta parte debes programar las siguientes funciones de activación:

*   `relu`, que para cada valor $x$ en un tensor computa el máximo entre $0$ y $x$,  
*   `swish`, propuesta en el artículo [Searching for Activation Functions](https://arxiv.org/abs/1710.05941), y
*   `celu`, propuesta en el artículo [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).

En cada caso tu función debe recibir un tensor (de cualquier cantidad de dimensiones) y entregar otro tensor con la función aplicada a todos sus elementos. La aplicación de las funciones debe ser *punto a punto*, por lo que el tensor de salida de cada función debe tener las mismas dimensiones que el tensor de entrada. **Importante**:  tanto `swish` como `celu` tienen un parámetro que puede modificarse durante el entrenamiento de una red que utilice estas funciones de activación por lo que para estas funciones además del tensor debes recibir el parámetro correspondiente. 

Como ejemplo, estas son implementaciones de las funciones `sig` y `tanh`.

{% highlight python lineanchors %}

def sig(T):
  return torch.reciprocal(1 + torch.exp(-1 * T))

def tanh(T):
  E = torch.exp(T)
  e = torch.exp(-1 * T)
  return (E - e) * torch.reciprocal(E + e)

# Tu código acá

def relu(T):
  pass

def swish(T, ...):
  pass

def celu(T, ...):
  pass

{% endhighlight %}

## 1b) Softmax

En esta parte debes programar la función `softmax`. Esta es una función tal que para una secuencia de valores $(x_1,\ldots,x_n)$  el resultado de $\text{softmax}(x_1,\ldots,x_n)$ es otra secuencia $(s_1,\ldots,s_n)$ que cumple con
\begin{equation}
s_i = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
\end{equation}
Para esto primero demuestra que si a cada elemento de $(x_1,\ldots,x_n)$ se le resta el mismo valor, entonces el resultado de `softmax` no varía. Es decir que $\text{softmax}(x_1-M,\ldots,x_n-M)=\text{softmax}(x_1,\ldots,x_n)$. Usa este hecho para programar una versión de `softmax` que primero le resta a todos los elementos el máximo valor de la secuencia. Esta nueva versión debiera ser numéricamente más estable.

Tu función debe recibir un tensor y el resultado de `softmax` debiera calcularse sobre alguna dimensión del tensor dejando todas las demás dimensiones fijas. Para esto tu función debe también recibir el parámetro `dim` (que indica la dimensión). El resultado de `softmax` se calculará para cada secuencia de valores obtenidos recorriendo la dimensión `dim` fijando las otras dimensiones. Por ejemplo, si `softmax` recibe un tensor de dos dimensiones $T_{ij}$ y se elige la dimensión $1$, entonces se debe computar $\text{softmax}(T_{i1},\ldots,T_{in})$ para cada $i$. Note que en este caso se calculó `softmax` sobre la dimensión $1$ (la segunda dimensión) dejando fija la dimensión $0$ (la primera dimensión). Como otro ejemplo, si el input de la función es un tensor de tres dimensiones $T_{ijk}$ y se elige la dimensión $2$ (tercera dimensión), entonces se debe computar $\text{softmax}(T_{ij1},\ldots,T_{ijn})$ para cada par $i$, $j$. Por si estás familiarizado con `numpy`, el uso de `dim` en esta parte es muy similar al de `axis` en `numpy`.

Nota que el resultado de `softmax` es siempre un tensor de las mismas dimensiones de la entrada.

(La demostración puedes entregarla en otro archivo o incluirla directamente en una celda de la [hoja de respuestas](https://drive.google.com/file/d/1NANjiWP7fWyRBWOf2Pp2gkpRkXheB54s/view?usp=sharing))
"""

# Tu código acá

def softmax(T, dim, estable=True):
  pass

"""# Parte 2: Red neuronal y pasada hacia adelante (forward)

En esta parte empezaremos a programar nuestra red neuronal, en particular la pasada hacia adelante para una red que resolverá problemas de clasificación con varias clases. Supondremos que cada capa se verá de la forma
\begin{equation}
h^{(\ell)} = f^{(\ell)}(h^{(\ell-1)} W^{(\ell)}+b^{(\ell)})
\end{equation}
y que la predicción final estará dada por
\begin{equation}
\hat{y} = \text{softmax}(h^{(L)}U+c).
\end{equation}

(Para entender los detalles de estas fórmulas puedes ver los [apuntes de redes feed forward](https://github.com/jorgeperezrojas/cc6204-DeepLearning-DCCUChile/raw/master/2019/clases/apuntes/1_FFNN.pdf).)

## 2a) Clase para red neuronal

Programa una clase `FFNN` que en su inicializador reciba los siguientes parámetros:

*   Cantidad de neuronas de la capa de entrada `F`
*   Lista de cantidades de neuronas en cada capa escondida `l_h`
*   Lista de funciones de activación `l_a`
*   Cantidad de neuronas de la capa de salida `C` (`C` $\geq 2$)

En pytorch, todas las redes neuronales que construyamos deben construirse como subclases de [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module), lo que tendrá varias ventajas que explicaremos después. El inicio de tu código debe verse más o menos así:

```python
class FFNN(torch.nn.Module):
  def __init__(self, ...):
    super(FFNN, self).__init__()
```

El inicializador de tu clase debería crear todos los parámetros para la red como tensores (`torch.tensor`) de las dimensiones correspondientes, y almacenar lo necesario para poder computar la pasada hacia adelante (siguiente parte). Para poder aprovechar las funcionalidades de pytorch, debes **registrar** los parámetros como tales y para esto debes usar la clase [`torch.nn.Parameter`](https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter). Por ahora, para nosotros un parámetro de la red definido con `torch.nn.Parameter` no es nada mas que un tensor y en esta tarea lo usaremos como eso, es decir, utilizando solo las funciones básicas de [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html). El siguiente trozo de código crea un tensor de 100 x 100 con solo zeros y lo registra como parámetro en la red.

```python
class FFNN(torch.nn.Module):
  def __init__(self, ...):
    super(FFNN, self).__init__()
    T = torch.zeros(100,100)
    self.mi_parametro = torch.nn.Parameter(T)
```

Inicializa todos los parámetros con números aleatorios pequeños y los sesgos como 0. No olvides definir los parámetros como artibutos de la clase para que otros métodos de la clase tengan acceso a ellos también. En tu caso, como tendrás una cantidad variables de parámetros, debes usar una [torch.nn.ParameterList](https://pytorch.org/docs/stable/nn.html#torch.nn.ParameterList) para agregar los parámetros, que estos queden todos registrados y que las otras funciones tengan acceso a ellos.

Una llamada de ejemplo para crear un objeto de tu clase es:
```python
red_neuronal = FFNN(300,[50,30],[relu,sig],10)
``` 
lo que debiera crear todos los parámetros para una red con 300 neuronas en la capa de entrada, luego una capa escondida de 50 neuronas con activación relu, luego una capa con 30 neuronas y activación sigmoid y finalmente una capa de 10 neuronas de salida. <br><br>

Puedes agregarle al inicializador de tu clase todos los parámetros opcionales que estimes conveniente, y construir las funciones que te parezcan importantes de construir. Por ejemplo, es recomendable tener alguna forma de asignar los valores iniciales de los parámetros de la red, lo que servirá por ejemplo para cargar redes pre-entrenadas, inicializar los valores de manera más efectiva, o para hacer debugging del código. También puedes pedir los valores iniciales de los parámetros adicionales de las funciones `celu` y `swish`  (no olvides registrarlos como parámetros también!).
"""

# Tu código debiera comenzar así (ojo que acá solo empieza el código, 
# lo iremos completando más adelante).

class FFNN(torch.nn.Module):
  def __init__(self, F, l_h, l_a, C):
    super(FFNN, self).__init__()
    pass

"""## 2b) Iterando por los parámetros

Uno de los puntos positivos de construir una red en pytorch desde `torch.nn.Module` y registrar los parámetros usando `torch.nn.Parameter` es que puedes tener acceso a un "iterador" sobre todos los parámetros llamando a `.parameters()` (esto será muy importante más adelante cuando veamos el algoritmo de backpropagation). Usa este iterador para mostrar/imprimir un resumen de los parámetros de tu red. Debes mostrar al menos las dimensiones de cada uno de los parámetros. Puede serte útil usar el iterador `.named_parameters()` también



"""

# Tu código debiera continuar así

class FFNN(torch.nn.Module):
  def __init__(self, F, l_h, l_a, C):
    super(FFNN, self).__init__()
    pass
  
  def resumen(self):
    # usa self.parameters() o self.named_parameters()
    pass

"""## 2c) Moviendo los parámetros de la red entre dispositivos

Otro de los puntos positivos de construir una red en pytorch desde `torch.nn.Module` y registrar los parámetros usando `torch.nn.Parameter` es que puedes mover tu red completa entre la CPU y la GPU de manera muy simple. De hecho, como subclase de `torch.nn.Module` podemos usar el método `.to(device)`, de la siguiente forma. Si haces esto:
```python
red_neuronal = FFNN(300,[50,30],[relu,sig],10)
red_neuronal.to('cuda')
```
tu red pasa automáticamente a la GPU, lo que significa que todos los parámetros de la red quedan efectivamente almacenados en la memoria de la GPU. Para devolver los parámetros a la CPU puedes simplemente hacer
```python
red_neuronal.to('cpu')
```
También puedes usar directamente los métodos `.cuda()` y `.cpu()` para lograr los mismo efectos descritos anteriormente.

En esta parte no debes programar nada nuevo, solo comprueba que todo anda bien ejecutando las llamadas anteriores y comprobando que efectivamente la memoria de la GPU se utiliza cuando haces la llamada `to.('cuda')`.

"""

# Crea una red (idealmente grande), y verifica que puedes pasar
# todos los parámetros a la GPU ejecutando !nvidia-smi para chequear
# la cantidad de GPU-RAM utilizada.

"""## 2d) Pasada hacia adelante

Programa la pasada hacia adelante de tu red neuronal en el método `forward` de la clase `FFNN`. La función debiera recibir un tensor de dimensiones `(B,F)` como entrada donde `B` es el tamaño del mini paquete de ejemplos pasados a tu red, y `F` la cantidad de *features* de cada ejemplo. Para computar la pasada hacia adelante, tu red debiera usar los parámetros creados en el inicializador y las funciones de activación entregadas también en el inicializador. Al finalizar, tu red debiera generar predicciones en la forma de probabilidades, aplicando la función `softmax`. El resultado del último `softmax` debiera ser un tensor de dimensiones `(B,C)` que es lo que la función debe retornar (donde `C` representa la cantidad de clases a clasificar).
"""

# Tu código debiera continuar así 

class FFNN(torch.nn.Module):
  def __init__(self, F, l_h, l_a, C):
    super(FFNN, self).__init__()
    pass
  
  def resumen(self):
    # Usa self.parameters() o self.named_parameters().
    pass
  
  def forward(self, x):
    # Usa los parámetros y funciones de activación.
    # El valor de retorno debiera ser y = softmax(capa_de_salida).
    pass

"""# Parte 3: Probando tu red con parámetros pre-entrenados para MNIST

En esta parte usarás la pasada hacia adelante de tu red con parámetros de una red pre-entrenada. La red fue entrenada con el conjunto de datos MNIST que contiene datos de dígitos escritos a mano. La versión original de los datos junto con una descripción del conjunto y resultados para distintos métodos de clasificación, se pueden encontrar en http://yann.lecun.com/exdb/mnist/

## 3a) Cargando y visualizando datos de MNIST

Esta parte no requiere que escribas código, sólo que te familiarices con el conjunto de datos. Sólo sigue las instrucciones.
Primero usaremos el paquete `torchvision` (más algunos otros utilitarios) para descargar y procesar los datos de MNIST.
"""

# Importamos MNIST desde torchvision.
from torchvision.datasets import MNIST

# Importamos una función para convertir imágenes en tensores.
from torchvision.transforms import ToTensor

# Importamos funcionalidades útiles para mirar los datos.
from matplotlib.pyplot import subplots
from random import randint

# Descarga y almacena el conjunto de prueba de MNIST.
dataset = MNIST('mnist', train=False, transform=ToTensor())
print('Cantidad total de datos:',len(dataset))

"""Los datos en todo dataset de pytorch se pueden acceder indexándolos como si fueran un arreglo. En el caso de MNIST cada dato es un par que contiene un tensor `T` y un entero `l`, en donde `T` representa a la imágen de un dígito, y `l` representa el valor numérico de ese dígito. Exploremos el primero de estos datos:"""

T, l = dataset[0]

print('Tensor')
print('tipo:', T.type())
print('dimensiones:', T.size())
print()
print('Entero')
print('valor:', l)

"""El tensor `T` representa una imagen de 28x28 pixeles. Nota que el dato en cuestión tiene una dimensión inicial (es un tensor de 1x28x28). Esto es porque la imagen que estamos considerando está en blanco y negro, por lo tanto tiene un solo canal de color (más adelante usaremos imágenes generales con tres canales y que por lo tanto serán representadas con tensores de dimensiones 3xHxW).

El siguiente código muestra el contenido de estos tensores de manera más amigable. Elige tres posiciones al azar y  muestra el valor `l` y el tensor `T` dibujado. Nota como se usa `view(28,28)` para redimensionar el tensor (sacarle la primera dimensión). En este caso también se usa `.numpy()` para pasar el tensor a un formato más amigable para graficar.
"""

# Muestra algunos ejemplos al azar
n_ejemplos = 3
fig, axs = subplots(nrows=n_ejemplos, figsize=(2,n_ejemplos*3))

for i in range(n_ejemplos):  
  idx = random.randint(0,len(dataset))
  T, l = dataset[idx]
  img = T.view(28,28).numpy()
  axs[i].set_title("clase: "+ str(l))
  axs[i].imshow(img)

# Note que se usó `view` para redimensionar el tensor, esto porque nuestro
# dataloader entrega un tensor de dimensiones (1,1,28,28).
# Es muy importante tener este hecho en cuenta en la siguiente parte.

"""## 3b) Cargando los parámetros pre-entrenados

En [este link](https://github.com/dccuchile/CC6204/tree/master/2020/tareas/tarea1/mnist_weights) encontrarás varios archivos de texto que representan los parámetros de una red con 2 capas escondidas que fue pre-entrenada para clasificar los datos de MNIST. La red pre-entrenada tiene esta arquitectura

784 --> 32 (relu) --> 16 (relu) --> 10 (softmax)

Nota que la cantidad de neuronas en la capa de entrada es 28*28 = 784, esto porque nuestras redes esperan un vector de características como input.

Los archivos de parámetros están nombrados como `W1`, `b1`, `W2`, `b2`, `U` y `c` que representan, respectivamente, a $W^{(1)}$, $b^{(1)}$, $W^{(2)}$, $b^{(2)}$, $U$ y $c$ en la descripción genérica que hemos utilizado para nuestras redes neuronales.

Supongamos que ya tenemos guardado el archivo `W1.txt`. Para convertir estos archivos en tensores de pytorch, puedes hacer algo como lo siguiente:

```python
from numpy import loadtxt
W1 = torch.from_numpy(loadtxt('W1.txt')).float()
```

Usa lo anterior para crear una red con la arquitectura descrita y cargar todos los parámetros pre-entrenados en la red. Esto debes hacerlo llamando a un método de inicialización de pesos de tu red. Si no hiciste ese método en las partes anteriores, es el momento de implementarlo.
"""

# Tu código acá

"""## 3c) Cálcula la predicción de un ejemplo al azar

Prueba con un código tan simple como puedas, la predicción que entrega tu red para un ejemplo al azar del conjunto de datos y muestra también la imágen y la clase real del ejemplo.
"""

# Tu código acá

"""## 3d) Pasando todos los ejemplos por la red con un `DataLoader`

Un `DataLoader` en pytorch es una manera muy útil de entregarle paquetes de ejemplos a una red. Será especialmente útil cuando estemos entrenando. Por ahora lo usaremos sólo para computar la predicción de la red pre-entrenada y calcular el porcentaje de acierto.

Para crear un `DataLoader` solo se debe especificar el conjunto de datos que se usará en la forma de un objeto `DataSet`, y el tamaño del paquete de cada paquete que usaremos. En el siguiente código estamos creando un `DataLoader` desde nuestro objeto `dataset` con paquetes de tamaño 100.

```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=100)
```

Una vez creado podemos iterar por todo el dataset haciendo simplemente:

```python
for x, y in dataloader:
  # lo que necesitemos hacer con los datos
```

En cada iteración `x` será un tensor con 100 ejemplos, por lo tanto tendrá dimensiones 100 x 1 x 28 x 28, e `y` será un tensor con las clases correspondientes a cada uno de esos ejemplos, por lo tanto tendrá dimensión 100 (es un tensor con 100 valores enteros).

Escribe una función que use un `DataLoader` para pasar todos los ejemplos por la red en paquetes y calcula el porcentaje total de acierto que la red obtiene en la predicción. Recuerda que el porcentaje total de acierto es la cantidad de ejemplos clasificados correctamente dividido por la cantidad de ejemplos. Tu función debe recibir a la red con los parámetros cargados, el dataset que usarás, el tamaño del batch para pasar por la red y si el trabajo debe hacerse en la GPU o en la CPU. Aprovecha de probar como varía el tiempo de ejecución de tu función si cambias el tamaño del paquete y si usas la GPU vs la CPU.
"""

# Acá tu código
def calcula_acierto(red, dataset, batch_size=100, device='cuda'):
  pass

"""## 3e) Opcional: Muestra los casos en donde la red se equivoca

Muestra imágenes de 5 casos en donde la red se equivoca en la predicción (muestra la imagen y el dígito que la red predice). ¿Es razonable el error que comete?
"""

# Acá tu código

"""## 3d) Opcional: Crea tus propios ejemplos de dígitos para clasificar

Usa el código en [este link](https://colab.research.google.com/drive/1pdoj2grwFUNa7ZTY5TPefsedY0VDW2_4#scrollTo=8K6u9gS-JXIT) para generar nuevos casos de prueba manualmente y ver cómo lo clasifica la red. Trata de entender en qué casos comete errores.
"""

# Acá tu código