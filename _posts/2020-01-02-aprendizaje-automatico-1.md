---
layout:     post
title:      Aprendizaje Automático 1
date:       2020-01-02 12:31:19
summary:    Backpropagation y otros grandes temas del aprendizaje automático
categories: Curso Deep Learning
youtubeId1:  G4dnRSSC6Kw
youtubeId2:  1EUAoM1EhM0
youtubeId3:  Gp2rY7LvTyQ
youtubeId4:  pLUNS_tK-K8
youtubeId5:  e_1lis8ByyI
youtubeId6:  y6aD4WG-rOw
---

#### Clase 4: Descenso de Gradiente para encontrar los parámetros de una red
Una vez que logramos hacer fluir nuestros tensores desde el input hasta el output, ¿cómo lo hacemos para la red aprenda? Aquí veremos uno de los métodos más usados: el descenso de gradiente.

Spoiler: tal vez quieras repasar tus apuntes de cálculo, pero si eres totalmente nuevo en esto te instamos a seguir adelante con el curso, complementando tu formación con lo que necesites a medida que avanzas.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::egg::egg:
{% include youtubePlayer.html id=page.youtubeId1 %}


#### Clase 5: Introducción a Backpropagation
Si ya sabemos cómo modificar un parámetro dependiendo de qué tanto condiciona el output, es momento de aplicar el descenso de gradiente de manera iterativa, capa tras capa pero en reversa, para ajustar todos los parámetros de la red. Esa técnica, amigos míos, se conoce como backpropagation.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::egg::egg:
{% include youtubePlayer.html id=page.youtubeId2 %}

#### Clase 6: Continuación Backpropagation
Ya vimos el fundamento matemático del backpropagation, ahora estudiaremos más a fondo cómo lo hace un framework de deep learning internamente.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::egg::egg:
{% include youtubePlayer.html id=page.youtubeId3 %}

#### Clase 7: Tensores, Notación de Einstein, y Regla de la Cadena Tensorial
Trabajar con tensores de muchas dimensiones puede llegar a ser un verdadero dolor de cabeza, por eso es importante conocer las bases de la derivación tensorial, para que cuando hagas cosas geniales con el Deep Learning sepas desenvolverte bien con la literatura y el código que necesites.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::hatching_chick::hatching_chick:
{% include youtubePlayer.html id=page.youtubeId4 %}

#### Clase 8: Entropía Cruzada y Backpropagation a mano con Tensores
Cuando estamos haciendo predicciones usando machine learning, podemos imaginarlo como dos probabilidades, una es la real (que sólo podemos percibir a través de los datos) y la otra es la que aproxima mi red neuronal (a partir de sus parámetros). El concepto de entropía cruzada es una métrica que nos permitirá evaluar la precisión de un modelo y está estrechamente ligada con la estimación por máxima verosimilitud, que seguramente has visto en estadística. 
Finalmente, haremos una pasada de backpropagation a mano, tras lo cual es probable que sientas mucha alegría de que existan frameworks como Tensorflow o Pytorch.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::hatching_chick::egg:
{% include youtubePlayer.html id=page.youtubeId5 %}

#### Clase 9: Red FF a mano en pytorch (y la versión estilo pytorch)
En esta clase usaremos Goggle Colab para para construir una red neuronal a mano en Pytorch.

Dificultad: :hatching_chick::hatching_chick::hatching_chick::hatching_chick::egg:
{% include youtubePlayer.html id=page.youtubeId6 %}

#### Tarea 2 
[Ir al Notebook](https://colab.research.google.com/drive/1-obk_k_xCowFHc5n5JDqqfXZd6EXNN3u)