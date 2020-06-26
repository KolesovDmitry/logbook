date: 2020-06-26
title: Маскирование облачности для Sentinel-2
tags: Remote Sensing, GEE, Clouds
Category: GEE

## Суть дела

Когда анализируешь космосьемку, то постоянно мучаешься с облаками, которые закрывают
собой все интересное и мешаются при автоматической обработке.

При работе с Sentinel-2 можно использовать грубую маску облачности, которая распространяется
совмесно с самим продуктом: к снимкам добавлен QA канал, содержащий вспмогательную информацию, 
в частности, по нему можно установить, какие пиксели закрыты облаками. Но данная маска очень
грубая и качество ее оставляет желать лучшего. 

Гораздо более качественную маску облачности дает инструмент [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector),
это пакет Python, предназначенный для поиска облачности на снимках Sentinel-2. Он обеспечивает попискельную
классификацию и возвращает растр, в пикселях которого записана вероятность того, что данный пиксель -- облако.

Этот инструмент был разработан на базе машинного обучения, в основе лежит модель градиентного бустинга. Собственно
сама модель [хранится в репозитории](https://github.com/sentinel-hub/sentinel2-cloud-detector/tree/master/s2cloudless/models),
а инструмент представляет собой не более чем обертку над библиотекой lightGBM, которая и использует данную модель.


Все хорошо ровно до тех пор, пока не нужно искать облака в системе, в которую сложновато всунуть посторонюю модель. 
В частности, для того чтобы добавить возможность поиска облаков на базе этой модели в Google EarthEngine,
приходится несколько поднапрячься.

## Как это работает
В GEE встроено много возможностей по созданию моделей машинного обучения, но, к сожалению, там не предусмотрен импорт моделей lightGBM.
Зато в GEE можно подключить модель, написанную на TensorFlow и использовать ее. Поэтому сама собой приходит в голову мысль,
что можно построить на TensorFlow аппроксимирующую модель -- то есть обучаем новую модель повторять ответы, которые выдает
инструмент s2cloudless.

После того, как мы построим на TensorFlow свою модель (в нашем случае -- нейросеть), которая будет с приемлемой точностью
воспроизводить поведение исходной модели, ее можно будет импортировать в GEE и испльзовать, как обычно.
Тут сталкиваемся с двумя альтернативами, каждая из которых со своими сложностями:

 * мы программируем все на языке Python, затем работаем с GEE тажже через API на питоне; но это не удобно для нас -- у нас большая часть работы завязана на веб-интерфейс GEE;
 * мы сохраняем модель TensorFlow, а затем подключаем ее через ee.Model.fromAiPlatformPredictor; это также не удобно -- нужно настраивать AI Platform.

В итоге получается, что быстрее всего будет воспользоваться [готовой реализацией многослойного перцептрона](https://code.earthengine.google.com/d0932076572dd95fffed43b0aa716bc8?noload=true).
Эту реализацию я писал несколько лет назад специально под GEE, когда там еще не было возможности подключать внешние модели.
Поскольку слои перцептрона реализуются довольно просто, достаточно перемножить матрицы весовых коэффициентов и вектор выходов предыдущего слоя сети,
то трехслойная сеть может быть записана в GEE буквально парой десятков строчек кода:

```{javascript}
var basicMLP = {
  init: function (weightsList, biasesList){ 
    this.biases = biasesList.map(array2Image);
    this.weights = weightsList.map(array2Image);
  },
  
  output: function(inputArray){
    /*
      Evaluate NNet output for given input 'inputArray'
      
        inputArray: ee.Array of NNet input values.
    */
    var l1 = inputArray.matrixMultiply(ee.Image(ee.Array(this.weights.get(0))));
    l1 = relu(l1.add(ee.Array(this.biases.get(0))));
    
    var l2 = l1.matrixMultiply(ee.Image(ee.Array(this.weights.get(1))));
    l2 = relu(l2.add(ee.Array(this.biases.get(1))));
    
    var l3 = l2.matrixMultiply(ee.Image(ee.Array(this.weights.get(2))));
    l3 = l3.add(ee.Array(this.biases.get(2)));
    
    return l3;
  }

```


Итак, порядок действий таков:

1. Обучаем перцептрон под TensorFlow.
2. Экспортируем коэффициенты нейронной сети.
3. Импортируем коэффициенты в GEE и подключаем их к многослойному перцептрону.



## Детали реализации
Построение модели на TensorFlow производилось следующим кодом:

```{Python}
import numpy as np
from lightgbm import Booster
import tensorflow as tf

# Читаем модель
bst = Booster(model_file='lightGBM.model')

# генерируем обучающее множество
def batch(count, model=bst, scale=1.0):
    data = np.random.rand(count, 10) * scale
    pred = bst.predict(data)
    return data, pred


# Строим сеть
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(10, )),
    tf.keras.layers.Dense(50, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid', ),
])
net.compile(optimizer='adam', loss='mse', metrics=['mae'])
net.summary()

# Собственно обучение
EPOCHS = 1
X, y = batch(500000000)
_ = net.fit(X, y, epochs=EPOCHS, validation_split = 0.05, verbose=1)
```

В результате была построена следующая сеть и были получены следующие оценки ее точности:
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 50)                550       
_________________________________________________________________
dense_1 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 51        
=================================================================
Total params: 3,151
Trainable params: 3,151
Non-trainable params: 0
_________________________________________________________________
Train on 475000000 samples, validate on 25000000 samples
Epoch 1/1
475000000/475000000 [==============================] - 23026s 48us/step - loss: 0.0049 - mean_absolute_error: 0.0298 - val_loss: 0.0044 - val_mean_absolute_error: 0.0285
```

Таким образом буквально несколько строк позволили построить на базе исходного 
аналогичный детектор облачности, отличающийся от первоначального приблизительно на плюс-минус три процента.

Результирующий инструмент можно посмотреть:

 * собственно [код детектора](https://code.earthengine.google.com/68c182c59bfefe05479539cb7015f9eb?noload=true) и [коэффиценты сети](https://code.earthengine.google.com/fee34a8031983d0d6cb03638012dc7d0?noload=true);
 * [пример использования](https://code.earthengine.google.com/8410e65b4bf52bdc0a6261a9c442ea99);

