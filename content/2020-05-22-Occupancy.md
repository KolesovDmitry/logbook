date: 2020-05-22
title: Monitoring carnivore populations at the landscape scale: occupancy modelling of tigers from sign surveys 
tags: Occupancy models
Category: Review


**Обзор основных методов статьи
Karanth, K. U., Gopalaswamy, A. M., Kumar, N. S., Vaidyanathan, S., Nichols, J. D., & MacKenzie, D. I. (2011). Monitoring carnivore populations at the landscape scale: occupancy modelling of tigers from sign surveys. Journal of Applied Ecology, 48(4), 1048-1056.**


## Суть задачи
Нужно создать методику оценки ареала вида по собранным следам присутствия. 
В статье методика рассматривается на примере популяции тигров для тестового участка, расположенного в Индии.



## Организация сбора данных

Исследуемая территория находится в центральной части Западных Гат (горная цепь на западе Индостана). Территория обследовалась
в течении 15-ти месяцев и 2021 человеко-дней, было пройдено 4174 км, обнаружено 403 следов присутствия тигров.

Территория была поделена на ячейки квадратной формы такого размера, чтобы ячейчка была несколько больше типичного размера "домашнего" участка тигра. Обычно в подобных исследованиях проводятся полевые работы в отдельных ячейках, а потом производится экстраполяция результатов на остальные ячейки, но в данной работе были обследованы все ячейки территории.

Тигры выходят на дороги и в течении дня проходят по ним от 1 до 20-ти километров. Поэтому полевые группы также следовали вдоль дорог в поисках следов присутствия тигров (собственно следы, съеденные тиграми животные и пр.). Во избежание дублирования следы присутствия одного типа фиксировались 
единожды в пределах стометрового участка дороги, т.е. если на стометровом участке были найдены, к примеру, два отпечатка
лап, то эти отпечатки заносились в список находок как одна находка. Таки образом с каждым стометровым сегментом дороги
были связаны метки 0 (не найдено) и 1 (есть следы). Помимо того на каждом сегменте отмечалось присутсвие следов видов-жертв и следов присутствия человека, в последствии эти данные использовались в качестве дополнительных переменных модели.

## Вычисления

В качестве основной модели использовалась модель с явным указанием пространственной зависимости
в собираемых данных, эта модель приводится в [2].


Пусть $\psi$ -- доля территории, занятой тиграми, т.е. искомый параметр, а $p$ -- вероятность обнаружения тигра
в ячейке, при условии, что он находится в ней.

Классическая модель, представленная в [3], подразумевает, что обнаруженные/не обнаруженные в ячейках тигры могут
быть описаны в виде схемы Бернулли. То есть получается, что каждый стометровый отрезок представляет собой
одну попытку в схеме Бернулли. Но поскольку тигры проходят по дорогам значительные расстояния, то такой "наивный" подход
нарушает требование независимости повторных экспериментов в модели (схеме). Поэтому дополнительно к данной модели 
строилась еще одна, аналогичная приведенной в работе [2]. В этой модели в явном виде вводится пространственная 
зависимость между наблюдениями, для этого постулируется, что система "тигр-наблюдатель" может быть описана
в виде Марковского процесса: вероятность обнаружения тигра на текущем сегменте зависит от обнаружения его на предыдущем.

Таким образом модель в целом следует подходу, описанному в статье [2] эту статью я уже [разбирал ранее]({filename}/2019-12-09-Occupancy.md). Однако на построении данной модели дело не закончилось, далее авторы:

* сравнили две модели (классическая на базе [3] и модель на базе [2]), используя критерий AIC;
* занялись выявлением основных эколого-географических факторов, связанных с тиграми.

В рамках этого обзора данная часть не рассматривается.




# Литература

[1] Hines, J. E. 2006. PRESENCE2: software to estimate patch
occupancy and related parameters. U.S. Geological Survey,
Patuxent Wildlife Research Center, Laurel, Maryland, USA.
hhttp://www.mbr-pwrc.usgs.gov/software/presence.html

[2] Hines, J.E., Nichols, J.D., Royle, J.A., MacKenzie, D.I., Gopalaswamy, A.M.,
Kumar, N.S. & Karanth, K.U. (2010) Tigers on trails: occupancy modeling
for cluster sampling. Ecological Applications, 20, 1456–1466.

[3] MacKenzie, D.I., Nichols, J.D., Lachman, G.B., Droege, S., Royle, J.A. &
Langtimm, C.A. (2002) Estimating site occupancy rates when detection
probabilities are less than one. Ecology, 83, 2248–2255.

