date: 2020-02-20
title: A threshold selection method from gray-level histograms
tags: Raster Processing
Category: Review


**Обзор методов статьи 
[Otsu, N. A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man, Cybern. 9, 62–66](http://webserver2.tecgraf.puc-rio.br/~mgattass/cg/trbImg/Otsu.pdf)**

## О чем пойдет речь

В статье авторы рассматривают изображение в градациях серого и задаются вопросом о том, как привести его к бинарному виду.
Например, изображение может представлять собой результат классификации, когда "черные" пиксели относятся к одному классу, а "белые" к другому, и возникает вопрос,
как автоматически провести границу между классами, т.е. как выбрать такой порог яркости, ниже которого пиксели будут
считаться первым классом, а выше -- вторым.


## Что предлагается

### Обозначения

Пусть $L$ - число возможных градаций серого на изображении ([1, 2, ..., $L$]), $n_i$ -- число пикселей для каждой градации и  $N=n_1, + \dots + n_L$.

Пусть $p_i = n_i/N$ - частоты, с которыми встречаются соотвествующие яркости пикселей.

Пусть $k$ -- граница между классами (искомый порог), тогда $C_0$ -- пиксели с яркостями не превышающими порога ([1, 2,... $k$]), а $C_1$ -- пиксели с яркостями выше порога ([$k+1$, ... $L$]).


### Логика рассуждений

Разобьем изображение на два класса в соответствии с порогом $k$. Для пикселей каждого класса по отдельности мы можем расчитать частоту их встречаемости, а также выборочные средние и дисперсии.

Частоты:
$$
\omega_0(k) = \sum_{i=1}^k p_i,
$$
$$
\omega_1(k) = \sum_{i=k+1}^L p_i.
$$


Средние по классам:
$$
\mu_0(k) = \sum_{i=1}^{k} i Pr(i| C_0),
$$

$$
\mu_1(k) = \sum_{i=k+1}^{L} i Pr(i| C_1).
$$
Общее среднее:
$$
\mu_1(k) = \sum_{i=1}^{L} i p_i.
$$



Дисперсии
$$
\sigma_0(k) = \sum_{i=1}^{k}(i-\mu_0)  Pr(i| C_0),
$$
$$
\sigma_1(k) = \sum_{i=k+1}^{L}(i-\mu_1)  Pr(i| C_1).
$$


Далее предлагается воспользоваться подходом, похожим на те, что используются в дискриминантном и кластерном анализах: нужно выбрать такую величину $k$, которая позволит максимально отделить
друг от друга классы $C_0$ и $C_1$. Другими словами выбирать порог следует таким образом, чтобы внутриклассовая дисперсия была как можно меньше по сравнению с общей дисперсией или межклассовой дисперсией.

Авторы рассматривают три критерия и показывают, что они эквивалентны между собой (для краткости записи опустим функциональную зависимоть дисперсий от $k$):

$$
\lambda(k) = \frac{\sigma^2_b}{\sigma^2_w} \to \max_{1\leq k \leq L},
\qquad \kappa(k) = \frac{\sigma^2_t}{\sigma^2_w} \to \max_{1\leq k \leq L},
\qquad \eta(k) = \frac{\sigma^2_b}{\sigma^2_t} \to \max_{1\leq k \leq L}, 
$$

где $\sigma_b$, $\sigma_w$ и $\sigma_t$ -- соответственно межклассовая, внутриклассовая и общая дисперсии, которые можно вычислить по формулам (вывод формул опускаю):

$$
\sigma^2_b = \omega_0\omega_1(\mu_1 - \mu_2)^2,
$$
$$
\sigma^2_w = \omega_0\sigma_0^2 + \omega_1\sigma_1^2,
$$
$$
\sigma^2_t = \sum_{i=1}^L (i-\mu_t)^2.
$$

Авторы показывают, что задача поиска максимума по данным критериям в конечном итоге эквивалентна задаче максимизации величины $\sigma_b$ и, таким образом,
оптимальный порог $k$ должен быть выбран таким, чтобы он давал максимум величины $\sigma_b$.

## Несколько замечаний

Данный подход выглядит интересным, но он подразумевает, что на изображении присутствуют пиксели двух классов, другими словами, он будет хорошо работать в случае бимодальной
функции распределения яркостей пикселей изображения. (Авторы приводят способ расширить подход на несколько классов, а также дают пример бинаризации изображения с унимодальной функцией распределения
яркостей, но все же, все же...)

Поэтому у меня есть некоторые сомнения, о том, насколько хорошо ляжет данный подход на наши задачи. Вообще задача бинаризации растровых изображений у нас возникает часто,
но на данную статью я наткнулся, разбирая [работу по мониторингу пожаров]({filename}/2020-02-07-Fire-Monitoring.md). В задаче мониторинга предлагаемый подход должен сработать
по организации эксперимента -- там мониторятся заведомо изменившиеся участки и на изображении будут заведомо присуствовать два класса (гарь/не-гарь).

В нашем же случае мониторинга лесоизменений ситуация чуть сложнее -- мы мониторим участки, которые (а) могут вообще не менятся (б) на них могут присутвовать различные естественные изменения (например, листопады).
Поэтому данный подход может "сбоить" в наших условиях. Но тем не менее, его обязательно нужно попробовать, т.к. даже если он и будет требовать ручной доводки оператором, все равно полуавтоматический
выбор пороговой дает какую-то отправную точку для дальнейшего поиска оптимальной величины.


