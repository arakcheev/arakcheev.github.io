---
layout: post
title: "Действительные числа"
author: "Artem"
category: mathematics,analisys
---

# Действительные числа. 

Математический анализ начинается достаточно необычно для непросвещенного взгляда, а именно с аксиоматического определения множества действительных чисел. И вообще с определения чисел. Действительно, кто задумывался как определить число (ввести определение числа) с точки зрения математики? То, что я пытаюсь сейчас описать не претендует на строгое определение чисел, а должно скорее рассматривается как пример логики мышления. 

Итак, начнем мы с простого. Нам нужно придумать способ счета, что бы понять сколько купить картошки, например. Для этого мы можем ввести цифры: $1,2,3,4,…$. В принципе такой ряд можно продолжать до бесконечности, ну и правда, а кто нас остановит? Таким образом мы интуитивно можем вести некое множество чисел, естественно возникающих при перечислении или счете.  Дадим таким числам название **натуральных чисел**: 

> Множество натуральных чисел $\mathbb{N}$ это числа $1$ и $n+1$, где $n$ - натуральное число.

Кстати, очень интересное рекурсивное определение). Операцию сложения тут тоже можно ввести интуитивно, *2 яблока + 1 яблоко = 3 яблока*. Можно ввести отрицательное число, как такое число, которое в сумме с натуральным дает нуль: 

$$
\mathbf{0} : n + \mathbf{0} = n \\
(-n): n + (-n) = \mathbf{0}
$$

И это тоже может быть интуитивно понятно. Обладая отрицательными числами и нулем мы можем обзавестись множеством **целых чисел**: 

> Множество целых чисел $\mathbb{Z}$ это числа вида где само число является целым, либо отрицательное число является целым, либо нуль, т.е.
> $$
> \mathbb{Z}: n \in \mathbb{Z}, (-n) \in \mathbb{Z}, 0 \in \mathbb{Z}
> $$
>

Потрясающе, теперь мы можем ввести дроби, или **рациональные числа**, как числа показывающие, что мы можем разделить одно яблоко на троих людей, или круг на четыре части: 

> Множество рациональных чисел это числа вида $\frac{p}{q}$, где $p \in \mathbb{Z}, q \in \mathbb{N}$. 

И это тоже понятно, так как на нуль делить нельзя). Очень важным свойством рациональных чисел является их **счетность** и **полнота** (бесконечность), т.е. для каждых двух рациональных чисел, таких что $r_1, r_2: r_1 > r_2$ всегда найдется рациональное число, которое лежит между двумя этими числами - действительное, такое число может быть найдено, и, например, равно $\frac{r_1+r_2}{2}$.  (Я тут пропустил операцию умножения, ну и деление соответсвенно, так как цель этого изложения немного другая. Операции сложения, умножения (и следствие деления), отношение порядка корректно вводятся и доказываются, см например [1] ). 

Я все это веду к тому, что бы поставить интересный вопрос - а как определить **действительное (тут я имею в виду иррациональное)** число? Например, $\sqrt{2} = 1.414221…(\infty \space цифр)$? Для нас это кажется вполне очевидным, но потребовался неочевидный "хак", что бы корректно ввести это понятие. Что же это за математический объект? И нужен ли вообще нам такой математический объект, с бесконечным числом знаков после запятой?  

Сначала небольшой пример, зачем такие объекты вообще нужны. Представим себе квадрат со сторонами в 1 сантиметр. Тогда диагональ квадрата будет равняться корню из двух. Пока все логично, так как теорему Пифагора можно доказать чисто геометрически, без арифметики, и из нее вычислить диагональ квадрата.  Таким образом у нас появилась потребность ввести некий новый класс чисел. Мы пока не знаем как его определить, но знаем где можно использовать такие числа. 

Но это не самая большая проблема. Другая проблема заключается в том, мы привыкли к непрерывным величинам (все мы знаем что действительные числа непрерывны), но что такое это непрерывность и как её определить? Вопрос о том, почему мы сознаем потребность в непрерывных величинах это отдельный разговор, который несет очень фундаментальное значение для математики, физики и философии. Например, а гравитационная сила непрерывна, т.е. она существует везде или только в определенных точках пространства (например только на орбитах планет)? Если она непрерывна, то какими числами можно ее описать? Я сейчас задаю такие провокационные вопросы с целью, что бы задуматься, что нам действительно не хватает такого класса чисел, как действительные (иррациональные числа).     

Этим я хочу показать, что у людей возникла потребность ввести нечто, что не является чисто интуитивно понятным (вряд ли кто то пытался поделить яблоко на корень из семи частей), более того, что является бесконечным (бесконечная точность числа) по своей сути. И это не абстракция вроде эфира или модели атома, а это число. Еще одним из таких примеров - **комплексные числа**, которые вообще не соотносятся с реальностью но с потрясающей детерменированностью описывают тот же микромир. 

Итак, как же можно подойти к действительным числам? Вообще люди давно их использовали, но строгую теорию одним из первых построил Рихард Дедеки́нд. Для введения иррационального числа рассмотрим сначала так называемые сечения в области рациональных чисел. 

Разобьем множество всех рациональных чисел на два непустых множества $M, M^{\prime}$ причем так, что бы  любое число во множестве $M$ было меньше любого числа множества $M^{\prime}$. Таким образом мы можем получить три типа разбиения: 

1. Разбиения типа $ M: \\{ r: r< 1 \\}, \space M^{\prime}: \\{r: r\geq 1\\}$. В нижнем классе $M$ отсутствует верхняя граница а в классе $M^{'}$ присутствует. В нижнем классе мы можем как угодно близко подходить к $1$, но не подойдем к ней никогда (рациональные числа обладают полнотой). А в верхней классе у нас единица достижима. Таким образом в нижнем классе отсутствует наибольшее число.  

2. Обратное разбиение, $ M: \\{ r: r\leq 1 \\}, \space M^{\prime}: \\{r: r > 1\\}$. В этом случае в нижнем классе у нас уже присутствует максимальный элемент, но отсутствует минимальный в верхнем классе. 

3. Ну и наконец рассмотрим третье возможное разбиение $M: \\{ r: r\leq 0 \bigcap r^2 <2 \\}, \space M^{\prime}: \\{r: r^2 > 2\\}$. При таком разбиении у нас нет максимального элемента в нижнем множестве и нет минимального элемента в верхнем. 

В первых двух случаях мы имеем дело с сечением, которое определяет рациональное число, являющееся пограничным (либо максимальным в нижнем либо минимальным в верхнем числом). А в третьем случае пограничного числа не существует, поэтому для сечений такого типа мы и вводим новое понятие - **иррациональное число**. Дедеки́нд условился называть любое сечение третьего типа иррациональным числом. Вместе числа рациональные и иррациональные составляют действительные (вещественные) числа. Итак, вот оно *определение* иррационального числа, через сечения в области рациональных чисел. 

Хорошо, а если рассмотреть подобные сечения в области иррациональных чисел, не появятся ли там новые числа? Другими словами, находясь в рамках рациональных чисел, мы смогли определить сечения во множестве рациональных чисел, такое, что, граница этого сечения не описывается иррациональным числом. Есть ли подобное для иррациональных чисел? Ответ - нет, и это составляет суть одной из основных теорем Дедеки́нда в теории вещественных чисел:

> Для всякого сечения в области вещественных чисел существует вещественное число, которое производит это сечение - это число либо наибольшее в нижнем классе, либо наименьшее в верхнем. 

Это свойство вещественных чисел называют **непрерывностью**. И это просто то, что нам нужно. 

Для иррациональных чисел доказываются теоремы, что все основные операции (сложение, умножение, нуль и так далее, а так же отношения порядка) остаются справедливыми, как и с рациональными числами (было бы странным, если это не было справедливым). После того как строгая теория чисел была все таки построена, в настоящее время используются 16 аксиом вещественных чисел, которые ложатся в основу математического анализа:

1. 4 аксиомы сложения 
2. 4 аксиомы умножения 
3. 4 аксиомы отношения порядка 
4. Аксиома связи сложения и умножения, сложения и отношения порядка, умножения и отношения порядка 
5. Аксиома непрерывности. 

Какие можно слетать выводы? Идея достаточно простая, потребовалось немало времени, что бы математики ввели, а уже потом строго аксиоматезировали привычные нам действительные числа. Поему я посчитал это важным? Потому что мало кто задумывается о таких очевидных вещах как **что такое число**, или  **что такое время и пространство** - а ведь зачастую их можно формально описать вывести немало полезных и интересных свойств. А закончить хочу одной знаменитой цитатой.  

> Почему именно я создал теорию относительности? Когда я задаю себе такой вопрос, мне кажется, что причина в следующем. Нормальный взрослый человек вообще не задумывается над проблемой пространства и времени. По его мнению, он уже думал об этой проблеме в детстве. Я же развивался интеллектуально так медленно, что пространство и время занимали мои мысли, когда я стал уже взрослым. Естественно, я мог глубже проникать в проблему, чем ребёнок с нормальными наклонностями.
>
> ​																**Альберт Эйнштейн**

[1]: Фихтенгольц Г.М. Курс дифференциального и интегрального исчисления. Том 1
[2]: https://ru.wikipedia.org/wiki/Вещественное_число	(Интересно почитать историю)
[3]: https://ru.wikipedia.org/wiki/Конструктивные_способы_определения_вещественного_числа



