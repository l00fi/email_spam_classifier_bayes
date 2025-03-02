# Наивный Байес для классификации спама
## План проекта
- [x] Написать наивный байесовский классификатор (пока с нормальным ядром);
- [x] Достать данные (буду парсить собственную почту);
- [x] Сформировать dataset для работы;
- [ ] Перписать написанные классификатор для спама/не спама.
## Отчет
- \[27.02.2025\]: Наивный  байес написал по этой [статье](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Training)
- \[27.02.2025\]: Данные из почты достал с помощью Google Архиватор - удобно, можно выбрать что выгрузить и в каком формате. Но создание архива может занять вплоть до нескольких дней... 
- \[28.02.2025\]: Сырые данные в формате mbox привел к json с помощью утилиты [mbox-to-json](https://github.com/PS1607/mbox-to-json), информативная часть находится в словарях по ключу body. Теперь нужно пройтись по содержимому, очистить его от html и прочего мусора и токенизировать оставшееся для дальнейшего формирования датасета. 
- \[28.02.2025\]: Датасет сформирован, в нём хранятся частоты появления каждого слова в спаме/не спаме. Для этого пришлось создать отдельный скрипт make_dataset.py, там я выделил только информативный текст и накинул на каждый метку спам/не спам, дальше посчитал частоты. Осталось только реализовать специфичный классификатор.
- \[02.03.2025\]: Реализован класс SpamClassifier, accuracy на test - 98%, но это без кросс-валидации. Её стоит сделать. Также надо привести весть код в нормальный вид (например весь модуль make_dataset) стоит переписать как класс. Также нужно подумать над более гибкой реализации работы классификатора, достаточно много привязано к названию столбцов и тексту меток (то есть если в метку попадёт "спам" вместо "Спам", то модель сломается). Пока это все замечания которые имею к своей же работе, в дальнейшем могут появится новые.   