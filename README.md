# Наивный Байес для классификации спама
## План проекта
- [x] Написать наивный байесовский классификатор (пока с нормальным ядром);
- [x] Достать данные (буду парсить собственную почту);
- [ ] Сформировать dataset для работы;
- [ ] Перписать написанные классификатор для спама/не спама.
## Отчет
- \[27.02.2025\]: Наивный  байес написал по этой [статье](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Training)
- \[27.02.2025\]: Данные из почты достал с помощью Google Архиватор - удобно, можно выбрать что выгрузить и в каком формате. Но создание архива может занять вплоть до нескольких дней... 
- \[28.02.2025\]:Сырые данные в формате mbox привел к json, информативная часть находится в словарях по ключу body. Теперь нужно пройтись по содержимому, очистить его от html и прочего мусора и токенизировать оставшееся для дальнейшего формирования датасета. 