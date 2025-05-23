## Трудности установки
Сделана на Pade, протестировано на Python 3.7. Библиотека давно не обновлялась, так что лучше использовать его.
На Windows легче всего установить Python 3.7 через conda, на официальном сайте Python теперь не получится удобно скачать.

Предлагаю такую последовательность:  
- создаём окружение conda с Python 3.7:  
`conda create -n for_pade python=3.7`
- создали окружение, после чего его нужно активировать, например, так:  
`conda activate for_pade`
- но Conda не найдет нужные библиотеки, поэтому лучше использовать pip:  
`pip install -r requirements.txt`

Для запуска нужно выполнить в консоли/терминале `pade start-runtime task1_2.py`.  
После этого можно два раза нажать enter.

Используйте CTRL + C для завершения программы.

## Про сами задачки

Константы для задачки можно поменять в начале файла.

Есть опция показывать граф, который генерируется.

Для задачи 1, 2 среднее находится таким образом: вершины обмениваются друг с другом информацией о значениях в соседних вершинах, каждая считает среднее по отдельности по данным, которые у неё имеются. 

Настоящее среднее выводится до того, когда появляется график, в этот момент его можно увидеть и запомнить.

Для задачи 1 нужно просто установить `CONNECTION_PROBABILITY = 1` и `INTERFERENCE_STD = None`.
