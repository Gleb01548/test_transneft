# Test_transneft

Проект реализует векторный и полнотекстовый поиск по набору текстовых данных (/data/data.csv).
А также их комбинацию.

Установка [pdm](https://pdm-project.org/latest/)

Установка зависимостей:
```
pdm install
```

Формирование docker образа:

```docker build -t test_transneft .```

Запуск контейнера:

```docker run --rm -it -p 8890:8888 test_transneft```

Проведение тестов:
```pytest tests/test_app.py```

Взаимодействие осуществляется через API. Пример запроса:

```
data = {
            "query": "Газпром преобразован в акционерное общество",
            "search_engine": "bm25", # возможные значения: vector, bm25, comb
            "k": 2,
        }
res = requests.post("http://0.0.0.0:8890/search", json=data)
```

**query** - Зарос по которому будет поиск

**k** - кол-во документов котор

**search_engine** - движок запроса. Возможные значения:

    - bm25 - полнотекстовый поиск на основе bm25. Алгоритм применяется к нормальзованных текстам (сервис сам их нормализует).

    - vector - косинусное сходство эмбеддингов запроса и текстов в базе. Для получения эмбеддингов использовалась cointegrated/rubert-tiny2, потому что у нее хорошие эмбеддинги и хорошо работает на CPU. 

    - comb - комбинацич bm25 и vector. Работает так: осуществляется поиск обоими методами. Потом переранжируем в зависимости от их ранга. Например получили выдачу документов от bm25: bm25_doc_rank1, bm25_doc_rank2 и от vector: vector_doc_rank1, vector_doc_rank2. Метод comb отранжирует документы следующим образом: bm25_doc_rank1, vector_doc_rank1, bm25_doc_rank2, vector_doc_rank2.

Пример ответа сервиса:
``` 
{
    'status': 'success',
    'time_search': 15.327692031860352,
    'search_result': [
    {'text': '26 июня 1998 Решением собрания акционеров РАО «Газпром» преобразовано в Открытое акционерное общество.',
    'score': 14.1,
    'method': 'bm25'},
    {'text': '17 февраля 1993 Постановлением Правительства РФ во исполнение Указа Президента РФ Государственный газовый концерн «Газпром» преобразован в Российское акционерное общество.',
    'score': 12.64,
    'method': 'bm25'}
    ]
} 
```
**status** - успешно или не успешно завершена работа программы.

**time_search** - время работы сервиса над запросом в миллисекундах.

**search_result** - результат работы алгоритма. Список словарей с информацией о найденных документах, методе который их нашел и скором сходства.

### Особенности скоров
vector - это косинусное сходство при котором скор принимает значения от 0 до 1. Где 0 наименьшее сходство, а 1 - полное сходство.

bm25 - это оценка алгоритма bm25, возможные значения от 0 до бесконечности. Где 0 наименьшее сходство.

### Дополнительные примеры работы алгоритма
Дополнительные примеры работы алгоритма расположены в ноутбуке по пути: notebook/notebook.ipynb