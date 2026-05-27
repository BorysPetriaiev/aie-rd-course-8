# 🧪 Vector DB Benchmark Lab

**Тема:** Lesson 8 · Vector Databases у продакшні

Benchmark 5 векторних БД на реальному датасеті [BeIR/quora](https://huggingface.co/datasets/BeIR/quora) (~523K векторів).  
Мета — побудувати **Pareto-frontier** «recall vs latency» і обрати оптимальну БД для продакшну.

---

## 📋 Вимоги

- Python 3.12+ (рекомендовано) або 3.14
- Docker Desktop (для Qdrant і pgvector)
- ~3 GB вільного місця (датасет + embeddings)
- ~30-60 хв на повний запуск бенчмарку

---

## 🚀 Швидкий старт

### 1. Клонувати репо та створити venv

```bash

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Запустити Docker

```bash
docker compose up -d
docker compose ps   # перевірити що hw_qdrant і hw_pgvector мають статус running
```

### 3. Завантажити датасет BeIR/quora

```bash
python src/load_data.py
```


### 4. Згенерувати embeddings

```bash
python src/embed.py --model BAAI/bge-small-en-v1.5 \
    --input data/corpus.jsonl \
    --output data/embeddings.npy

python src/embed.py --model BAAI/bge-small-en-v1.5 \
    --input data/queries.jsonl \
    --output data/query_embeddings.npy
```

### 5. Запустити бенчмарк

```bash
python src/runner.py --output results/results.csv
```

Запускає всі 5 БД послідовно (~30-60 хв). Параметри:

| Параметр | Опис | Приклад |
|---|---|---|
| `--output` | Шлях до CSV з результатами | `results/results.csv` |
| `--queries` | Кількість queries (default: 1000) | `--queries 500` |
| `--skip` | Пропустити певні БД | `--skip faiss_flat chroma` |

Приклад запуску тільки окремих БД:
```bash
python src/runner.py --output results/results_qdrant.csv --skip faiss_flat faiss_hnsw chroma pgvector
```

### 6. Згенерувати графіки

```bash
python src/plot.py --input results/results.csv --output results/
```

Генерує 4 файли:
```
results/pareto_frontier.png       # recall vs latency
results/latency_distribution.png  # p50 / p95 / p99
results/disk_size_chart.png       # розмір індексу
results/results_table.png         # зведена таблиця
```

### 7. Переглянути результати

```bash
open results/pareto_frontier.png    # macOS
```

### 8. Зупинити Docker

```bash
docker compose down
```

---

## 📊 Результати

| DB | Recall@10 | MRR@10 | p50 ms | p95 ms | p99 ms | Index s | Disk MB |
|---|---|---|---|---|---|---|---|
| faiss_flat  | 0.9308 | 0.8556 | 15.14 | 16.50 | 17.49 | 0.8 | 0.0 |
| faiss_hnsw  | 0.9308 | 0.8567 | 0.27  | 0.37  | 0.42  | 148.5 | 0.0 |
| qdrant      | 0.9308 | 0.8568 | 3.88  | 4.99  | 5.74  | 335.0 | 0.0 |
| chroma      | 0.9288 | 0.8549 | 1.09  | 1.39  | 1.57  | 295.2 | 934.3 |
| pgvector    | 0.9228 | 0.8494 | 2.86  | 5.61  | 7.87  | 705.7 | 1857.1 |

**Висновок:** `faiss_hnsw` — найкращий вибір для продакшну: recall майже ідентичний faiss_flat (baseline), але latency в **55x** менша (0.27ms vs 15ms). Qdrant і Chroma — хороші варіанти якщо потрібне персистентне зберігання.

---

## 📁 Структура репо

```
homework-vector-db-benchmark/
├── README.md
├── requirements.txt
├── docker-compose.yml           # Qdrant + Postgres (pgvector)
├── .env.example
├── data/
│   ├── .gitignore               # *.npy, *.jsonl, *.tsv не комітяться
│   ├── embeddings.ids.json      # ID корпусу (в git)
│   └── query_embeddings.ids.json
├── src/
│   ├── load_data.py             # завантаження BeIR/quora
│   ├── embed.py                 # embedding + cache (.npy)
│   ├── metrics.py               # recall@K, MRR, latency percentiles
│   ├── runner.py                # запуск бенчмарків
│   ├── plot.py                  # генерація графіків
│   └── benchmarks/
│       ├── base.py              # абстрактний VectorDB interface
│       ├── faiss_flat.py
│       ├── faiss_hnsw.py
│       ├── qdrant_db.py
│       ├── chroma_db.py
│       └── pgvector_db.py
└── results/
    ├── results.csv
    ├── pareto_frontier.png
    ├── latency_distribution.png
    ├── disk_size_chart.png
    └── results_table.png
```

---