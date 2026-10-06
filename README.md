# ai-music-gen — управляемая генерация фортепианной MIDI-музыки

Цель проекта: пользователь задаёт **жанр, длину (в тактах) и темп** и получает фортепианную пьесу в MIDI.
С нуля обучаются и сравниваются n-gram (бейзлайн), LSTM, GRU и decoder-only Transformer.

План по неделям — [PLAN.md](PLAN.md), ход работы — [JOURNAL.md](JOURNAL.md).

**Текущий этап: неделя 1 — данные и токенизатор.**

## Данные

[ADL Piano MIDI](https://github.com/lucasnfe/adl-piano-midi): 11 087 фортепианных MIDI в 18 жанрах
(структура архива: `Жанр/Поджанр/Исполнитель/Название.mid`).
После фильтров (размер 4/4 или 2/2, ≥ 8 тактов, ≥ 64 нот, без дубликатов) остаётся **8 091 пьеса, 30M токенов**.
Мелкие жанры объединены в 7 классов + `unknown`:

| класс | исходные жанры ADL | пьес |
|---|---|---|
| rock | Rock | 2567 |
| pop | Pop, Rap, Electronic, Reggae, Children | 930 |
| soundtrack | Soundtracks, Ambient | 814 |
| jazz | Jazz, Blues, Soul | 759 |
| classical | Classical | 617 |
| latin_world | World, Latin | 563 |
| country_folk | Country, Folk, Religious | 302 |
| unknown | Unknown (жанр не размечен) | 1539 |

Полная сводка и графики — [reports/eda_summary.md](reports/eda_summary.md), [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

## Токенизация (REMI, свой токенизатор)

```
BOS GENRE_jazz TEMPO_88 LENGTH_64 | BAR POSITION_0 PITCH_55 DURATION_15 VELOCITY_8 PITCH_58 ... BAR ... EOS
└──────── условия (префикс) ─────┘ └─────────────────────── тело пьесы ───────────────────────────┘
```

| тип | кол-во | смысл |
|---|---|---|
| `BAR` | 1 | начало такта |
| `POSITION` | 16 | позиция в такте (шаг 1/16) |
| `PITCH` | 88 | высота (клавиши фортепиано 21–108) |
| `DURATION` | 32 | длительность, 1–32 шестнадцатых |
| `VELOCITY` | 16 | громкость (velocity // 8) |
| `TEMPO` | 16 | бин темпа (48–208 BPM, ближайший в лог-шкале) |
| `LENGTH` | 10 | бин длины (8–256 тактов) |
| `GENRE` | 8 | жанровый класс |
| служебные | 3 | `PAD`, `BOS`, `EOS` |

Итого **190 токенов**. Условия стоят в конце словаря, поэтому id музыкальных токенов не зависят от списка жанров.
Условия берутся из MIDI: темп — BPM, который звучит дольше всего; длина — число тактов (ведущие пустые такты отрезаются).

Round-trip MIDI → токены → MIDI точный с точностью до сетки 1/16 и бина громкости — проверяется тестами,
в том числе на каждом сотом файле реального датасета.

## Как запустить

```bash
pip install -r requirements.txt
pip install -e .                        # пакет musicgen из src/

python scripts/analyze_dataset.py       # скачать ADL (37 МБ) и разобрать все файлы, ~1 мин
python scripts/eda.py                   # графики и сводка в reports/
python -m pytest                        # тесты токенизатора
jupyter notebook notebooks/01_eda.ipynb # демо: EDA и MIDI -> токены -> MIDI
```

## Структура

```
configs/data.yaml        источник данных, группировка жанров, фильтры
src/musicgen/
  data/vocab.py          словарь токенов и условий
  data/tokenizer.py      MIDI <-> REMI-токены
  data/adl.py            датасет: скачивание, обход архива, разбор файлов, фильтры
  utils.py
scripts/                 analyze_dataset.py, eda.py
notebooks/01_eda.ipynb   EDA и демо токенизатора
reports/                 сводка EDA и графики
samples/week_1/          оригинальный и восстановленный MIDI
tests/                   тесты токенизатора
```

Данные (`data/`) в git не хранятся — скачиваются скриптом.
