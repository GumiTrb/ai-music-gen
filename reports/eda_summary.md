# EDA: ADL Piano MIDI

Сгенерировано `scripts/eda.py`. Графики в `reports/figures/`.

## Жанры

Файлов в архиве: **11087**, принято после фильтров: **8091**.

| исходный жанр | файлов | → класс |
|---|---|---|
| Rock | 3144 | rock |
| Unknown | 2171 | unknown |
| Classical | 1398 | classical |
| Soundtracks | 947 | soundtrack |
| Pop | 750 | pop |
| World | 503 | latin_world |
| Jazz | 492 | jazz |
| Soul | 354 | jazz |
| Latin | 253 | latin_world |
| Country | 247 | country_folk |
| Ambient | 185 | soundtrack |
| Electronic | 161 | pop |
| Rap | 158 | pop |
| Blues | 129 | jazz |
| Folk | 90 | country_folk |
| Religious | 55 | country_folk |
| Reggae | 26 | pop |
| Children | 24 | pop |

| класс | пьес после фильтров | исполнителей |
|---|---|---|
| classical | 617 | 171 |
| rock | 2567 | 1429 |
| pop | 930 | 576 |
| jazz | 759 | 425 |
| soundtrack | 814 | 343 |
| latin_world | 563 | 292 |
| country_folk | 302 | 175 |
| unknown | 1539 | 1189 |

## Фильтрация

| причина | файлов |
|---|---|
| принят | 8091 |
| not_4_4 | 2561 |
| duplicate | 151 |
| too_few_notes | 126 |
| parse_error | 89 |
| too_short | 39 |
| bad_tempo | 25 |
| too_long | 5 |

## Медианы по классам

| класс | тактов | BPM | нот/такт | ср. высота | токенов |
|---|---|---|---|---|---|
| classical | 52 | 110 | 14.0 | 62.9 | 2692 |
| rock | 89 | 110 | 10.3 | 61.0 | 3151 |
| pop | 93 | 112 | 9.7 | 61.3 | 2984 |
| jazz | 94 | 106 | 12.8 | 62.8 | 4210 |
| soundtrack | 56 | 120 | 13.5 | 60.8 | 2568 |
| latin_world | 84 | 102 | 10.1 | 61.4 | 2947 |
| country_folk | 80 | 100 | 10.1 | 60.7 | 2992 |
| unknown | 80 | 113 | 10.4 | 61.6 | 2742 |

## Тональности

Оценка по профилям Крумхансла: мажор 72%, минор 28%. Тональность пользователь не задаёт.

## Токены

Словарь: **190** токенов (тело 156, условия 34). Всего токенов в датасете: **30.0M**, медиана на пьесу 3031, токенов на такт ≈ 40.

| тип | доля |
|---|---|
| PAD | 0.0% |
| BOS | 0.0% |
| EOS | 0.0% |
| BAR | 2.3% |
| POSITION | 12.7% |
| PITCH | 28.3% |
| DURATION | 28.3% |
| VELOCITY | 28.3% |

Окно 512 токенов покрывает ≈ 13 тактов, окно 1024 — ≈ 26.
