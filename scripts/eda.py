"""Неделя 1: анализ датасета (EDA). Графики -> reports/figures/, сводка -> reports/eda_summary.md.

    python scripts/eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from musicgen.data.vocab import Vocab
from musicgen.utils import resolve

FIG = resolve("reports/figures")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def main():
    d = resolve("data/processed")
    df = pd.read_csv(d / "pieces.csv", keep_default_na=False, low_memory=False)
    for c in ("bpm", "n_bars", "n_notes", "notes_per_bar", "mean_pitch", "n_tokens"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    kept = df[df["reject"] == ""].copy()
    vocab = Vocab.load(d / "vocab.json")
    counts = np.load(d / "token_counts.npy")
    genres = [g for g in vocab.genres]
    md = ["# EDA: ADL Piano MIDI\n", "Сгенерировано `scripts/eda.py`. Графики в `reports/figures/`.\n"]

    # 1. исходные жанры и итоговые классы
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    raw = df["genre_raw"].value_counts()
    ax[0].barh(raw.index[::-1], raw.values[::-1], color="#7a8ba6")
    ax[0].set_title("Исходные жанры ADL (все файлы)")
    cls = kept["genre"].value_counts().reindex(genres)
    ax[1].barh(cls.index[::-1], cls.values[::-1], color="#3b6ea5")
    ax[1].set_title("Итоговые классы (после фильтров)")
    save(fig, "01_genres.png")
    md += ["## Жанры\n", f"Файлов в архиве: **{len(df)}**, принято после фильтров: **{len(kept)}**.\n",
           "| исходный жанр | файлов | → класс |", "|---|---|---|"]
    gmap = df.drop_duplicates("genre_raw").set_index("genre_raw")["genre"]
    md += [f"| {g} | {n} | {gmap[g]} |" for g, n in raw.items()]
    md += ["", "| класс | пьес после фильтров | исполнителей |", "|---|---|---|"]
    tab = kept.groupby("genre").agg(n=("path", "size"), artists=("artist", "nunique")).reindex(genres)
    md += [f"| {g} | {r.n} | {r.artists} |" for g, r in tab.iterrows()]
    md.append("")

    # 2. отбраковка и размеры такта
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    rej = df["reject"].replace("", "принят").value_counts()
    ax[0].bar(rej.index, rej.values, color="#a56b3b"); ax[0].set_title("Фильтрация файлов")
    ax[0].tick_params(axis="x", rotation=30)
    ts = df.loc[df["error"] == "", "time_signatures"].value_counts().head(10)
    ax[1].bar(ts.index, ts.values, color="#7a8ba6"); ax[1].set_title("Размеры такта (топ-10)")
    ax[1].tick_params(axis="x", rotation=30)
    save(fig, "02_filtering.png")
    md += ["## Фильтрация\n", "| причина | файлов |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rej.items()] + [""]

    # 3. длина, темп, плотность по жанрам
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    data = [kept.loc[kept["genre"] == g, "n_bars"] for g in genres]
    ax[0].boxplot(data, tick_labels=genres, showfliers=False); ax[0].set_title("Длина, тактов")
    data = [kept.loc[kept["genre"] == g, "bpm"] for g in genres]
    ax[1].boxplot(data, tick_labels=genres, showfliers=False); ax[1].set_title("Темп, BPM")
    data = [kept.loc[kept["genre"] == g, "notes_per_bar"] for g in genres]
    ax[2].boxplot(data, tick_labels=genres, showfliers=False); ax[2].set_title("Нот на такт")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    save(fig, "03_length_tempo_density.png")
    desc = kept.groupby("genre")[["n_bars", "bpm", "notes_per_bar", "mean_pitch", "n_tokens"]].median().reindex(genres)
    md += ["## Медианы по классам\n", "| класс | тактов | BPM | нот/такт | ср. высота | токенов |", "|---|---|---|---|---|---|"]
    md += [f"| {g} | {r.n_bars:.0f} | {r.bpm:.0f} | {r.notes_per_bar:.1f} | {r.mean_pitch:.1f} | {r.n_tokens:.0f} |"
           for g, r in desc.iterrows()]
    md.append("")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(kept["bpm"], bins=60, color="#3b6ea5"); ax[0].set_title("Распределение темпа (BPM)")
    for b in [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 140, 152, 168, 184, 208]:
        ax[0].axvline(b, color="k", alpha=0.15, lw=0.8)
    ax[1].hist(kept["n_bars"], bins=60, color="#3b6ea5"); ax[1].set_title("Длина пьес (тактов)")
    save(fig, "04_tempo_length_hist.png")

    # 4. тональности (оценка Крумхансла–Шмуклера)
    keys = kept["key_est"].replace("", np.nan).dropna()
    mode = keys.str.split().str[1].value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    kc = keys.value_counts().head(24)
    ax.bar(kc.index, kc.values, color="#6a9a5b"); ax.tick_params(axis="x", rotation=60)
    ax.set_title(f"Оценка тональности: мажор {mode.get('major', 0):.0%}, минор {mode.get('minor', 0):.0%}")
    save(fig, "05_keys.png")
    md += ["## Тональности\n", f"Оценка по профилям Крумхансла: мажор {mode.get('major', 0):.0%}, "
           f"минор {mode.get('minor', 0):.0%}. Тональность пользователь не задаёт.\n"]

    # 5. статистика токенов
    kinds = {}
    for t, c in enumerate(counts[: vocab.n_body]):
        kinds[vocab.kind(t)] = kinds.get(vocab.kind(t), 0) + int(c)
    a, b = vocab.ranges["PITCH"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].bar(range(21, 109), counts[a:b], color="#3b6ea5"); ax[0].set_title("Распределение высот (MIDI)")
    a, b = vocab.ranges["DURATION"]
    ax[1].bar(range(1, 33), counts[a:b], color="#a56b3b"); ax[1].set_title("Длительности (1/16)")
    ax[2].bar(list(kinds), list(kinds.values()), color="#7a8ba6"); ax[2].set_title("Типы токенов")
    save(fig, "06_tokens.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(kept["n_tokens"], bins=80, color="#3b6ea5"); ax.set_title("Длина последовательности (токенов на пьесу)")
    ax.axvline(512, color="r", ls="--", label="окно 512"); ax.axvline(1024, color="r", ls=":", label="окно 1024")
    ax.legend()
    save(fig, "07_sequence_length.png")
    total = int(counts.sum())
    md += ["## Токены\n", f"Словарь: **{len(vocab)}** токенов (тело {vocab.n_body}, условия {len(vocab) - vocab.n_body}). "
           f"Всего токенов в датасете: **{total / 1e6:.1f}M**, медиана на пьесу {kept['n_tokens'].median():.0f}, "
           f"токенов на такт ≈ {(kept['n_tokens'] / kept['n_bars']).median():.0f}.\n",
           "| тип | доля |", "|---|---|"] + [f"| {k} | {v / total:.1%} |" for k, v in kinds.items()] + [""]
    md += ["Окно 512 токенов покрывает ≈ "
           f"{512 / (kept['n_tokens'] / kept['n_bars']).median():.0f} тактов, окно 1024 — ≈ "
           f"{1024 / (kept['n_tokens'] / kept['n_bars']).median():.0f}.\n"]

    out = resolve("reports/eda_summary.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Готово: {out}, графики в {FIG}")


if __name__ == "__main__":
    main()
