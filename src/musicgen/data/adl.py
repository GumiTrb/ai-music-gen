"""Работа с датасетом ADL Piano MIDI: скачивание, обход архива, разбор файлов, фильтры.

Структура архива: adl-piano-midi/<Жанр>/<Поджанр>/<Исполнитель>/<Название>.mid
Файлы читаются прямо из zip (часть имён недопустима для Windows).
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .tokenizer import Piece, REMITokenizer, read_midi
from .vocab import Vocab

MIDI_EXT = (".mid", ".midi")


def download(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {url} -> {dest}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def normalize_title(title: str) -> str:
    title = re.sub(r"\(\d+\)|\.\d+$", "", title.lower())
    return re.sub(r"[^a-z0-9]", "", title)


def list_zip(zip_path: str | Path) -> pd.DataFrame:
    """Таблица файлов архива с жанром, поджанром, исполнителем и названием."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(MIDI_EXT):
                continue
            parts = name.split("/")
            if len(parts) != 5:
                continue
            _, genre, sub, artist, fname = parts
            title = fname.rsplit(".", 1)[0].strip()
            rows.append(dict(path=name, genre_raw=genre, subgenre=sub, artist=artist.strip(), title=title))
    return pd.DataFrame(rows)


# ---- разбор одного файла (вызывается в пуле процессов) ------------------------

_ZIP = None


def _init_worker(zip_path):
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def parse_one(path: str) -> tuple[dict, list[int] | None]:
    """Возвращает (метаданные, токены тела) для файла из архива."""
    meta: dict = {"path": path}
    try:
        data = _ZIP.read(path)
        piece: Piece = read_midi(data)
    except Exception as e:  # битые файлы встречаются
        meta["error"] = f"{type(e).__name__}: {e}"[:200]
        return meta, None
    notes = piece.notes
    meta.update(
        bpm=piece.bpm,
        time_signatures=";".join(sorted(set(piece.time_signatures))) or "none",
        four_four=piece.is_four_four,
        key_meta=piece.key,
        n_bars=piece.n_bars,
        n_notes=len(notes),
        n_dropped=piece.n_dropped,
        mean_pitch=float(np.mean([n.pitch for n in notes])) if notes else np.nan,
        min_pitch=min((n.pitch for n in notes), default=np.nan),
        max_pitch=max((n.pitch for n in notes), default=np.nan),
        mean_velocity=float(np.mean([n.velocity for n in notes])) if notes else np.nan,
        notes_per_bar=len(notes) / max(piece.n_bars, 1),
        content_md5=hashlib.md5(str([(n.start, n.pitch, n.duration) for n in notes]).encode()).hexdigest(),
        error="",
    )
    pcs = np.bincount([n.pitch % 12 for n in notes], minlength=12) if notes else np.zeros(12)
    meta["key_est"] = estimate_key(pcs)
    tokens = REMITokenizer(VOCAB).encode_notes(notes)
    meta["n_tokens"] = len(tokens)
    return meta, tokens


VOCAB = Vocab()  # id токенов тела не зависят от списка жанров

# ---- оценка тональности (Крумхансл–Шмуклер) ----------------------------------

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(pc_hist) -> str | None:
    pc_hist = np.asarray(pc_hist, dtype=float)
    if pc_hist.sum() == 0:
        return None
    best, name = -2.0, None
    for tonic in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = np.corrcoef(pc_hist, np.roll(prof, tonic))[0, 1]
            if r > best:
                best, name = r, f"{_NAMES[tonic]} {mode}"
    return name


# ---- фильтры -------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, f: dict) -> pd.Series:
    """Возвращает причину отбраковки (пустая строка = файл принят)."""
    reason = pd.Series("", index=df.index, dtype=object)

    def mark(cond, why):
        sel = cond & (reason == "")
        reason[sel] = why

    mark(df["error"].fillna("") != "", "parse_error")
    mark(df["n_notes"].fillna(0) < f["min_notes"], "too_few_notes")
    mark(df["n_bars"].fillna(0) < f["min_bars"], "too_short")
    mark(df["n_bars"].fillna(0) > f["max_bars"], "too_long")
    mark((df["bpm"] < f["min_bpm"]) | (df["bpm"] > f["max_bpm"]), "bad_tempo")
    if f.get("only_four_four", True):
        mark(~df["four_four"].fillna(False).astype(bool), "not_4_4")
    # дубликаты: одинаковое содержимое или тот же исполнитель + то же название
    dup = df["content_md5"].duplicated() | df.assign(t=df["title"].map(normalize_title)).duplicated(["artist", "t"])
    mark(dup, "duplicate")
    return reason


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
