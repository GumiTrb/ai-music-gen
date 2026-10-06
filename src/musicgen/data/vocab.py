"""Словарь REMI-токенов с управляющими токенами (жанр, темп, длина).

Порядок блоков в словаре выбран так, чтобы id «музыкальных» токенов не зависели
от списка жанров: условия (TEMPO, LENGTH, GENRE) лежат в конце. Поэтому один раз
токенизированный датасет можно использовать с разной группировкой жанров.

Последовательность выглядит так:
    BOS GENRE_g TEMPO_t LENGTH_l  BAR [POSITION_i] PITCH_p DURATION_d VELOCITY_v ... BAR ... EOS
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PAD, BOS, EOS, BAR = 0, 1, 2, 3
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "BAR"]

N_POSITIONS = 16  # шаг 1/16 такта 4/4
PITCH_MIN, PITCH_MAX = 21, 108  # 88 клавиш фортепиано
N_DURATIONS = 32  # 1..32 шестнадцатых (до двух тактов)
N_VELOCITIES = 16  # velocity // 8

# Центры бинов темпа (BPM). Ближайший бин ищется в лог-шкале.
TEMPO_BINS = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 140, 152, 168, 184, 208]
# Центры бинов длины (в тактах).
LENGTH_BINS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

DEFAULT_GENRES = [
    "classical", "rock", "pop", "jazz", "soundtrack", "latin_world", "country_folk", "unknown",
]
UNKNOWN_GENRE = "unknown"
PREFIX_LEN = 4  # BOS, GENRE, TEMPO, LENGTH


def _nearest_log(value: float, centers: list[int]) -> int:
    value = max(float(value), 1e-6)
    return min(range(len(centers)), key=lambda i: abs(math.log(value) - math.log(centers[i])))


@dataclass
class Vocab:
    genres: list[str] = field(default_factory=lambda: list(DEFAULT_GENRES))

    def __post_init__(self):
        if UNKNOWN_GENRE not in self.genres:
            self.genres = list(self.genres) + [UNKNOWN_GENRE]
        names = list(SPECIAL_TOKENS)
        self.ranges: dict[str, tuple[int, int]] = {}

        def add(kind: str, items):
            start = len(names)
            names.extend(f"{kind}_{x}" for x in items)
            self.ranges[kind] = (start, len(names))

        add("POSITION", range(N_POSITIONS))
        add("PITCH", range(PITCH_MIN, PITCH_MAX + 1))
        add("DURATION", range(1, N_DURATIONS + 1))
        add("VELOCITY", range(N_VELOCITIES))
        add("TEMPO", TEMPO_BINS)
        add("LENGTH", LENGTH_BINS)
        add("GENRE", self.genres)
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}

    # ---- размеры и типы -------------------------------------------------
    def __len__(self) -> int:
        return len(self.names)

    @property
    def n_body(self) -> int:
        """Число токенов, которые могут встречаться в теле пьесы (без условий)."""
        return self.ranges["VELOCITY"][1]

    def kind(self, tok: int) -> str:
        if tok < len(SPECIAL_TOKENS):
            return SPECIAL_TOKENS[tok]
        for k, (a, b) in self.ranges.items():
            if a <= tok < b:
                return k
        raise ValueError(f"unknown token id {tok}")

    def is_kind(self, tok: int, kind: str) -> bool:
        a, b = self.ranges[kind]
        return a <= tok < b

    def value(self, tok: int) -> int:
        """Индекс внутри своего блока (для PITCH возвращает MIDI-высоту)."""
        k = self.kind(tok)
        a, _ = self.ranges[k]
        if k == "PITCH":
            return tok - a + PITCH_MIN
        if k == "DURATION":
            return tok - a + 1
        return tok - a

    # ---- конструкторы токенов -------------------------------------------
    def position(self, i: int) -> int:
        return self.ranges["POSITION"][0] + i

    def pitch(self, p: int) -> int:
        return self.ranges["PITCH"][0] + p - PITCH_MIN

    def duration(self, d: int) -> int:
        return self.ranges["DURATION"][0] + min(max(d, 1), N_DURATIONS) - 1

    def velocity(self, v: int) -> int:
        return self.ranges["VELOCITY"][0] + min(max(v, 0), 127) // 8

    def tempo_bin(self, bpm: float) -> int:
        return _nearest_log(bpm, TEMPO_BINS)

    def tempo(self, bpm: float) -> int:
        return self.ranges["TEMPO"][0] + self.tempo_bin(bpm)

    def length_bin(self, n_bars: int) -> int:
        return _nearest_log(n_bars, LENGTH_BINS)

    def length(self, n_bars: int) -> int:
        return self.ranges["LENGTH"][0] + self.length_bin(n_bars)

    def genre(self, g: str) -> int:
        g = g if g in self.genres else UNKNOWN_GENRE
        return self.ranges["GENRE"][0] + self.genres.index(g)

    def prefix(self, genre: str, bpm: float, n_bars: int) -> list[int]:
        return [BOS, self.genre(genre), self.tempo(bpm), self.length(n_bars)]

    def decode_prefix(self, tokens) -> dict:
        out = {}
        for t in tokens[:PREFIX_LEN]:
            t = int(t)
            if t >= self.n_body:
                k = self.kind(t)
                v = self.value(t)
                if k == "TEMPO":
                    out["bpm"] = TEMPO_BINS[v]
                elif k == "LENGTH":
                    out["n_bars"] = LENGTH_BINS[v]
                elif k == "GENRE":
                    out["genre"] = self.genres[v]
        return out

    def to_str(self, tokens) -> list[str]:
        return [self.names[int(t)] for t in tokens]

    # ---- сохранение -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"genres": self.genres, "names": self.names}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        v = cls(genres=data["genres"])
        assert v.names == data["names"], "vocab.json не совпадает с текущим кодом словаря"
        return v
