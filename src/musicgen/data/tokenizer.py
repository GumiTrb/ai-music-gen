"""REMI-токенизатор: MIDI <-> токены.

Время квантуется сеткой 1/16 такта 4/4 (4 шага на долю). Внутри такта ноты
кодируются как POSITION (только если позиция поменялась), PITCH, DURATION, VELOCITY.
Пустые такты сохраняются (BAR подряд), ведущие пустые такты отрезаются.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .vocab import BAR, BOS, EOS, N_DURATIONS, N_POSITIONS, PAD, PITCH_MAX, PITCH_MIN, Vocab

STEPS_PER_BEAT = 4
STEPS_PER_BAR = N_POSITIONS  # 4/4


@dataclass(frozen=True)
class Note:
    start: int  # в шестнадцатых от начала пьесы
    pitch: int
    duration: int  # в шестнадцатых
    velocity: int  # 0..127


@dataclass
class Piece:
    notes: list[Note]
    bpm: float
    time_signatures: list[str]
    key: str | None = None
    n_dropped: int = 0  # ноты вне диапазона фортепиано

    @property
    def is_four_four(self) -> bool:
        # 2/2 имеет ту же длину такта, что и 4/4, сетка 1/16 подходит
        return all(ts in ("4/4", "2/2") for ts in self.time_signatures)

    @property
    def n_bars(self) -> int:
        return 0 if not self.notes else max(n.start for n in self.notes) // STEPS_PER_BAR + 1


# --------------------------------------------------------------------------- MIDI

def _dominant_tempo(midi) -> float:
    """Темп, который звучит дольше всего (по тикам)."""
    changes = sorted(midi.tempo_changes, key=lambda t: t.time)
    if not changes:
        return 120.0
    end = max([midi.max_tick] + [c.time for c in changes]) + 1
    weights: dict[float, int] = {}
    for i, c in enumerate(changes):
        nxt = changes[i + 1].time if i + 1 < len(changes) else end
        bpm = round(float(c.tempo), 1)
        weights[bpm] = weights.get(bpm, 0) + max(nxt - c.time, 0)
    bpm = max(weights, key=weights.get)
    return bpm if 20 <= bpm <= 400 else 120.0


def read_midi(source: str | Path | bytes) -> Piece:
    """Читает MIDI (путь или байты), объединяет все не-барабанные партии, квантует на сетку 1/16."""
    import io

    import miditoolkit

    if isinstance(source, (bytes, bytearray)):
        midi = miditoolkit.MidiFile(file=io.BytesIO(source))
    else:
        midi = miditoolkit.MidiFile(str(source))
    step = midi.ticks_per_beat / STEPS_PER_BEAT
    best: dict[tuple[int, int], Note] = {}
    dropped = 0
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if not PITCH_MIN <= n.pitch <= PITCH_MAX:
                dropped += 1
                continue
            start = int(round(n.start / step))
            dur = max(1, int(round(n.end / step)) - start)
            note = Note(start, n.pitch, min(dur, N_DURATIONS), int(n.velocity))
            key = (start, n.pitch)
            # дубликаты (одна высота в один момент) — оставляем самую длинную
            if key not in best or best[key].duration < note.duration:
                best[key] = note
    notes = sorted(best.values(), key=lambda n: (n.start, n.pitch))
    if notes:  # отрезаем ведущие пустые такты, сохраняя выравнивание по тактам
        shift = (notes[0].start // STEPS_PER_BAR) * STEPS_PER_BAR
        if shift:
            notes = [Note(n.start - shift, n.pitch, n.duration, n.velocity) for n in notes]
    ts = [f"{t.numerator}/{t.denominator}" for t in midi.time_signature_changes]
    key = midi.key_signature_changes[0].key_name if midi.key_signature_changes else None
    return Piece(notes=notes, bpm=_dominant_tempo(midi), time_signatures=ts, key=key, n_dropped=dropped)


def write_midi(notes: list[Note], bpm: float, path: str | Path, ticks_per_beat: int = 480) -> None:
    import miditoolkit
    from miditoolkit.midi.containers import Instrument, TempoChange, TimeSignature
    from miditoolkit.midi.containers import Note as MNote

    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    step = ticks_per_beat // STEPS_PER_BEAT
    piano = Instrument(program=0, is_drum=False, name="Piano")
    for n in notes:
        piano.notes.append(MNote(velocity=int(n.velocity), pitch=int(n.pitch),
                                 start=n.start * step, end=(n.start + n.duration) * step))
    midi.instruments.append(piano)
    midi.tempo_changes.append(TempoChange(float(bpm), 0))
    midi.time_signature_changes.append(TimeSignature(4, 4, 0))
    midi.max_tick = max([0] + [n.end for n in piano.notes])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(path))


# --------------------------------------------------------------------------- токены

class REMITokenizer:
    def __init__(self, vocab: Vocab | None = None):
        self.vocab = vocab or Vocab()

    # notes -> тело последовательности (без префикса условий и EOS)
    def encode_notes(self, notes: list[Note]) -> list[int]:
        v = self.vocab
        notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        n_bars = 0 if not notes else notes[-1].start // STEPS_PER_BAR + 1
        tokens: list[int] = []
        i = 0
        for bar in range(n_bars):
            tokens.append(BAR)
            prev_pos = -1
            bar_end = (bar + 1) * STEPS_PER_BAR
            while i < len(notes) and notes[i].start < bar_end:
                n = notes[i]
                pos = n.start % STEPS_PER_BAR
                if pos != prev_pos:
                    tokens.append(v.position(pos))
                    prev_pos = pos
                tokens += [v.pitch(n.pitch), v.duration(n.duration), v.velocity(n.velocity)]
                i += 1
        return tokens

    def encode_piece(self, piece: Piece, genre: str, with_prefix: bool = True) -> list[int]:
        body = self.encode_notes(piece.notes)
        if not with_prefix:
            return body
        return self.vocab.prefix(genre, piece.bpm, piece.n_bars) + body + [EOS]

    def encode_midi(self, path, genre: str = "unknown") -> list[int]:
        return self.encode_piece(read_midi(path), genre)

    # токены -> ноты (терпимо к ошибкам модели: невалидные токены пропускаются)
    def decode(self, tokens) -> tuple[list[Note], dict]:
        v = self.vocab
        info = v.decode_prefix(tokens) if len(tokens) and int(tokens[0]) == BOS else {}
        notes: list[Note] = []
        bar, pos = -1, 0
        cur = None  # [start, pitch, duration, velocity] ноты, которая ещё собирается
        expect = None  # какой атрибут ждём: "dur" / "vel"

        def flush():
            nonlocal cur
            if cur is not None:
                notes.append(Note(cur[0], cur[1], cur[2], cur[3]))
                cur = None

        for t in tokens:
            t = int(t)
            if t in (PAD, BOS) or t >= v.n_body:
                continue
            if t == EOS:
                break
            k = v.kind(t)
            if k == "BAR":
                flush(); bar += 1; pos = 0; expect = None
            elif k == "POSITION":
                flush(); pos = v.value(t); expect = None
            elif k == "PITCH":
                flush()
                cur = [max(bar, 0) * STEPS_PER_BAR + pos, v.value(t), STEPS_PER_BEAT, 64]
                expect = "dur"
            elif k == "DURATION" and cur is not None and expect == "dur":
                cur[2] = v.value(t); expect = "vel"
            elif k == "VELOCITY" and cur is not None and expect == "vel":
                cur[3] = v.value(t) * 8 + 4; expect = None
        flush()
        info["n_bars_decoded"] = bar + 1
        return notes, info

    def decode_to_midi(self, tokens, path, bpm: float | None = None) -> list[Note]:
        notes, info = self.decode(tokens)
        write_midi(notes, bpm or info.get("bpm", 120), path)
        return notes
