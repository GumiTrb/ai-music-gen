"""Токенизатор: словарь, round-trip MIDI -> токены -> MIDI, квантование."""

import numpy as np
import pytest

from musicgen.data.tokenizer import Note, REMITokenizer, read_midi, write_midi
from musicgen.data.vocab import BAR, BOS, EOS, PREFIX_LEN, TEMPO_BINS, Vocab


@pytest.fixture
def vocab():
    return Vocab()


def random_notes(seed=0, n=200, bars=12):
    """Случайные ноты на сетке; ноты одной высоты не перекрываются (MIDI это не различает)."""
    rng = np.random.default_rng(seed)
    busy_until: dict[int, int] = {}
    notes = []
    for start in sorted(rng.integers(0, bars * 16, n)):
        pitch = int(rng.integers(21, 109))
        if busy_until.get(pitch, -1) > start:
            continue
        dur = int(rng.integers(1, 33))
        busy_until[pitch] = start + dur
        notes.append(Note(int(start), pitch, dur, int(rng.integers(1, 128))))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def quantized(notes):
    return {(n.start, n.pitch, n.duration, n.velocity // 8) for n in notes}


def test_vocab_layout(vocab):
    assert len(vocab) == len(set(vocab.names))
    # тело словаря не зависит от жанров — условия в конце
    assert Vocab(genres=["a", "b"]).n_body == vocab.n_body
    assert vocab.kind(vocab.pitch(60)) == "PITCH" and vocab.value(vocab.pitch(60)) == 60
    assert vocab.value(vocab.duration(7)) == 7
    assert TEMPO_BINS[vocab.tempo_bin(121)] == 120
    p = vocab.prefix("jazz", 120, 32)
    assert p[0] == BOS and vocab.decode_prefix(p) == {"genre": "jazz", "bpm": 120, "n_bars": 32}
    assert vocab.genre("not-a-genre") == vocab.genre("unknown")


def test_round_trip_tokens(vocab):
    tok = REMITokenizer(vocab)
    notes = random_notes()
    body = tok.encode_notes(notes)
    decoded, _ = tok.decode(body)
    assert quantized(decoded) == quantized(notes)
    assert body.count(BAR) == notes[-1].start // 16 + 1


def test_round_trip_midi_file(vocab, tmp_path):
    """MIDI -> токены -> MIDI: ноты совпадают с точностью до квантования 1/16 и бина громкости."""
    tok = REMITokenizer(vocab)
    notes = random_notes(1)
    src = tmp_path / "src.mid"
    write_midi(notes, 100, src)
    piece = read_midi(src)
    assert quantized(piece.notes) == quantized(notes)
    tokens = tok.encode_piece(piece, "pop")
    assert tokens[:PREFIX_LEN] == vocab.prefix("pop", 100, piece.n_bars) and tokens[-1] == EOS
    out = tmp_path / "out.mid"
    tok.decode_to_midi(tokens, out)
    again = read_midi(out)
    assert quantized(again.notes) == quantized(notes)
    assert again.bpm == TEMPO_BINS[vocab.tempo_bin(100)]  # темп восстанавливается до центра бина


def test_quantization_and_leading_bars(vocab, tmp_path):
    # нота не на сетке + два пустых такта в начале
    write_midi([Note(32 + 5, 60, 4, 80)], 120, tmp_path / "a.mid")
    piece = read_midi(tmp_path / "a.mid")
    assert piece.notes == [Note(5, 60, 4, 80)]  # пустые такты отрезаны, выравнивание по такту сохранено


def test_decoder_is_tolerant(vocab):
    tok = REMITokenizer(vocab)
    garbage = [BAR, vocab.duration(3), vocab.pitch(60), vocab.pitch(64), vocab.duration(8), BAR, vocab.velocity(10)]
    notes, info = tok.decode(garbage)
    assert [n.pitch for n in notes] == [60, 64] and info["n_bars_decoded"] == 2


def test_round_trip_real_dataset(vocab):
    """Round-trip на каждом 100-м файле ADL Piano MIDI (если архив скачан)."""
    import zipfile

    from musicgen.utils import resolve

    path = resolve("data/raw/adl-piano-midi.zip")
    if not path.exists():
        pytest.skip("датасет не скачан: python scripts/analyze_dataset.py")
    tok = REMITokenizer(vocab)
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.lower().endswith(".mid")][::100]
    checked = 0
    for name in names:
        try:
            piece = read_midi(z.read(name))
        except Exception:
            continue  # битые файлы бывают, их отсеивает фильтр
        notes, info = tok.decode(tok.encode_piece(piece, "unknown"))
        assert quantized(notes) == quantized(piece.notes), name
        assert info["n_bars_decoded"] == piece.n_bars
        checked += 1
    assert checked > 50
