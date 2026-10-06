"""Неделя 1: разбор всего ADL Piano MIDI для анализа данных.

Для каждого файла: жанр/исполнитель из пути, условия (темп, длина в тактах), статистика нот,
оценка тональности, число REMI-токенов и причина отбраковки. Токены здесь считаются только для
статистики; сохранение токенизированного датасета и разбиение train/val/test — неделя 2.

    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --limit 300      # быстрый прогон на части файлов

Результат в data/processed/:
    pieces.csv        по строке на файл
    token_counts.npy  частоты токенов по принятым пьесам
    vocab.json        словарь
    summary.json      сводка
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from musicgen.data import adl
from musicgen.data.vocab import Vocab
from musicgen.utils import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=0, help="взять только N файлов (для отладки)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    out = resolve(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    zip_path = adl.download(cfg["dataset_url"], resolve(cfg["raw_zip"]))

    files = adl.list_zip(zip_path)
    if args.limit:
        files = files.sample(n=min(args.limit, len(files)), random_state=0).sort_index()
    print(f"Файлов в архиве: {len(files)}")

    vocab = Vocab(genres=cfg["genres"])
    metas, token_lists = [], []
    with Pool(args.workers, initializer=adl._init_worker, initargs=(str(zip_path),)) as pool:
        for meta, toks in tqdm(pool.imap(adl.parse_one, files["path"], chunksize=16), total=len(files)):
            metas.append(meta)
            token_lists.append(toks)

    df = files.merge(pd.DataFrame(metas), on="path", how="left")
    df["genre"] = df["genre_raw"].map(cfg["genre_map"]).fillna("unknown")
    df["reject"] = adl.apply_filters(df, cfg["filters"])
    kept = df["reject"] == ""
    print("Отбраковка:\n" + df["reject"].replace("", "OK").value_counts().to_string())

    counts = np.zeros(len(vocab), dtype=np.int64)
    for ok, toks in zip(kept, token_lists):
        if ok:
            counts += np.bincount(toks, minlength=len(vocab))
    np.save(out / "token_counts.npy", counts)
    df.to_csv(out / "pieces.csv", index=False)
    vocab.save(out / "vocab.json")

    k = df[kept]
    summary = {
        "files": int(len(df)),
        "kept": int(kept.sum()),
        "rejected": df.loc[~kept, "reject"].value_counts().to_dict(),
        "tokens_total": int(counts.sum()),
        "by_genre": k["genre"].value_counts().to_dict(),
        "artists": int(k["artist"].nunique()),
    }
    adl.save_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
