# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

def read_text_files(data_dir: str) -> List[Tuple[str, str]]:
    p = Path(data_dir)
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(list(p.glob(ext)))
    out = []
    for fp in sorted(files):
        content = fp.read_text(encoding="utf-8", errors="ignore")
        out.append((content, fp.name))
    return out

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def split_into_chunks(text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    chunks: List[Chunk] = []
    start, cid, n = 0, 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        t = text[start:end].strip()
        if t:
            chunks.append(Chunk(text=t, source=source, chunk_id=cid))
            cid += 1
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks
