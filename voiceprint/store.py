"""SQLite store. Speaker vectors are encrypted before they touch the database file.

Every ciphertext is bound (AES-GCM associated data) to the speaker name, the field, the
embedding model and the template shape, so a blob copied into another row fails to decrypt.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from voiceprint.security import decrypt, dpapi_protect, dpapi_unprotect, encrypt

SCHEMA_VERSION = 2


class ModelMismatch(ValueError):
    """The template was enrolled with a different embedding model."""


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _aad(name: str, field: str, embedder: str, model_id: str, n_seg: int, dim: int) -> bytes:
    return '\x1f'.join(['voiceprint', str(SCHEMA_VERSION), name, field, embedder, model_id,
                        str(n_seg), str(dim)]).encode('utf-8')


class Store:
    def __init__(self, db_path: Path, key_path: Path):
        self.db_path = Path(db_path)
        self.key_path = Path(key_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        # Deleted templates are overwritten in the file instead of lingering in free pages.
        self.db.execute("pragma secure_delete = on")
        self._prepare_schema()
        self.key = self._load_key()

    def _prepare_schema(self):
        columns = [row[1] for row in self.db.execute("pragma table_info(speakers)")]
        if columns and 'model_id' not in columns:
            self.db.close()
            raise RuntimeError(
                f'{self.db_path} 是旧版模板库（没有模型绑定和密文绑定），不能直接使用。'
                '请把它移走后重新注册。')
        self.db.execute("""create table if not exists speakers (
            name text primary key, created_at text, seed blob, segments blob, n_seg integer, dim integer,
            embedder text not null, model_id text not null)""")
        self.db.execute("""create table if not exists events (
            id integer primary key, ts text, name text, kind text, score real, decision text)""")
        self.db.execute("""create table if not exists audit (
            id integer primary key, ts text, action text, detail text)""")
        self.db.execute(f"pragma user_version = {SCHEMA_VERSION}")
        self.db.commit()

    def _load_key(self) -> bytes:
        if self.key_path.exists():
            return dpapi_unprotect(self.key_path.read_bytes())
        if self.db.execute("select count(*) from speakers").fetchone()[0]:
            self.db.close()
            raise RuntimeError(
                f'找不到主密钥 {self.key_path}，但库里已有模板。'
                '不会自动生成新密钥：请恢复原来的 master.key，否则这些模板无法解密。')
        key = os.urandom(32)
        tmp = self.key_path.with_name(self.key_path.name + '.tmp')
        tmp.write_bytes(dpapi_protect(key))
        os.replace(tmp, self.key_path)
        return key

    def exists(self, name: str) -> bool:
        return self.db.execute("select 1 from speakers where name=?", (name,)).fetchone() is not None

    def save_speaker(self, name: str, seed: bytes, segments: np.ndarray, embedder: str, model_id: str,
                     replace: bool = False):
        """replace=False refuses to overwrite: re-enrolling must not silently take over a name."""
        segments = np.asarray(segments, dtype=np.float32)
        n_seg, dim = segments.shape
        row = (name, _now(),
               encrypt(self.key, seed, _aad(name, 'seed', embedder, model_id, n_seg, dim)),
               encrypt(self.key, segments.tobytes(), _aad(name, 'segments', embedder, model_id, n_seg, dim)),
               n_seg, dim, embedder, model_id)
        if replace:
            done = self.db.execute(
                "update speakers set created_at=?, seed=?, segments=?, n_seg=?, dim=?, embedder=?, "
                "model_id=? where name=?", row[1:] + row[:1]).rowcount
            if not done:
                raise KeyError(name)
        else:
            try:
                self.db.execute("insert into speakers values (?,?,?,?,?,?,?,?)", row)
            except sqlite3.IntegrityError:
                raise FileExistsError(name) from None
        self.db.commit()

    def load_speaker(self, name: str, embedder: str | None = None,
                     model_id: str | None = None) -> tuple[bytes, np.ndarray]:
        """Pass the running model to refuse templates that were enrolled with another one."""
        row = self.db.execute(
            "select seed, segments, n_seg, dim, embedder, model_id from speakers where name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(name)
        blob_seed, blob_segments, n_seg, dim, stored_embedder, stored_model = row
        if embedder is not None and (stored_embedder, stored_model) != (embedder, model_id):
            raise ModelMismatch(
                f'{name} 的模板是用 {stored_embedder}（{stored_model[:12]}）注册的，'
                f'当前模型是 {embedder}（{(model_id or "")[:12]}）。请用原模型运行，或重新注册。')
        seed = decrypt(self.key, blob_seed, _aad(name, 'seed', stored_embedder, stored_model, n_seg, dim))
        raw = decrypt(self.key, blob_segments, _aad(name, 'segments', stored_embedder, stored_model, n_seg, dim))
        return seed, np.frombuffer(raw, dtype=np.float32).reshape(n_seg, dim).copy()

    def names(self, embedder: str | None = None, model_id: str | None = None) -> list[str]:
        """All names, or only those enrolled with the given model."""
        if embedder is None:
            return [r[0] for r in self.db.execute("select name from speakers order by name")]
        return [r[0] for r in self.db.execute(
            "select name from speakers where embedder=? and model_id=? order by name", (embedder, model_id))]

    def delete(self, name: str):
        # The score history belongs to the deleted template; a later speaker with the same name starts clean.
        self.db.execute("delete from speakers where name=?", (name,))
        self.db.execute("delete from events where name=?", (name,))
        self.db.commit()

    def event(self, name: str, kind: str, score: float, decision: str):
        self.db.execute("insert into events(ts,name,kind,score,decision) values (?,?,?,?,?)",
                        (_now(), name, kind, score, decision))
        self.db.commit()

    def events(self, name: str) -> list[tuple]:
        return self.db.execute(
            "select ts, kind, score, decision from events where name=? order by id", (name,)).fetchall()

    def recent_failures(self, name: str, minutes: float) -> int:
        """Verify attempts in a row that were not accepted, counted back from now within the window."""
        since = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec='seconds')
        rows = self.db.execute(
            "select decision from events where name=? and kind='verify' and ts>=? order by id desc", (name, since))
        count = 0
        for (decision,) in rows:
            if decision == 'accept':
                break
            count += 1
        return count

    def audit(self, action: str, detail: str):
        self.db.execute("insert into audit(ts,action,detail) values (?,?,?)", (_now(), action, detail[:500]))
        self.db.commit()

    def close(self):
        self.db.close()
