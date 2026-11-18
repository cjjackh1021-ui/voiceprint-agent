"""Template encryption and cancellable projection.

AES-256-GCM hides the stored vectors. A per-speaker orthogonal matrix rotates them so a leaked
template can be revoked by changing the seed, without changing cosine similarity.
The master key is wrapped with Windows DPAPI and is not written in the clear.
"""
import ctypes
import hashlib
import os
from ctypes import wintypes

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class DATA_BLOB(ctypes.Structure):
    _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes):
    buf = ctypes.create_string_buffer(data, len(data))
    blob = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def dpapi_protect(data: bytes) -> bytes:
    incoming, keep = _blob(data)
    outgoing = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(incoming), None, None, None, None, 0, ctypes.byref(outgoing)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(outgoing.pbData, outgoing.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(outgoing.pbData)
        del keep


def dpapi_unprotect(data: bytes) -> bytes:
    incoming, keep = _blob(data)
    outgoing = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(incoming), None, None, None, None, 0, ctypes.byref(outgoing)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(outgoing.pbData, outgoing.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(outgoing.pbData)
        del keep


def encrypt(key: bytes, plaintext: bytes, aad: bytes | None = None) -> bytes:
    """aad binds the ciphertext to its owner: a blob moved to another row no longer decrypts."""
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def decrypt(key: bytes, blob: bytes, aad: bytes | None = None) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad)


def projection(seed: bytes, dim: int) -> np.ndarray:
    # All 256 bits of the digest seed the generator (the first version kept only 64).
    digest = hashlib.sha256(seed).digest()
    rng = np.random.default_rng([int(word) for word in np.frombuffer(digest, dtype='<u4')])
    raw = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def apply_projection(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32) @ matrix.T


def undo_projection(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32) @ matrix
