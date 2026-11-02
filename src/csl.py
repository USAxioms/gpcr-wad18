"""
Cognitive State Ledger (CSL) — append-only SHA-256 block chain.

Each block:
  L_i = { i, T_i, S_i, H_{i-1}, H_i }

  i      : sequential index (int, starts at 0)
  T_i    : ISO-8601 UTC timestamp at block creation
  S_i    : state payload dict (mathematical invariants, free energies, etc.)
  H_{i-1}: SHA-256 hex digest of the previous block's canonical JSON
  H_i    : SHA-256 hex digest of canonical_json(i, T_i, S_i, H_{i-1})

Block 0 (genesis): H_{-1} = "0" * 64 (null anchor).

The canonical serialisation is json.dumps sorted by key, separators
(",", ":") — the same rule as the R3 capsule's canon().

The ledger is never mutated after a block is appended.  Any read-back
that recomputes the hash chain and finds a mismatch raises IntegrityError.
"""

import hashlib
import json
import os
from datetime import datetime, timezone


GENESIS_PREV = "0" * 64


class IntegrityError(Exception):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _block_hash(i, T_i, S_i, H_prev):
    payload = _canon({"H_prev": H_prev, "S_i": S_i, "T_i": T_i, "i": i})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CSL:
    def __init__(self, path):
        self.path    = path
        self._blocks = []

        if os.path.exists(path):
            with open(path) as fh:
                content = fh.read().strip()
            if content:
                self._blocks = json.loads(content)
                self._verify()

    # ---------------------------------------------------------------- public
    def append(self, state: dict, label: str = "") -> dict:
        """Append a new block and return it."""
        i      = len(self._blocks)
        T_i    = datetime.now(timezone.utc).isoformat()
        H_prev = self._blocks[-1]["H_i"] if self._blocks else GENESIS_PREV
        S_i    = dict(state)
        if label:
            S_i["_label"] = label
        H_i    = _block_hash(i, T_i, S_i, H_prev)
        block  = {"H_i": H_i, "H_prev": H_prev, "S_i": S_i, "T_i": T_i, "i": i}
        self._blocks.append(block)
        self._flush()
        return block

    def root_hash(self) -> str:
        if not self._blocks:
            return GENESIS_PREV
        return self._blocks[-1]["H_i"]

    def verify(self):
        self._verify()
        return True

    def blocks(self):
        return list(self._blocks)

    # --------------------------------------------------------------- private
    def _verify(self):
        prev = GENESIS_PREV
        for blk in self._blocks:
            expected = _block_hash(blk["i"], blk["T_i"], blk["S_i"], prev)
            if expected != blk["H_i"]:
                raise IntegrityError(
                    "Hash mismatch at block %d: expected %s, got %s"
                    % (blk["i"], expected, blk["H_i"])
                )
            if blk["H_prev"] != prev:
                raise IntegrityError(
                    "H_prev mismatch at block %d" % blk["i"]
                )
            prev = blk["H_i"]

    def _flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self._blocks, fh, sort_keys=True, indent=2)
