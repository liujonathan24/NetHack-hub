"""Parser/serializer for the engine's NLELVLF1 level-file container.
Format (nle_fast_reset.c:505):
  "NLELVLF1" | u32 count | per entry: u32 namelen, name, u64 size, bytes
"""
import struct
MAGIC = b"NLELVLF1"

def parse(blob):
    assert blob[:8] == MAGIC, blob[:8]
    off = 8
    (count,) = struct.unpack_from("<I", blob, off); off += 4
    out = []
    for _ in range(count):
        (nl,) = struct.unpack_from("<I", blob, off); off += 4
        name = blob[off:off+nl].decode(); off += nl
        (sz,) = struct.unpack_from("<Q", blob, off); off += 8
        out.append((name, blob[off:off+sz])); off += sz
    return out

def build(entries):
    out = [MAGIC, struct.pack("<I", len(entries))]
    for name, data in entries:
        nb = name.encode()
        out.append(struct.pack("<I", len(nb)) + nb + struct.pack("<Q", len(data)) + data)
    return b"".join(out)
