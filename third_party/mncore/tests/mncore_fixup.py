#!/usr/bin/env python3
"""Reference consumer for the MN-Core relocation contract (kernel ABI §13).

The `.vsm` address fields are static immediates, so a kernel buffer argument
cannot appear in an instruction. Instead the compiler emits, for every
relocatable operand:

  * the address field of that operand as the placeholder 0;
  * a trailing `# @mncore.fixup <site>` comment on the instruction line;
  * one sidecar entry per site in the `@mncore.fixup.begin` /
    `@mncore.fixup.end` comment block at end of file:

      kernel: main
      fixups:
        - site: 0
          arg: 0
          space: DRAM
          unit: longword
          addend_units: 0
          width_bits: 32
          opcode: mvp
          operand: 1

  * a `# @mncore.kernel <entry>` comment ahead of each kernel's instructions,
    because site numbers restart at 0 per kernel entry. The primary key is
    therefore `(kernel entry, site)`, not the module-wide emission order: one
    .vsm may hold several kernels and a bare site number would be ambiguous.
    A sidecar from an older compiler has no `kernel:` line; those records are
    keyed under `""`, which is also the entry a line outside any
    `@mncore.kernel` marker resolves against.

`relocate()` is the loader side (§13.3): it binds argument indices to device
DRAM bases (in *bytes*, §13.1), converts to the record's unit, patches the
address subfield, and returns the instruction text with the fixup comments
removed so `assemble3` consumes it unchanged.

Every failure is closed (§13.4): a site with no binding, a byte base that is
not a multiple of the unit, an address that does not fit `width_bits`, and a
record whose opcode/operand do not match the line it is attached to are all
rejected rather than guessed at.
"""

import re

UNIT_BYTES = {"longword": 8, "word": 4}
SPACES = ("DRAM", "PDM", "L2BM")
MV_OPERAND = {1: 1, 2: 2}  # record operand -> index in (mnemonic, src, dst)

_SITE = re.compile(r"\s*#\s*@mncore\.fixup\s+(\d+)\s*$")
_KERNEL = re.compile(r"\s*#\s*@mncore\.kernel\s+(\S+)\s*$")
_BEGIN = re.compile(r"\s*#\s*@mncore\.fixup\.begin\s*$")
_END = re.compile(r"\s*#\s*@mncore\.fixup\.end\s*$")
_KEY = re.compile(r"\s*#\s+(-\s+)?([A-Za-z_]+):\s*(\S+)\s*$")
_MEM = re.compile(r"^\$(d|p|lc|di)(\d+)(.*)$")

REQUIRED_FIELDS = ("site", "arg", "space", "unit", "addend_units",
                   "width_bits", "opcode", "operand")


class RelocError(Exception):
    pass


def _parse_sidecar(lines):
    """Return {(kernel, site): record} from the comment block at end of file.

    A leading `kernel:` line opens a group and every following `- site:` entry
    belongs to it until the next `kernel:` line. Records from a sidecar with
    no `kernel:` line at all are grouped under `""`.
    """
    out = {}
    i = 0
    while i < len(lines) and not _BEGIN.match(lines[i]):
        i += 1
    if i == len(lines):
        return out
    i += 1
    cur = None
    kernel = ""

    def flush():
        nonlocal cur
        if cur is None:
            return
        if "site" not in cur:
            raise RelocError(f"sidecar entry under kernel {kernel!r} has no site")
        key = (kernel, cur["site"])
        if key in out:
            raise RelocError(f"duplicate (kernel, site) {key}")
        out[key] = cur
        cur = None

    while i < len(lines):
        line = lines[i]
        i += 1
        if _END.match(line):
            flush()
            return out
        if line.strip() in ("#", "", "# fixups:"):
            continue
        m = _KEY.match(line)
        if not m:
            raise RelocError(f"malformed sidecar line: {line!r}")
        starts_entry, key, value = m.group(1), m.group(2), m.group(3)
        if key == "kernel" and not starts_entry:
            flush()
            kernel = value
            continue
        if starts_entry:
            flush()
            cur = {}
        if cur is None:
            raise RelocError(f"sidecar field {key!r} outside an entry")
        if key in cur:
            raise RelocError(f"duplicate field {key!r} for site {cur}")
        cur[key] = value
    raise RelocError("unterminated @mncore.fixup.begin block")


def split(vsm_text):
    """Split compiler output into (instruction text, records).

    `records` is keyed by `(kernel entry, site)`. The instruction text still
    contains the `# @mncore.fixup <n>` comments; `relocate()` strips them.
    """
    lines = vsm_text.splitlines()
    raw = _parse_sidecar(lines)
    records = {}
    for (kernel, _), rec in raw.items():
        where = kernel or "<no kernel>"
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise RelocError(f"{where} site {rec.get('site')} is missing {missing}")
        try:
            for f in ("site", "arg", "addend_units", "width_bits", "operand"):
                rec[f] = int(rec[f])
        except ValueError as e:
            raise RelocError(f"{where} site {rec.get('site')}: {e}") from None
        rec["kernel"] = kernel
        if rec["space"] not in SPACES:
            raise RelocError(
                f"{where} site {rec['site']}: unknown space {rec['space']!r}")
        if rec["unit"] not in UNIT_BYTES:
            raise RelocError(
                f"{where} site {rec['site']}: unknown unit {rec['unit']!r}")
        key = (kernel, rec["site"])
        if key in records:
            # Two spellings of the same number ("0" and "00") are distinct to
            # the text parser but not to the loader; refuse rather than let
            # one silently overwrite the other.
            raise RelocError(f"duplicate (kernel, site) {key}")
        records[key] = rec
    return vsm_text, records


def relocate(vsm_text, bindings):
    """Apply relocations and drop the fixup comments.

    `bindings` maps a kernel argument index to its device base in *bytes*.
    Returns (instruction_text, applied) where `applied` lists
    (site, arg, units) for auditing.
    """
    _, records = split(vsm_text)
    seen = set()
    out = []
    applied = []
    in_sidecar = False
    # Site numbers restart per kernel entry, so a line only identifies a
    # record together with the `@mncore.kernel` marker that precedes it.
    kernel = ""

    for line in vsm_text.splitlines():
        # The sidecar block is metadata, not instruction text; assemble3 would
        # reject its `- site:` entries, so drop the whole block.
        if _BEGIN.match(line):
            in_sidecar = True
            continue
        if in_sidecar:
            if _END.match(line):
                in_sidecar = False
            continue
        km = _KERNEL.match(line)
        if km:
            kernel = km.group(1)
            continue  # our own metadata, not an instruction
        m = _SITE.search(line)
        if not m:
            out.append(line)
            continue
        site = int(m.group(1))
        key = (kernel, site)
        where = f"{kernel} site {site}" if kernel else f"site {site}"
        if key in seen:
            raise RelocError(f"{where} appears on more than one line")
        seen.add(key)
        rec = records.get(key)
        if rec is None:
            raise RelocError(f"line carries {where} but no sidecar entry")

        body = line[:m.start()].rstrip()
        tokens = body.split()
        if len(tokens) < 3:
            raise RelocError(f"{where}: not an MV line: {body!r}")
        if not tokens[0].startswith(rec["opcode"]):
            raise RelocError(
                f"{where}: sidecar says {rec['opcode']!r} but the line "
                f"is {tokens[0]!r}")
        idx = MV_OPERAND.get(rec["operand"])
        if idx is None:
            raise RelocError(
                f"{where}: operand {rec['operand']} is not an MV memory "
                "operand")
        mm = _MEM.match(tokens[idx])
        if not mm:
            raise RelocError(
                f"{where}: operand {tokens[idx]!r} is not a memory operand")
        kind = {"d": "DRAM", "p": "PDM", "lc": "L2BM", "di": "DRAM"}[mm.group(1)]
        if kind != rec["space"]:
            raise RelocError(
                f"{where}: sidecar says {rec['space']} but the operand is "
                f"{kind}")

        if rec["arg"] not in bindings:
            raise RelocError(f"{where}: argument {rec['arg']} is unbound")
        base_bytes = bindings[rec["arg"]]
        unit = UNIT_BYTES[rec["unit"]]
        if base_bytes % unit != 0:
            raise RelocError(
                f"{where}: base {base_bytes} B is not a multiple of the "
                f"{rec['unit']} unit ({unit} B)")
        value = base_bytes // unit + rec["addend_units"]
        if value < 0 or value >= (1 << rec["width_bits"]):
            raise RelocError(
                f"{where}: address {value} does not fit "
                f"{rec['width_bits']} bits")

        tokens[idx] = f"${mm.group(1)}{value}{mm.group(3)}"
        out.append(" ".join(tokens))
        applied.append((kernel, site, rec["arg"], value))

    unbound = [k for k in records if k not in seen]
    if unbound:
        raise RelocError(f"site(s) {unbound} are described but never used")
    return "\n".join(out) + "\n", applied
