#!/usr/bin/env python3
"""Contract tests for the relocation reference consumer (§13.3/§13.5).

Positive: a placeholder address is replaced by `base + addend` and the fixup
comments are stripped so assemble3 sees plain instructions.

Negative (fail-closed, §13.5): an unbound site, a byte base that is not a
multiple of the unit, an address that does not fit `width_bits`, a space /
opcode / operand mismatch, and a described site that never appears are all
rejected.

Usage: python run_reloc.py
"""

import sys

import mncore_fixup as fx

HEAD = "mvp/n64i05 $d0 $lc256 # @mncore.fixup 0\n" \
       "mvp/n64i06 $lc320 $d0 # @mncore.fixup 1\n" \
       "nop/2\n"


def sidecar(*entries):
    out = ["# @mncore.fixup.begin", "# fixups:"]
    for e in entries:
        first = True
        for k, v in e.items():
            out.append(f"#   - {k}: {v}" if first else f"#     {k}: {v}")
            first = False
    out.append("# @mncore.fixup.end")
    return "\n".join(out) + "\n"


def ksidecar(groups):
    """Sidecar with per-kernel groups: [(kernel, [entry...]), ...]."""
    out = ["# @mncore.fixup.begin"]
    for kernel, entries in groups:
        out.append(f"# kernel: {kernel}")
        out.append("# fixups:")
        for e in entries:
            first = True
            for k, v in e.items():
                out.append(f"#   - {k}: {v}" if first else f"#     {k}: {v}")
                first = False
    out.append("# @mncore.fixup.end")
    return "\n".join(out) + "\n"


def rec(site, arg, operand, addend=0, space="DRAM", unit="longword",
        width=32, opcode="mvp"):
    return {"site": site, "arg": arg, "space": space, "unit": unit,
            "addend_units": addend, "width_bits": width, "opcode": opcode,
            "operand": operand}


def case(name, text, bindings, expect_error=None, expect_lines=None):
    try:
        out, applied = fx.relocate(text, bindings)
    except fx.RelocError as e:
        if expect_error is None:
            return f"{name}: unexpected rejection: {e}"
        if expect_error not in str(e):
            return f"{name}: wrong rejection: {e!r} (want {expect_error!r})"
        return None
    if expect_error is not None:
        return f"{name}: accepted but should have been rejected ({expect_error})"
    got = [l for l in out.splitlines() if l.strip()]
    if got != expect_lines:
        return f"{name}: got {got}, want {expect_lines}"
    return None


def main() -> int:
    good = sidecar(rec(0, 0, 1), rec(1, 2, 2, addend=64))
    cases = []

    # Positive: arg0 -> 0 longwords, arg2 -> 256 longwords (2048 B).
    cases.append(case("relocate", HEAD + good, {0: 0, 2: 2048}, None,
                      ["mvp/n64i05 $d0 $lc256", "mvp/n64i06 $lc320 $d320",
                       "nop/2"]))

    # A longword base that is not a multiple of 8 bytes is rejected, not
    # rounded (§13.1).
    cases.append(case("misaligned", HEAD + good, {0: 4, 2: 2048},
                      "not a multiple of the longword unit"))

    # An unbound argument is refused.
    cases.append(case("unbound", HEAD + good, {0: 0}, "argument 2 is unbound"))

    # 32-bit address field: 2**32 longwords no longer fits.
    cases.append(case("overflow", HEAD + good, {0: 0, 2: (1 << 32) * 8},
                      "does not fit 32 bits"))

    # A record attached to a line of the wrong opcode.
    cases.append(case("opcode-mismatch", HEAD + sidecar(rec(0, 0, 1),
                                                        rec(1, 2, 2, opcode="mvd")),
                      {0: 0, 2: 2048}, "sidecar says 'mvd'"))

    # The record must describe the memory space the operand actually uses.
    cases.append(case("space-mismatch", HEAD + sidecar(rec(0, 0, 1, space="PDM"),
                                                       rec(1, 2, 2)),
                      {0: 0, 2: 2048}, "sidecar says PDM"))

    # A described site that no line carries means the artifact is inconsistent.
    cases.append(case("unused-site",
                      "mvp/n64i05 $d0 $lc256 # @mncore.fixup 0\n" +
                      sidecar(rec(0, 0, 1), rec(1, 2, 2)),
                      {0: 0, 2: 2048}, "described but never used"))

    # A line with a site comment but no sidecar entry.
    cases.append(case("missing-entry", HEAD + sidecar(rec(0, 0, 1)),
                      {0: 0, 2: 2048}, "no sidecar entry"))

    # Unit = word: a 4-byte base is integral, an 8-byte one is too.
    wordy = "mvp/n64i05 $d0 $lc256 # @mncore.fixup 0\n" + \
        sidecar(rec(0, 3, 1, unit="word", addend=2))
    cases.append(case("word-unit", wordy, {3: 8}, None,
                      ["mvp/n64i05 $d4 $lc256"]))

    # A site number is only unique inside one kernel entry. Two kernels may
    # each own a site 0, and each line must resolve against its own kernel.
    two = ("# @mncore.kernel first\n"
           "mvp/n64i05 $d0 $lc256 # @mncore.fixup 0\n"
           "# @mncore.kernel second\n"
           "mvp/n64i06 $lc256 $d312 # @mncore.fixup 0\n")
    cases.append(case(
        "multi-kernel", two + ksidecar([("first", [rec(0, 0, 1)]),
                                        ("second", [rec(0, 1, 2, addend=64)])]),
        {0: 0, 1: 2048}, None,
        ["mvp/n64i05 $d0 $lc256", "mvp/n64i06 $lc256 $d320"]))

    # The second kernel's site 0 must NOT fall back to the first kernel's
    # record: (kernel entry, site) is the primary key, not the site alone.
    cases.append(case(
        "kernel-ambiguous", two + ksidecar([("first", [rec(0, 0, 1)])]),
        {0: 0, 1: 2048}, "second site 0 but no sidecar entry"))

    # The record collides only within one kernel; the same site number in two
    # kernels is legal and must not be reported as a duplicate.
    cases.append(case(
        "kernel-duplicate", two + ksidecar([("first", [rec(0, 0, 1),
                                                      rec(0, 0, 2)]),
                                            ("second", [rec(0, 1, 2)])]),
        {0: 0, 1: 2048}, "duplicate (kernel, site)"))

    # A negative addend below the base underflows the address field and is
    # rejected instead of wrapping.
    cases.append(case("negative-addend",
                      "mvp/n64i05 $d0 $lc256 # @mncore.fixup 0\n" +
                      sidecar(rec(0, 0, 1, addend=-8)),
                      {0: 0}, "does not fit 32 bits"))

    failures = [c for c in cases if c]
    for f in failures:
        print("FAIL  " + f, file=sys.stderr)
        return 1
    print(f"RELOC CONTRACT OK: {len(cases)} cases "
          "(2 positive, 10 rejections, 1 word-unit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
