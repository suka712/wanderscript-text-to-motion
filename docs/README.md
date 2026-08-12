# Docs

Three files. Read in this order.

| | |
|---|---|
| **[IN_FLIGHT.md](IN_FLIGHT.md)** | **Read first.** What is running right now, where everything lives on the boxes, the next concrete command. Volatile — update or delete it when its work lands. |
| **[RESULTS.md](RESULTS.md)** | Everything established, one section per build-order stage, with the oracle each number is quoted against. |
| `../CLAUDE.md` | Architecture, locked vs swappable decisions, risks, build order. Authoritative for *what we are building*; RESULTS is authoritative for *what is true*. |

`archive/` is superseded material kept for provenance — the per-stage docs that RESULTS.md
replaces, dated progress reports, original step specs, the two parallel-track write-ups.
Nothing there is authoritative; if it disagrees with RESULTS.md, RESULTS.md wins.

## Conventions for adding a stage

- Append a section to `RESULTS.md` when a stage is **done**. Do not create per-stage files.
- Result first, then only method details that change how a number reads, then what it does
  not establish.
- Every generative number is quoted against an oracle. A model number without one is not
  interpretable.
- Bug fixes belong inside the stage they affect and never restructure the sequence.
