# Vendored A2A schema

- `a2a-v1.0.1.json`: the A2A protocol's JSON Schema bundle for release v1.0.1 (tag `v1.0.1` of
  https://github.com/a2aproject/A2A, released 2026-05-28), fetched from
  https://a2a-protocol.org/v1.0.1/spec/a2a.json on 2026-11-18.
- sha256: `ba4b702b5edbdcbfe972c272484eb38e1fcd72af3c0b7eaa114b50e97455d8dd`
- The bundle is non-normative (generated from `specification/a2a.proto`). Its `version` field says only
  `v1`, so the tag above pins it. Tests also check what the schema leaves out: the proto's `REQUIRED`
  field behaviours, camelCase field names only, and enum names only (`tests/a2a_schema.py`).
