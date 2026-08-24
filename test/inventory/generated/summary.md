# Tests & Oracle inventory

Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.

## Headline

| Measure | Count |
|---|---:|
| Test methods | 13,760 |
| Declared invocations | 14,396 |
| Active methods | 13,465 |
| Open-bug methods | 215 |
| Oracle-owned methods | 657 |
| FuzzMatrix methods | 91 |
| Committed op-corpus rows | 116,971 |
| Corpus op keys | 366 |
| Specialized flags/layout/NPY cases | 1,600 |

## Test projects

| Project | Methods |
|---|---:|
| `NumSharp.Tests` | 13,157 |
| `NumSharp.Tests.Interop` | 444 |
| `NumSharp.Tests.Oracle` | 159 |

## Oracle strength queue

- Ops below 10 cases: **64**
- Ops with one/no serialized layout: **77**
- Ops with one/no dtype: **20**
- Ops without a recorded error row: **329**
