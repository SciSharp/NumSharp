# Tests & Oracle inventory

Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.

## Headline

| Measure | Count |
|---|---:|
| MSTest method declarations | 14,264 |
| DataRow cases / methods | 732 / 84 |
| DynamicData methods (not expanded) | 34 |
| Default-run declarations | 13,998 |
| Open-bug methods | 186 |
| Oracle-tagged methods | 748 |
| Oracle test classes | 18 |
| FuzzMatrix methods | 102 |
| Corpus Oracle test cases | 117,917 |
| Specialized flags/layout/NPY test cases | 1,600 |
| Total Oracle test cases | 119,517 |
| Oracle operation keys | 372 |
| Host-pin metadata records (not cases) | 2 |

## Test projects

| Project | Methods |
|---|---:|
| `NumSharp.Tests` | 13,570 |
| `NumSharp.Tests.Interop` | 524 |
| `NumSharp.Tests.Oracle` | 170 |

## Oracle evidence review

- Oracle operations below 10 test cases: **64**
- Oracle operations recording exactly one layout: **75**
- Oracle operations recording exactly one dtype: **12**
- Oracle operations with explicit error test cases: **37** (1,338 test cases)
- Absence of an error test case is neutral unless the API defines invalid inputs.
