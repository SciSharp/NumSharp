# Tests & Oracle inventory

Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.

## Headline

| Measure | Count |
|---|---:|
| MSTest method declarations | 13,831 |
| DataRow cases / methods | 716 / 80 |
| DynamicData methods (not expanded) | 34 |
| Default-run declarations | 13,536 |
| Open-bug methods | 215 |
| Oracle-tagged methods | 664 |
| Oracle test classes | 18 |
| FuzzMatrix methods | 98 |
| Corpus Oracle test cases | 116,969 |
| Specialized flags/layout/NPY test cases | 1,600 |
| Total Oracle test cases | 118,569 |
| Oracle operation keys | 367 |
| Host-pin metadata records (not cases) | 2 |

## Test projects

| Project | Methods |
|---|---:|
| `NumSharp.Tests` | 13,221 |
| `NumSharp.Tests.Interop` | 444 |
| `NumSharp.Tests.Oracle` | 166 |

## Oracle evidence review

- Oracle operations below 10 test cases: **64**
- Oracle operations recording exactly one layout: **75**
- Oracle operations recording exactly one dtype: **12**
- Oracle operations with explicit error test cases: **37** (1,338 test cases)
- Absence of an error test case is neutral unless the API defines invalid inputs.
