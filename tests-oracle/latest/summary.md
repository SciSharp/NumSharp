# Tests & Oracle inventory

Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.

## Headline

| Measure | Count |
|---|---:|
| MSTest method declarations | 20,506 |
| DataRow cases / methods | 3,992 / 173 |
| DynamicData methods (not expanded) | 34 |
| Default-run declarations | 20,253 |
| Open-bug methods | 170 |
| Oracle-tagged methods | 1,291 |
| Oracle test classes | 31 |
| FuzzMatrix methods | 241 |
| Corpus Oracle test cases | 836,054 |
| Specialized flags/layout/NPY/string test cases | 283,251 |
| String dtype test cases | 281,297 |
| Total Oracle test cases | 1,119,305 |
| Oracle operation keys | 1,448 |
| Host-pin metadata records (not cases) | 8 |

## Test projects

| Project | Methods |
|---|---:|
| `NumSharp.Tests` | 19,277 |
| `NumSharp.Tests.Interop` | 800 |
| `NumSharp.Tests.Oracle` | 429 |

## Oracle evidence review

- Oracle operations below 10 test cases: **90**
- Oracle operations recording exactly one layout: **148**
- Oracle operations recording exactly one dtype: **36**
- Oracle operations with explicit error test cases: **631** (82,385 test cases)
- Absence of an error test case is neutral unless the API defines invalid inputs.
