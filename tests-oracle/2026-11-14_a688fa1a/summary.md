# Tests & Oracle inventory

Deterministic inventory of MSTest declarations and committed oracle evidence. It is not a runtime pass rate.

## Headline

| Measure | Count |
|---|---:|
| MSTest method declarations | 21,115 |
| DataRow cases / methods | 4,208 / 182 |
| DynamicData methods (not expanded) | 34 |
| Default-run declarations | 20,786 |
| Open-bug methods | 242 |
| Oracle-tagged methods | 1,295 |
| Oracle test classes | 31 |
| FuzzMatrix methods | 242 |
| Corpus Oracle test cases | 838,505 |
| Specialized flags/layout/NPY/string test cases | 283,743 |
| String dtype test cases | 281,526 |
| Total Oracle test cases | 1,122,248 |
| Oracle operation keys | 1,455 |
| Host-pin metadata records (not cases) | 8 |

## Test projects

| Project | Methods |
|---|---:|
| `NumSharp.Tests` | 19,883 |
| `NumSharp.Tests.Interop` | 802 |
| `NumSharp.Tests.Oracle` | 430 |

## Oracle evidence review

- Oracle operations below 10 test cases: **88**
- Oracle operations recording exactly one layout: **148**
- Oracle operations recording exactly one dtype: **36**
- Oracle operations with explicit error test cases: **650** (83,286 test cases)
- Absence of an error test case is neutral unless the API defines invalid inputs.
