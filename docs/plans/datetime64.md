# NumPy `datetime64` / `timedelta64` — representation, format, and operation support

> Investigation of how NumPy 2.4.2 represents and operates on temporal values, as
> reference for a future NumSharp port. All facts below were read from the vendored
> NumPy source at `refs/numpy/` and cross-checked against a live `numpy==2.4.2`
> (the project oracle version). File:line citations are into `refs/numpy/`.

---

## 0. Is `timedelta64` a separate dtype? — **Yes.**

`datetime64` and `timedelta64` are **two distinct dtypes**, not one dtype in two modes.
They share almost all *machinery* (identical storage, the same unit-metadata struct, the
same NaT sentinel, the same conversion factors) but differ in *type number*, *type letter*,
*scalar-type hierarchy*, and *semantics*.

| | `datetime64` | `timedelta64` |
|--|--|--|
| type number | `NPY_DATETIME` = **21** | `NPY_TIMEDELTA` = **22** |
| type letter / kind | `M` (a.k.a. `M8`) | `m` (a.k.a. `m8`) |
| scalar MRO | `datetime64 → generic → object` | `timedelta64 → signedinteger → integer → number → generic → object` |
| meaning | a **point in time** (a moment) | a **duration** (elapsed time) |
| `np.dtype('M8') == np.dtype('m8')` | — | **`False`** |

The scalar-type inheritance is the crux (`multiarraymodule.c:4814`):

```c
/* Datetime doesn't fit in any category */
SINGLE_INHERIT(Datetime, Generic);
/* Timedelta is an integer with an associated unit */
SINGLE_INHERIT(Timedelta, SignedInteger);
```

A `datetime64` is a bare moment that belongs to no numeric category; a `timedelta64` **is a
signed integer that carries a unit**. This single distinction drives every semantic
difference below:

- `timedelta + timedelta` is valid (durations add); `datetime + datetime` is **an error**
  (you cannot add two moments).
- `datetime − datetime → timedelta` (the gap between two moments is a duration).
- `datetime ± timedelta → datetime` (shift a moment by a duration).
- Reductions that need addition (`sum`, `mean`) work on `timedelta64` but **not** on
  `datetime64`.

`PyTypeNum_ISDATETIME(t)` is `NPY_DATETIME <= t <= NPY_TIMEDELTA`
(`ndarraytypes.h:1683`), so the two are treated as one *family* for dispatch but remain two
dtypes.

For a NumSharp port this means **two new `NPTypeCode` entries** (or one storage class with a
kind flag), not one.

---

## 1. Representation — one `int64` + a unit tag in the dtype

A value of either dtype is **a single signed `int64`** counting *ticks of a chosen unit*
from the **Unix epoch, 1970-01-01T00:00:00**. The unit lives in the **dtype metadata**, not
in the value or the array element.

```c
typedef npy_int64 npy_datetime;    // npy_common.h:983  — the datetime value
typedef npy_int64 npy_timedelta;   // npy_common.h:982  — the timedelta value

typedef struct {                   // ndarraytypes.h:861 — the unit metadata (on the dtype)
    NPY_DATETIMEUNIT base;         //   which unit (Y/M/W/D/h/m/s/ms/us/ns/ps/fs/as/generic)
    int num;                       //   a multiplier (datetime64[5m] → num = 5)
} PyArray_DatetimeMetaData;
```

- **`itemsize` is always 8 bytes**, regardless of unit or multiplier.
- **`num` is a unit multiplier.** `datetime64[5m]` stores *5-minute* ticks — verified:
  `np.datetime_data(np.datetime64('2024-03-15T13:45','5m').dtype) == ('m', 5)`, stored int
  `5701701` = 5701701 × 5 minutes since epoch. `np.datetime_data(dtype)` returns
  `(unit_str, num)`.
- The value is a plain count, so **dates before 1970 are negative**.
  - `2024-03-15T13:45:30` as `datetime64[us]` → `1710510330000000` (µs since 1970).
  - `1970-01-02` as `[D]` → `1`; `1970-02` as `[M]` → `1`; `1971` as `[Y]` → `1`.
  - `[Y]` counts **calendar years** since 1970, `[M]` counts **calendar months** since 1970
    — these are calendar indices, not elapsed durations.

### Exploded ("struct") form — used only for parse/format/calendar work

```c
typedef struct {                   // ndarraytypes.h:875
    npy_int64 year;
    npy_int32 month, day, hour, min, sec, us, ps, as;
} npy_datetimestruct;              // NaT is year == INT64_MIN
```

Sub-second precision is three int32 triples: `us` (microseconds, 6 digits), `ps` (down to
10⁻¹²), `as` (down to 10⁻¹⁸). The int64 ↔ struct conversions live in
`datetime.c`: `get_datetimestruct_days` (`:121`, days-from-1970 via a proleptic-Gregorian
calendar walk; `is_leapyear` at `:110`), `NpyDatetime_ConvertDatetimeStructToDatetime64`
(`:281`, struct → int64 per unit), and `set_datetimestruct_days` (`:253`, the inverse).

---

## 2. Units and the range ↔ resolution trade-off

14 units (`_datetime_strings`, `datetime.c:82`; enum `NPY_DATETIMEUNIT`,
`ndarraytypes.h:294`). Note the value-3 gap where the removed 1.6 "business day" unit sat.

| enum | `NPY_FR_Y` | `M` | `W` | `D` | `h` | `m` | `s` | `ms` | `us` | `ns` | `ps` | `fs` | `as` | `GENERIC` |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| value | 0 | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| factor to next finer | — | — | ×7 →D | ×24 →h | ×60 | ×60 | ×1000 | ×1000 | ×1000 | ×1000 | ×1000 | ×1000 | base | — |

(`_datetime_factors`, `datetime.c:956`.)

Because the payload is always int64, **the representable window depends entirely on the
unit** (empirically confirmed against 2.4.2):

| unit | representable range |
|--|--|
| `[Y]` | ± ~9.2 × 10¹⁸ years |
| `[s]` | year −292 277 022 657 … +292 277 026 596 |
| `[ms]`| ± ~2.9 × 10⁸ years |
| `[us]`| year −290 308 … +294 247 |
| **`[ns]`** | **1677-09-21 … 2262-04-11** (the well-known pandas limit) |
| `[ps]`| 1969-09-16 … 1970-04-17 (± ~7 months) |
| `[as]`| 1969-12-31T23:59:50 … 1970-01-01T00:00:09 (**± ~9.2 seconds**) |

`timedelta64[ns]` similarly maxes out at ~106 751 days (~292 years).

### Y and M are "non-linear" units

A year is not a fixed number of seconds, so Y/M → finer conversions use the **400-year leap
cycle average** (`get_datetime_conversion_factor`, `datetime.c:1072`):

- `Y → D` = `(97 + 400·365) / 400` = **365.2425** days
- `M → D` = `146097 / 4800` = **30.436875** days

As a consequence Y and M are **incompatible with D-and-finer units in `timedelta`
arithmetic** — see §5.

### `generic` unit

A `datetime64`/`timedelta64` with no unit (e.g. `np.datetime64('NaT')`) has the `generic`
unit — the default "unbound" unit. It adopts any concrete unit on first contact and **cannot
be instantiated except as NaT** ("Cannot create a NumPy datetime other than NaT with generic
units", `datetime.c:297`).

---

## 3. NaT — "Not a Time"

The missing/invalid sentinel is **`INT64_MIN`**:

```c
#define NPY_DATETIME_NAT NPY_MIN_INT64   // ndarraytypes.h:277  == -9223372036854775808
```

It behaves like an IEEE NaN, and applies to **both** dtypes:

- `np.isnat(nat)` → `True`; `nat == nat` → `False`; `nat != nat` → `True`; every ordering
  (`< <= > >=`) involving NaT → `False` (`loops.c.src:730`).
- It **propagates** through every arithmetic loop: any operand NaT → NaT result.
- `maximum`/`minimum` (and `fmax`/`fmin`) treat NaT as propagating, not skipping
  (`loops.c.src:760`).

---

## 4. String format — (almost) ISO 8601

Parsing: `datetime_strings.c::NpyDatetime_ParseISO8601Datetime` (`:223`). Grammar:

- `YYYY-MM-DDThh:mm:ss.ffffff…` — the `-` separators are **mandatory** (so `20100312` parses
  as *the year 20 100 312*, not a date). Either `T` **or a space** separates date and time.
- Only the seconds field may have a decimal point, up to **18** fractional digits
  (attosecond precision).
- **The unit is auto-detected from how much resolution the string carries** (verified):

  | string | detected unit |
  |--|--|
  | `'2024'` | `Y` |
  | `'2024-03'` | `M` |
  | `'2024-03-15'` | `D` |
  | `'2024-03-15T13'` | `h` |
  | `'2024-03-15T13:45'` | `m` |
  | `'2024-03-15T13:45:30'` | `s` |
  | `'...30.123'` | `ms` |
  | `'...30.123456'` | `us` |
  | `'...30.123456789'` | `ns` |

- Special values: `''` / `'NaT'` (case-insensitive → generic NaT), `'today'` (→ `D`, local
  date), `'now'` (→ `s`, UTC).
- **`datetime64` is timezone-naive** (since NumPy 1.11). A trailing `Z` / `±HH:MM` offset is
  *parsed and applied*, then discarded; the stored value has no zone.
- Does **not** handle: `YYYY-DDD` ordinal / `YYYY-Www` week dates, leap seconds (`:60`),
  or `24:00:00`.

Formatting: `NpyDatetime_MakeISO8601Datetime` (`:893`), surfaced as
`np.datetime_as_string(arr, unit=None|'auto'|<unit>, timezone='naive'|'UTC'|'local', casting='same_kind')`.
`timezone='UTC'` appends a `Z`.

---

## 5. Operation support (verified against 2.4.2)

Arithmetic ufunc loops: `refs/numpy/numpy/_core/src/umath/loops.c.src`. Naming convention in
the loop names: **`M`** = datetime, **`m`** = timedelta, **`q`** = int64, **`d`** = double.
**Every loop NaT-propagates.** The result *unit* of a binary op is the **GCD of the two
operand units** (`compute_datetime_metadata_greatest_common_divisor`, `datetime.c:1468`),
e.g. `1h + 30m → 90m`.

### Supported ✅

| operation | signature | loop (`loops.c.src`) |
|--|--|--|
| datetime + timedelta (either order) | `M+m → M`, `m+M → M` | `DATETIME_Mm_M_add` `:806`, `DATETIME_mM_M_add` `:821` |
| datetime − timedelta | `M−m → M` | `DATETIME_Mm_M_subtract` `:851` |
| **datetime − datetime** | `M−M → m` | `DATETIME_MM_m_subtract` `:866` |
| timedelta ± timedelta | `m±m → m` | `:836`, `:881` |
| timedelta × int / float (either order) | `m*q, q*m, m*d, d*m → m` | `:897`–`:967` |
| timedelta ÷ int / float | `m/q, m/d → m` | `:971`, `:1020` |
| **timedelta ÷ timedelta** | `m/m → float64` | `TIMEDELTA_mm_d_divide` `:1041` |
| timedelta `//` timedelta | `m//m → int64` | `TIMEDELTA_mm_q_floor_divide` `:1084` |
| timedelta `%` timedelta | `m%m → m` | `TIMEDELTA_mm_m_remainder` `:1056` |
| `divmod(timedelta, timedelta)` | `→ (int64, m)` | `TIMEDELTA_mm_qm_divmod` `:1155` |
| unary `-`, `+`, `abs`, `sign` (timedelta) | `m → m` / `m → int` | `:650`–`:693` |
| `== != < <= > >=` | NaN-like NaT semantics, `→ bool` | `:730` |
| `maximum`/`minimum`/`fmax`/`fmin` | NaT-propagating | `:760` |
| `isnat`, `isfinite` | `→ bool` | `:701`, `:710` |
| `min`/`max`/`argmin`/`argmax`/`sort`/`ptp`/`diff` | via min/max/subtract | ✅ both dtypes |
| `sum`, `mean`, `cumsum` | **timedelta only** | ✅ |
| `np.arange(start, stop, step)` | datetime-aware generator | `datetime_arange` `datetime.c:3348` |
| `astype` between units | with the cast barrier of §6 | casting `datetime.c:1232` |

### Not supported ✗ (these raise)

- `datetime + datetime` → `UFuncTypeError`. Therefore **`sum` / `mean` / `median` of
  `datetime64` all fail** (they need add-reduce, which does not exist for moments).
- **`std` / `var` / `prod` of `timedelta64`** → need `square`, which has no timedelta loop →
  `TypeError: ufunc 'square' not supported`.
- Mixing **non-linear Y/M with D-and-finer units in `timedelta` arithmetic**: `1Y + 1D` →
  `TypeError: ... incompatible nonlinear base time units`. But `1Y + 1M → 13M` is fine
  (Y and M *are* mutually compatible), and **`datetime` arithmetic is relaxed** — e.g.
  `datetime64[D] + timedelta64[s]` works, promoting the result to `[s]`. The strictness is a
  per-operand flag: strict for `timedelta` operands, relaxed for `datetime`
  (`datetime_type_promotion`, `datetime.c:1624`, and the `strict_with_nonlinear_units`
  parameters of the GCD routine).

---

## 6. Casting rules (unit-to-unit)

`can_cast_datetime64_units` (`datetime.c:1232`), `can_cast_timedelta64_units` (`:1281`):

- **`unsafe`** — any unit → any unit.
- **`safe`** — only **coarser → finer** (`src_unit <= dst_unit` in enum order), because a
  finer→coarser cast loses information. So `D → s` is safe ✅, `s → D` is **not** ✅→✗
  (verified: `np.can_cast('datetime64[D]','datetime64[s]','safe')` → True; the reverse →
  False). `same_kind` allows any unit.
- **timedelta additionally enforces a hard date-unit vs time-unit barrier** for anything but
  `unsafe`: `{Y, M}` on one side and `{W, D, h, …}` on the other cannot cast into each other
  (the non-linear-unit barrier again).
- `generic` casts freely *to* a concrete unit but a concrete unit cannot cast *to* generic.
- `.astype('datetime64[D]')` from a finer unit **truncates** toward the epoch.

Type promotion for two temporal dtypes: `datetime_type_promotion` (`datetime.c:1624`) →
picks datetime if either side is datetime, else timedelta, with the unit = GCD of the two.

---

## 7. Business-day API

`datetime_busday.c` (1327 lines) + `datetime_busdaycal.c` (calendar object), registered in
`multiarraymodule.c:4627`:

- `np.busday_count(begindates, enddates, weekmask='1111100', holidays=(), busdaycal=None, out=None)`
  — business days in the half-open `[begin, end)`.
- `np.busday_offset(dates, offsets, roll='raise'|'forward'|'backward'|'following'|'preceding'|'modifiedfollowing'|'modifiedpreceding'|'nat', weekmask='1111100', holidays=None, busdaycal=None, out=None)`
  — shift by N business days. (`roll` values ↔ `NPY_BUSDAY_ROLL`, `ndarraytypes.h:327`.)
- `np.is_busday(dates, weekmask='1111100', holidays=(), busdaycal=None, out=None)`.
- `np.busdaycalendar(weekmask, holidays)` — a precomputed calendar object.

`weekmask` is a 7-character Mon…Sun string of `0`/`1`. All of these operate on
`datetime64[D]` (finer inputs are unsafe-cast to `[D]` first).

---

## 8. Python / scalar interop

`.item()` conversion (contract in `_datetime.h:243`, verified):

| numpy value | Python object from `.item()` |
|--|--|
| `datetime64[D]` and coarser | `datetime.date` |
| `datetime64[us]`…`[m]`/`[h]` | `datetime.datetime` |
| **`datetime64[ns]` and finer** | plain **`int`** (too fine for `datetime`) |
| `timedelta64[us]` and coarser | `datetime.timedelta` |
| `timedelta64[ns]` and finer | plain **`int`** |
| any NaT | `None` |

Construction accepts Python `datetime.datetime` / `datetime.date` /
`datetime.timedelta`, ISO strings, and `(int, unit)` pairs. `np.datetime_data(dtype)` →
`(unit_str, num)`.

---

## 9. Comparison to C# / .NET

| | NumPy `datetime64` / `timedelta64` | C# `System.DateTime` / `TimeSpan` |
|--|--|--|
| **Storage** | `int64` + a unit tag on the dtype | `DateTime`: `ulong` (62-bit ticks + 2-bit `Kind`); `TimeSpan`: `int64` ticks |
| **Unit** | **variable** — 14 units × a multiplier | **fixed** — 1 tick = 100 ns, always |
| **Epoch** | 1970-01-01 (Unix) | **0001-01-01** (proleptic Gregorian) |
| **Resolution** | attoseconds … years, chosen per array | 100 ns, fixed (no sub-100 ns) |
| **Range** | unit-dependent (see §2) | fixed years **1 … 9999** |
| **Missing value** | **NaT** (`INT64_MIN` sentinel) | none — non-nullable struct (`DateTime?` / `MinValue` by convention) |
| **Timezone** | naive (offset parsed then discarded) | `DateTime.Kind` (Utc/Local/Unspecified); `DateTimeOffset` carries an explicit offset |
| **Calendar (Y/M) math** | non-linear units, 400-yr-avg conversion, incompatible with sub-day | `AddMonths`/`AddYears` on `DateTime`; `TimeSpan` has **no** month/year concept |
| **Duration type** | `timedelta64` (int64 + unit) — a *separate dtype* | `TimeSpan` (int64 ticks) |
| Other analogues | — | `DateOnly` (day-number int), `TimeOnly` (ticks-since-midnight), .NET 6+ |

Mental model: NumPy stores **a `long` plus an enum saying what the `long` counts**, whereas
C# fixes that enum at "100-ns ticks from year 1". NumPy trades a fixed calendar range for a
per-array precision/range choice, and adds a NaN-like NaT that C# lacks.

---

## 10. NumSharp current state

- `src/NumSharp.Core/DateTime64.cs` (563 lines) already exists, but only as a **conversion
  helper struct**, *not* a real `NPTypeCode` dtype. Important: it deliberately uses the **C#
  convention** — `long _ticks` at **100 ns**, epoch **0001-01-01** — *not* NumPy's
  unit-parameterized / 1970-epoch model. It implements NaT (`long.MinValue`) with NumPy
  comparison semantics (NaT never compares equal, orderings false), while `Equals` follows
  .NET bit-equality so it can be a dictionary key. It exists so
  `Converts.ToDateTime64(x)` / `Converts.ToX(DateTime64)` can round-trip; it exposes **no**
  calendar arithmetic (delegate to `System.DateTime`).
- `src/NumSharp.Core/Creation/np.dtype.cs` recognizes the `M`/`m` type letters
  (`NPY_DATETIMELTR = 'M'`, `NPY_TIMEDELTALTR = 'm'`) and the strings `"datetime64"` /
  `"timedelta64"`, but they are **parsed then rejected** (`NotSupportedException`), and the
  `.npy` loader (`IO/NpyFormat.cs`) rejects `datetime64`/`timedelta64` descriptors the same
  way.
- **There is no real `datetime64` / `timedelta64` NPTypeCode**, no engine kernels, no
  arithmetic, no busday API.

A full port would need **two dtypes** (§0), each an int64 payload plus per-dtype unit
metadata (a NumSharp analogue of `PyArray_DatetimeMetaData` living on `DType`/`Shape`-side
metadata, since NumSharp's dtype has no per-array parameter slot today), the NaT sentinel,
the unit-conversion/GCD/casting rules, the arithmetic ufunc surface of §5, ISO-8601
parse/format, and (optionally) the busday family.

---

## 11. Source-file map (`refs/numpy/numpy/_core/`)

| file | contents |
|--|--|
| `include/numpy/ndarraytypes.h` | `NPY_DATETIMEUNIT` enum (`:294`), `PyArray_DatetimeMetaData` (`:861`), `npy_datetimestruct` (`:875`), `NPY_DATETIME_NAT` (`:277`), `NPY_BUSDAY_ROLL` (`:327`) |
| `include/numpy/npy_common.h` | `npy_datetime`/`npy_timedelta` = `npy_int64` (`:982`) |
| `src/multiarray/datetime.c` | representation, epoch/calendar math, unit factors + conversion, GCD/promotion, casting rules, `datetime_arange`, hashing (4375 lines) |
| `src/multiarray/datetime_strings.c` | ISO-8601 parse (`:223`) / format (`:893`) + `datetime_as_string` |
| `src/multiarray/datetime_busday.c` / `datetime_busdaycal.c` | business-day engine + calendar object |
| `src/multiarray/_datetime.h` | the internal C API surface (conversions, casts, promotion) |
| `src/umath/loops.c.src` (`:643`–`:1183`) | the arithmetic / comparison / min-max / isnat kernels |
| `src/umath/ufunc_type_resolution.c` | per-ufunc type+unit resolution: add `:723`, subtract `:909`, multiply `:1076`, divide `:1242`, remainder `:1350` |
| `multiarraymodule.c:4627` | Python-level `busday_*` / `is_busday` registration; `:4814` scalar-type inheritance |

---

### TL;DR

- **`datetime64` (`M8`, type 21) and `timedelta64` (`m8`, type 22) are two separate dtypes** —
  same int64+unit storage and same NaT, but datetime is a *moment* (inherits `generic`) and
  timedelta is a *duration* (inherits `signedinteger`), which is exactly why `td+td` works,
  `dt+dt` errors, and `dt−dt→td`.
- Each value is **one `int64` counting ticks of a per-dtype unit from 1970**; the unit
  (14 choices × a multiplier) lives on the dtype, and the range you get is entirely a
  function of that unit (`[ns]` → 1677–2262, `[as]` → ±9.2 s).
- **NaT = `INT64_MIN`**, a NaN-like sentinel that propagates and never compares equal.
- Supported: datetime±timedelta, datetime−datetime→timedelta, all timedelta arithmetic incl.
  `//`/`%`/`divmod`/`m÷m→float`, comparisons, min/max/sort/ptp/diff, timedelta
  `sum`/`mean`/`cumsum`, unit `astype`, `arange`, ISO-8601 strings, and the busday family.
  Unsupported: `datetime+datetime` (so datetime `sum`/`mean`/`median`), timedelta
  `std`/`var`/`prod`, and mixing non-linear Y/M with sub-day units in timedelta math.
- **vs C#**: variable unit vs fixed 100-ns ticks, 1970 vs year-1 epoch, per-array
  range/precision vs fixed years 1–9999, timezone-naive vs `Kind`/`DateTimeOffset`, and a
  first-class NaT that `DateTime` has no equivalent of.
- **NumSharp today** has only a `DateTime64` *conversion helper* (C#-tick/year-1 model) and
  recognizes the `M`/`m` letters, but the dtypes themselves are parsed-then-rejected — a real
  port needs two new dtypes with unit metadata.
