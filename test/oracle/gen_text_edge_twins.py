"""
Print numpy's answers for the offline text-edge twins as C# initializer rows.

test/NumSharp.Tests/Backends/Strings/TextBridgeEdgeTwinTests.cs holds the rows this script prints between the braces of
its Rows initializer: each is (label, dtype or null, elements or null, exception class chain or null, exception text or
null) - numpy 2.4.2's answer for one operation of the pythonnet tier's edge families (TextEdgeLiveParityTests) over the
four texts ['ab', 'c ', '', 'xyz']: cross-class operands, StringDType NA kinds, ufunc keywords and numpy.char
comparisons. The C# side computes the same operations in NumSharp and holds them to the rows, so the main suite (no
Python at test time) guards what the live tier proves.

    python test/oracle/gen_text_edge_twins.py > rows.txt     # then paste rows.txt into the Rows initializer

Requires numpy (2.4.2: the rows record its answers). A float prints as its repr, an S element as Latin-1 text, an NA as
null / double.NaN; an error row carries the exception class chain (most derived first) and str(e).
"""
import numpy as np
from numpy.dtypes import StringDType

DATA = ['ab', 'c ', '', 'xyz']


def make(kind):
    if kind == 'U': return np.array(DATA, dtype='U3')
    if kind == 'S': return np.array([d.encode() for d in DATA], dtype='S3')
    if kind == 'T': return np.array(DATA, dtype=StringDType())
    if kind == 'TNone':
        a = np.array(DATA, dtype=StringDType(na_object=None)); a[1] = None; return a
    if kind == 'Tnan':
        a = np.array(DATA, dtype=StringDType(na_object=np.nan)); a[1] = a.dtype.na_object; return a
    if kind == 'TNA':
        a = np.array(DATA, dtype=StringDType(na_object='NA')); a[1] = a.dtype.na_object; return a
    raise KeyError(kind)


def cs(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'



def value_token(v):
    if v is None: return 'null'
    if isinstance(v, float) and v != v: return 'double.NaN'
    if isinstance(v, (bool, np.bool_)): return 'true' if v else 'false'
    if isinstance(v, (bytes, np.bytes_)): return cs(v.decode('latin-1'))
    if isinstance(v, (int, np.integer)): return f'{int(v)}L'
    if isinstance(v, (float, np.floating)): return repr(float(v))
    return cs(str(v))


def row(label, f):
    try:
        r = np.asarray(f())
        vals = ', '.join(value_token(v) for v in r.ravel().tolist())
        dt = repr(r.dtype) if r.dtype.kind == 'T' else r.dtype.str
        return f'            ({cs(label)}, {cs(dt)}, new object[] {{ {vals} }}, null, null),'
    except Exception as e:
        mro = ','.join(c.__name__ for c in type(e).__mro__)
        return f'            ({cs(label)}, null, null, {cs(mro)}, {cs(str(e))}),'


kinds = ['U', 'S', 'T', 'TNone', 'Tnan', 'TNA']
print('// cross class')
for a in kinds:
    for b in kinds:
        print(row(f'{a} add {b}', lambda: np.strings.add(make(a), make(b)[::-1])))
        print(row(f'{a} equal {b}', lambda: np.equal(make(a), make(b)[::-1])))
        print(row(f'{a} less {b}', lambda: np.less(make(a), make(b)[::-1])))
print('// NA kinds')
for k in ['TNone', 'Tnan', 'TNA']:
    for name, f in [('str_len', lambda x: np.strings.str_len(x)), ('upper', lambda x: np.strings.upper(x)),
                    ('isalpha', lambda x: np.strings.isalpha(x)), ('isnan', lambda x: np.isnan(x)),
                    ('find l', lambda x: np.strings.find(x, 'b')), ('startswith a', lambda x: np.strings.startswith(x, 'a')),
                    ('sort', lambda x: np.sort(x)), ('argsort', lambda x: np.argsort(x, kind='stable')),
                    ('astype U4', lambda x: x.astype('U4')), ('astype float64', lambda x: x.astype('float64'))]:
        print(row(f'{k} {name}', lambda: f(make(k))))
print('// keywords')
u = make('U'); m = np.array([True, False, True, False])
print(row('str_len out int64 where', lambda: (lambda o: (np.strings.str_len(u, out=o, where=m), o)[1])(np.full(4, -1, np.int64))))
print(row('str_len out int8', lambda: (lambda o: (np.strings.str_len(u, out=o), o)[1])(np.full(4, -1, np.int8))))
print(row('str_len out float64', lambda: (lambda o: (np.strings.str_len(u, out=o), o)[1])(np.full(4, -1.0))))
print(row('str_len dtype=int32', lambda: np.strings.str_len(u, dtype=np.int32)))
print(row('equal out int64', lambda: (lambda o: (np.equal(u, u[::-1], out=o), o)[1])(np.full(4, 7, np.int64))))
print(row('add out U10 where', lambda: (lambda o: (np.strings.add(u, u, out=o, where=m), o)[1])(np.full(4, 'Q', dtype='U10'))))
print(row('add out U2', lambda: (lambda o: (np.strings.add(u, u, out=o), o)[1])(np.full(4, 'Q', dtype='U2'))))
print(row('add U out T', lambda: (lambda o: (np.strings.add(u, u, out=o), o)[1])(np.full(4, 'Q', dtype=StringDType()))))
print(row('add T out T', lambda: (lambda o: (np.strings.add(make('T'), make('T'), out=o), o)[1])(np.full(4, 'Q', dtype=StringDType()))))
print(row('add out S10 from U', lambda: (lambda o: (np.strings.add(u, u, out=o), o)[1])(np.full(4, b'Q', dtype='S10'))))
print('// numpy.char')
c1 = np.array(['ab ', 'c', 'x  ', '']); c2 = np.array(['ab', 'c  ', 'y', ' '])
print(row('char.equal', lambda: np.char.equal(c1, c2)))
print(row('char.less', lambda: np.char.less(c1, c2)))
print(row('np.equal', lambda: np.equal(c1, c2)))
print(row('chararray ==', lambda: np.char.array(c1) == np.char.array(c2)))
print(row('compare_chararrays == True', lambda: np.char.compare_chararrays(c1, c2, '==', True)))
print(row('compare_chararrays == False', lambda: np.char.compare_chararrays(c1, c2, '==', False)))
print(row('compare_chararrays >= True', lambda: np.char.compare_chararrays(c1, c2, '>=', True)))
