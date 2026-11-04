"""Records what NumPy 2.4.2 on CPython 3.12 answers for the codecs NumSharp refuses: the observations
``docs/plans/strings-codecs.md`` (section 2) quotes, one line per call, ``expression -> repr(result)`` or
``expression -> ExceptionType: text``.

The mbcs / oem lines depend on the host's ANSI / OEM code pages (the discovery ran on 1252 / 437) and exist on Windows
only; elsewhere those names are ``LookupError: unknown encoding``. Run it with the oracle interpreter:

    python -I docs/plans/strings-codecs/probe_numpy.py
"""
import codecs
import io
import os
import sys
import tempfile
import warnings

import numpy as np


def show(label, fn):
    """Prints one observation.

    :param label: the expression as the discovery quotes it.
    :param fn: a callable computing it.
    """
    try:
        print(f"{label} -> {fn()!r}")
    except BaseException as e:  # noqa: BLE001 - every outcome is the observation
        print(f"{label} -> {type(e).__name__}: {e}")


def charmap_and_code_pages():
    """``charmap`` (Latin-1 under another name) and the Windows code-page codecs ``mbcs`` / ``oem``."""
    print("== charmap / mbcs / oem ==")
    show("'\\u0100'.encode('charmap')", lambda: "Ā".encode("charmap"))
    show("b'\\xff'.decode('charmap')", lambda: b"\xff".decode("charmap"))
    show("b'\\x81'.decode('mbcs')", lambda: b"\x81".decode("mbcs"))
    show("b'\\x81'.decode('cp1252')", lambda: b"\x81".decode("cp1252"))
    show("'\\u0100'.encode('mbcs')", lambda: "Ā".encode("mbcs"))
    show("'\\u0100'.encode('mbcs', 'replace')", lambda: "Ā".encode("mbcs", "replace"))
    show("'\\u0100'.encode('cp1252', 'replace')", lambda: "Ā".encode("cp1252", "replace"))
    show("'\\u20ac'.encode('oem')", lambda: "€".encode("oem"))
    show("'\\udc80'.encode('mbcs', 'surrogateescape')", lambda: "\udc80".encode("mbcs", "surrogateescape"))


def cjk():
    """The multi-byte CJK codecs through ``np.strings``: tables, error reasons, state, pairs, handlers."""
    print("== CJK ==")
    enc, dec = np.strings.encode, np.strings.decode
    show("encode(['中文', 'abc'], 'gbk')", lambda: enc(np.array(["中文", "abc"]), "gbk"))
    show("encode(['日本語abc'], 'iso2022_jp')", lambda: enc(np.array(["日本語abc"]), "iso2022_jp"))
    show("encode(['€'], 'shift_jis')", lambda: enc(np.array(["€"]), "shift_jis"))
    show("encode(['a€b'], 'shift_jis', 'replace')", lambda: enc(np.array(["a€b"]), "shift_jis", "replace"))
    show("encode(['日€b'], 'iso2022_jp', 'xmlcharrefreplace')",
         lambda: enc(np.array(["日€b"]), "iso2022_jp", "xmlcharrefreplace"))
    show("encode(['中~', 'a~b'], 'hz')", lambda: enc(np.array(["中~", "a~b"]), "hz"))
    show("encode(['中€'], 'hz', 'backslashreplace')", lambda: enc(np.array(["中€"]), "hz", "backslashreplace"))
    show("encode(['\\udc80'], 'gbk', 'surrogateescape')", lambda: enc(np.array(["\udc80"]), "gbk", "surrogateescape"))
    show("encode(['\\ud800'], 'gbk', 'surrogatepass')", lambda: enc(np.array(["\ud800"]), "gbk", "surrogatepass"))
    show("encode(['€'], 'gbk', 'namereplace')", lambda: enc(np.array(["€"]), "gbk", "namereplace"))
    show("decode([b'\\x82'], 'shift_jis')", lambda: dec(np.array([b"\x82"]), "shift_jis"))
    show("decode([b'\\x82\\x20'], 'shift_jis')", lambda: dec(np.array([b"\x82\x20"]), "shift_jis"))
    show("decode([b'\\xff'], 'gbk', 'backslashreplace')", lambda: dec(np.array([b"\xff"]), "gbk", "backslashreplace"))
    show("decode([b'\\xff'], 'gbk', 'xmlcharrefreplace')", lambda: dec(np.array([b"\xff"]), "gbk", "xmlcharrefreplace"))
    show("decode([b'\\x81\\x30\\x81\\x30'], 'gb18030')", lambda: dec(np.array([b"\x81\x30\x81\x30"]), "gb18030"))
    show("decode([b'\\x95\\x32\\x82\\x36'], 'gb18030')", lambda: dec(np.array([b"\x95\x32\x82\x36"]), "gb18030"))
    show("decode([b'\\x88\\x62'], 'big5hkscs')", lambda: dec(np.array([b"\x88\x62"]), "big5hkscs"))
    show("decode([b'\\xa4\\xf7'], 'euc_jis_2004')", lambda: dec(np.array([b"\xa4\xf7"]), "euc_jis_2004"))
    show("encode(['\\u304b\\u309a'], 'euc_jis_2004')", lambda: enc(np.array(["か゚"]), "euc_jis_2004"))
    show("decode([b'\\x8f\\xa2\\xaf'], 'euc_jp')", lambda: dec(np.array([b"\x8f\xa2\xaf"]), "euc_jp"))
    d = codecs.getincrementaldecoder("iso2022_jp")()
    show("iso2022_jp incremental decode split inside an escape", lambda: (d.decode(b"\x1b$"), d.decode(b"BF|\x1b(Ba", True)))
    e = codecs.getincrementalencoder("iso2022_jp")()
    show("iso2022_jp incremental encode then final", lambda: (e.encode("日"), e.encode("", final=True)))


def utf7_and_escapes():
    """``utf-7``, ``unicode_escape`` and ``raw_unicode_escape`` through ``np.strings``."""
    print("== utf-7 / escape codecs ==")
    enc, dec = np.strings.encode, np.strings.decode
    show("encode(['a+b~中\\U0001F600'], 'utf-7')", lambda: enc(np.array(["a+b~中\U0001F600"]), "utf-7"))
    show("encode(['\\ud800'], 'utf-7')", lambda: enc(np.array(["\ud800"]), "utf-7"))
    show("decode([b'+2D3eAA-'], 'utf-7')", lambda: dec(np.array([b"+2D3eAA-"]), "utf-7"))
    show("decode([b'a+\\x80'], 'utf-7')", lambda: dec(np.array([b"a+\x80"]), "utf-7"))
    show("decode([b'+2D3'], 'utf-7')", lambda: dec(np.array([b"+2D3"]), "utf-7"))
    show("encode(['a\\\\b\\té€\\U0001F600\\ud800'], 'unicode_escape')",
         lambda: enc(np.array(["a\\b\té€\U0001F600\ud800"]), "unicode_escape"))
    show("encode(['a\\\\b\\té€\\U0001F600'], 'raw_unicode_escape')",
         lambda: enc(np.array(["a\\b\té€\U0001F600"]), "raw_unicode_escape"))
    for name in (b"euro sign", b"null", b"KEYCAP NUMBER SIGN", b"HANGUL SYLLABLE GA", b"hangul syllable ga",
                 b"CJK UNIFIED IDEOGRAPH-4E00", b"CJK UNIFIED IDEOGRAPH-4e00"):
        data = b"\\N{" + name + b"}"
        show(f"decode([{data!r}], 'unicode_escape')", lambda data=data: dec(np.array([data]), "unicode_escape"))
    show("decode([b'\\\\q'], 'unicode_escape')", lambda: dec(np.array([b"\\q"]), "unicode_escape"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        show("decode([b'\\\\q'], 'unicode_escape') under -W error", lambda: dec(np.array([b"\\q"]), "unicode_escape"))
    show("decode([b'a\\\\x4'], 'unicode_escape')", lambda: dec(np.array([b"a\\x4"]), "unicode_escape"))
    show("decode([b'\\\\U00110000'], 'raw_unicode_escape')", lambda: dec(np.array([b"\\U00110000"]), "raw_unicode_escape"))


def idna_and_punycode():
    """``idna`` (RFC 3490 over nameprep) and ``punycode`` (RFC 3492) through ``np.strings``."""
    print("== idna / punycode ==")
    enc, dec = np.strings.encode, np.strings.decode
    show("encode(['münchen.de', 'Bücher.example.'], 'idna')", lambda: enc(np.array(["münchen.de", "Bücher.example."]), "idna"))
    show("encode(['a..b'], 'idna')", lambda: enc(np.array(["a..b"]), "idna"))
    show("encode(['ü'], 'idna', 'replace')", lambda: enc(np.array(["ü"]), "idna", "replace"))
    show("encode(['\\u0627a'], 'idna')", lambda: enc(np.array(["اa"]), "idna"))
    show("decode([b'xn--mnchen-3ya.de', b'ABC.de'], 'idna')", lambda: dec(np.array([b"xn--mnchen-3ya.de", b"ABC.de"]), "idna"))
    show("decode([b'\\xff.de'], 'idna')", lambda: dec(np.array([b"\xff.de"]), "idna"))
    show("encode(['bücher', 'abc', '中文'], 'punycode')", lambda: enc(np.array(["bücher", "abc", "中文"]), "punycode"))
    show("decode([b'zzzzzzzzzzzzzzzzzzzzzzzzzz'], 'punycode')", lambda: dec(np.array([b"z" * 26]), "punycode"))
    show("decode([b'\\xffa-!b'], 'punycode', 'replace')", lambda: dec(np.array([b"\xffa-!b"]), "punycode", "replace"))
    show("decode([b'a-!'], 'punycode', 'backslashreplace')", lambda: dec(np.array([b"a-!"]), "punycode", "backslashreplace"))


def text_io():
    """``np.loadtxt`` / ``np.savetxt`` with refused codecs: a file NumPy opens (TextIOWrapper) and a byte stream."""
    print("== loadtxt / savetxt ==")
    tmp = tempfile.mkdtemp()
    sjis = os.path.join(tmp, "sjis.txt")
    with open(sjis, "wb") as f:
        f.write("名前 値\n東京 1\n".encode("shift_jis"))
    show("loadtxt(<shift_jis file>, dtype=str, encoding='shift_jis')", lambda: np.loadtxt(sjis, dtype=str, encoding="shift_jis"))
    show("loadtxt(BytesIO(<gbk>), delimiter=',', encoding='gbk')",
         lambda: np.loadtxt(io.BytesIO("1,2\n3,4\n".encode("gbk")), delimiter=",", encoding="gbk"))
    out = os.path.join(tmp, "out.txt")

    def save_file(encoding, newline):
        np.savetxt(out, np.array([["日本", "語"]]), fmt="%s", newline=newline, encoding=encoding)
        with open(out, "rb") as f:
            return f.read()

    def save_stream(encoding, newline):
        b = io.BytesIO()
        np.savetxt(b, np.array([["中", "文"]]), fmt="%s", newline=newline, encoding=encoding)
        return b.getvalue()

    show("savetxt(<file>, [['日本', '語']], encoding='iso2022_jp')", lambda: save_file("iso2022_jp", "\n"))
    show("savetxt(<file>, ..., newline='', encoding='iso2022_jp')", lambda: save_file("iso2022_jp", ""))
    show("savetxt(BytesIO, [['中', '文']], newline='', encoding='hz')", lambda: save_stream("hz", ""))
    u7 = os.path.join(tmp, "u7.txt")
    with open(u7, "wb") as f:
        f.write("é 1\n中 2\n".encode("utf-7"))
    show("loadtxt(<utf-7 file>, dtype=str, encoding='utf-7')", lambda: np.loadtxt(u7, dtype=str, encoding="utf-7"))
    show("loadtxt(BytesIO(b'xn--mnchen-3ya.de 1'), dtype=str, encoding='idna')",
         lambda: np.loadtxt(io.BytesIO(b"xn--mnchen-3ya.de 1\n"), dtype=str, encoding="idna"))


def main():
    """Runs every probe group."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"NumPy {np.__version__}, CPython {sys.version.split()[0]} on {sys.platform}")
    charmap_and_code_pages()
    cjk()
    utf7_and_escapes()
    idna_and_punycode()
    text_io()


if __name__ == "__main__":
    main()
