using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using NumSharp.IO; // PyLiteral.Repr: Python's repr of the missing key, the KeyError text

namespace NumSharp
{
    /// <summary>
    ///     A read-only, insertion-ordered string-keyed table that behaves like the Python <c>dict</c> it ports: a missing key
    ///     raises NumPy's <see cref="KeyError"/> carrying Python's <c>str(KeyError(key))</c> text (the key's repr, so
    ///     <c>np.sctypeDict["a"]</c> fails with <c>'a'</c>), where a <see cref="Dictionary{TKey,TValue}"/> would throw .NET's
    ///     <see cref="KeyNotFoundException"/>.
    /// </summary>
    /// <typeparam name="TValue">The value type.</typeparam>
    /// <remarks>
    ///     Enumeration follows insertion order (the order the source dictionary was filled in without removals), which is
    ///     the order a Python dict iterates in — module tables such as <c>np.typecodes</c> and <c>np.sctypeDict</c> rely on
    ///     it. <see cref="TryGetValue"/> and <see cref="ContainsKey"/> never throw, as Python's <c>dict.get</c> and
    ///     <c>in</c> do not. The instance never changes after construction, so concurrent readers need no lock.
    ///     <para>
    ///     It also presents the read-only <see cref="IDictionary{TKey,TValue}"/> view .NET's
    ///     <see cref="System.Collections.ObjectModel.ReadOnlyDictionary{TKey,TValue}"/> (the type these tables used to be)
    ///     presents: a caller that casts to it can read, and every mutator throws <see cref="NotSupportedException"/>.
    ///     </para>
    /// </remarks>
    internal sealed class PyReadOnlyDict<TValue> : IReadOnlyDictionary<string, TValue>, IDictionary<string, TValue>
    {
        /// <summary>The table, filled once by the constructor and never mutated afterwards.</summary>
        private readonly Dictionary<string, TValue> _items;

        /// <summary>
        ///     Copies <paramref name="items"/> in its enumeration order (insertion order for a dictionary filled without
        ///     removals), so later changes to the source are not seen.
        /// </summary>
        /// <param name="items">The entries, in the order the Python dict holds them.</param>
        internal PyReadOnlyDict(IEnumerable<KeyValuePair<string, TValue>> items)
        {
            _items = new Dictionary<string, TValue>();
            foreach (var kv in items)
                _items.Add(kv.Key, kv.Value);
        }

        /// <summary>The value under <paramref name="key"/> (Python's <c>d[key]</c>).</summary>
        /// <param name="key">The key.</param>
        /// <returns>The value.</returns>
        /// <exception cref="KeyError">The key is absent: Python's <c>KeyError</c>, whose text is the key's repr
        ///     (<c>'a'</c>; a null key reads <c>None</c>, as Python's <c>d[None]</c> reports it).</exception>
        public TValue this[string key]
        {
            get
            {
                // A null key cannot be in the table (Dictionary refuses null keys); answer it like any other missing key
                // instead of letting Dictionary throw ArgumentNullException.
                if (key is not null && _items.TryGetValue(key, out var value))
                    return value;
                throw new KeyError(key is null ? "None" : PyLiteral.Repr(key));
            }
        }

        /// <inheritdoc/>
        public IEnumerable<string> Keys => _items.Keys;

        /// <inheritdoc/>
        public IEnumerable<TValue> Values => _items.Values;

        /// <inheritdoc/>
        public int Count => _items.Count;

        /// <summary>Whether <paramref name="key"/> is present (Python's <c>key in d</c>); never throws, a null key included.</summary>
        /// <param name="key">The key.</param>
        /// <returns>True when the key is present.</returns>
        public bool ContainsKey(string key) => key is not null && _items.ContainsKey(key);

        /// <summary>Looks <paramref name="key"/> up without throwing (Python's <c>d.get(key)</c>).</summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value when present, else the default.</param>
        /// <returns>True when the key is present.</returns>
        public bool TryGetValue(string key, [MaybeNullWhen(false)] out TValue value)
        {
            if (key is not null)
                return _items.TryGetValue(key, out value);
            value = default;
            return false;
        }

        /// <summary>The entries in insertion order (Python's <c>d.items()</c>).</summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator<KeyValuePair<string, TValue>> GetEnumerator() => _items.GetEnumerator();

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        // ---------------------------------------------------------------- the read-only IDictionary view

        /// <summary>The IDictionary indexer: reads as <see cref="this[string]"/> (a missing key is the
        ///     <see cref="KeyError"/>); a write is refused.</summary>
        /// <param name="key">The key.</param>
        /// <returns>The value.</returns>
        /// <exception cref="KeyError">Reading an absent key.</exception>
        /// <exception cref="NotSupportedException">Writing (the table is read-only).</exception>
        TValue IDictionary<string, TValue>.this[string key]
        {
            get => this[key];
            set => throw ReadOnly();
        }

        /// <inheritdoc/>
        ICollection<string> IDictionary<string, TValue>.Keys => _items.Keys;

        /// <inheritdoc/>
        ICollection<TValue> IDictionary<string, TValue>.Values => _items.Values;

        /// <summary>True: the table never changes.</summary>
        bool ICollection<KeyValuePair<string, TValue>>.IsReadOnly => true;

        /// <summary>Refused: the table is read-only.</summary>
        /// <param name="key">Unused.</param>
        /// <param name="value">Unused.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        void IDictionary<string, TValue>.Add(string key, TValue value) => throw ReadOnly();

        /// <summary>Refused: the table is read-only.</summary>
        /// <param name="key">Unused.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        bool IDictionary<string, TValue>.Remove(string key) => throw ReadOnly();

        /// <summary>Refused: the table is read-only.</summary>
        /// <param name="item">Unused.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        void ICollection<KeyValuePair<string, TValue>>.Add(KeyValuePair<string, TValue> item) => throw ReadOnly();

        /// <summary>Refused: the table is read-only.</summary>
        /// <exception cref="NotSupportedException">Always.</exception>
        void ICollection<KeyValuePair<string, TValue>>.Clear() => throw ReadOnly();

        /// <summary>Refused: the table is read-only.</summary>
        /// <param name="item">Unused.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        bool ICollection<KeyValuePair<string, TValue>>.Remove(KeyValuePair<string, TValue> item) => throw ReadOnly();

        /// <summary>Whether the entry is present with that value (the default equality of the value type).</summary>
        /// <param name="item">The entry.</param>
        /// <returns>True when present.</returns>
        bool ICollection<KeyValuePair<string, TValue>>.Contains(KeyValuePair<string, TValue> item)
            => ((ICollection<KeyValuePair<string, TValue>>)_items).Contains(item);

        /// <summary>Copies the entries, in insertion order, into <paramref name="array"/> from <paramref name="arrayIndex"/>.</summary>
        /// <param name="array">The destination.</param>
        /// <param name="arrayIndex">The first destination slot.</param>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="arrayIndex"/> is negative.</exception>
        /// <exception cref="ArgumentException">The entries do not fit from <paramref name="arrayIndex"/> on.</exception>
        void ICollection<KeyValuePair<string, TValue>>.CopyTo(KeyValuePair<string, TValue>[] array, int arrayIndex)
            => ((ICollection<KeyValuePair<string, TValue>>)_items).CopyTo(array, arrayIndex);

        /// <summary>The refusal every mutator throws, with ReadOnlyDictionary's text.</summary>
        /// <returns>The exception (not thrown).</returns>
        private static NotSupportedException ReadOnly() => new NotSupportedException("Collection is read-only.");
    }
}
