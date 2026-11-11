using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.ExceptionServices;

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>BaseExceptionGroup</c>: several unrelated exceptions raised together (Python 3.11+, PEP 654), with
    ///     the operations Python's <c>except*</c> is built on — <see cref="subgroup(Type[])"/>, <see cref="split(Type[])"/>
    ///     and the <see cref="derive"/> hook. Its <see cref="Exception.Message"/> is Python's <c>str()</c>:
    ///     <c>message (2 sub-exceptions)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     It derives from .NET's <see cref="AggregateException"/>, the exception with the same meaning, so its members are
    ///     <see cref="AggregateException.InnerExceptions"/> (also <see cref="exceptions"/>) and a
    ///     <c>catch (AggregateException)</c> catches it. Footgun inherited from that base: <see cref="Exception.InnerException"/>
    ///     is the FIRST member, not Python's <c>__cause__</c>; a group has no separate cause.
    ///     </para>
    ///     <para>
    ///     Python's <c>BaseExceptionGroup(message, excs)</c> returns an <see cref="ExceptionGroup"/> when every member is
    ///     an <c>Exception</c>; a C# constructor cannot return a subclass, so <c>new BaseExceptionGroup(...)</c> always
    ///     builds this class, and <see cref="Create"/> is Python's call (it picks the class as CPython does).
    ///     <see cref="KeyboardInterrupt"/>, <see cref="SystemExit"/>, <see cref="GeneratorExit"/> and a group that is not an
    ///     <see cref="ExceptionGroup"/> are Python's <c>BaseException</c>-only members; every other .NET exception counts
    ///     as an <c>Exception</c>.
    ///     </para>
    ///     <para>
    ///     NumSharp never raises a group; the classes exist so a port of code that raises, splits or catches groups
    ///     compiles and behaves as in Python.
    ///     </para>
    /// </remarks>
    public class BaseExceptionGroup : AggregateException, INumSharpException
    {
        /// <summary>The message as given, without the "(n sub-exceptions)" suffix <see cref="Message"/> adds.</summary>
        private readonly string _message;

        /// <summary>
        ///     Python's <c>BaseExceptionGroup(message, exceptions)</c>, building this class whatever the members are (see
        ///     <see cref="Create"/> for Python's class choice).
        /// </summary>
        /// <param name="message">The group's message (Python's <c>message</c>).</param>
        /// <param name="exceptions">The members, in order; enumerated once.</param>
        /// <exception cref="TypeError"><paramref name="message"/> is null (CPython: <c>BaseExceptionGroup.__new__()
        ///     argument 1 must be str, not None</c>), or <paramref name="exceptions"/> is null (<c>second argument
        ///     (exceptions) must be a sequence</c>).</exception>
        /// <exception cref="ValueError"><paramref name="exceptions"/> is empty (<c>second argument (exceptions) must be a
        ///     non-empty sequence</c>) or holds a null (<c>Item 1 of second argument (exceptions) is not an
        ///     exception</c>).</exception>
        public BaseExceptionGroup(string message, IEnumerable<Exception> exceptions)
            : base(CheckMessage(message), CheckExceptions(exceptions))
        {
            _message = message;
        }

        /// <summary>
        ///     Python's <c>str()</c> of the group: <c>message (1 sub-exception)</c> / <c>message (n sub-exceptions)</c>,
        ///     replacing <see cref="AggregateException"/>'s text (which appends every member's message).
        /// </summary>
        public override string Message
        {
            get
            {
                int n = InnerExceptions.Count;
                return string.Create(CultureInfo.InvariantCulture, $"{_message} ({n} sub-exception{(n > 1 ? "s" : "")})");
            }
        }

        /// <summary>The group's message as given (Python's <c>message</c>), without the count <see cref="Message"/>
        ///     adds.</summary>
        public string message => _message;

        /// <summary>The members, in order (Python's <c>exceptions</c> tuple): the same collection as
        ///     <see cref="AggregateException.InnerExceptions"/>.</summary>
        public ReadOnlyCollection<Exception> exceptions => InnerExceptions;

        /// <summary>
        ///     Python's <c>BaseExceptionGroup(message, exceptions)</c> call: an <see cref="ExceptionGroup"/> when every member
        ///     is a Python <c>Exception</c>, else a <see cref="BaseExceptionGroup"/>.
        /// </summary>
        /// <param name="message">The group's message.</param>
        /// <param name="exceptions">The members, in order; enumerated once.</param>
        /// <returns>The new group.</returns>
        /// <exception cref="TypeError">A null message or member sequence.</exception>
        /// <exception cref="ValueError">An empty member sequence, or a null member.</exception>
        public static BaseExceptionGroup Create(string message, IEnumerable<Exception> exceptions)
        {
            // CPython's order: the message's type, then the members. They are materialized once, so the class test and the
            // group see the same members (the caller's sequence may be lazy).
            CheckMessage(message);
            Exception[] members = CheckExceptions(exceptions);
            foreach (Exception e in members)
                if (!IsPythonException(e))
                    return new BaseExceptionGroup(message, members);
            return new ExceptionGroup(message, members);
        }

        /// <summary>
        ///     Python's <c>subgroup(condition)</c> with exception types: the members that are instances of any of
        ///     <paramref name="types"/>, nested groups kept as nested (sub)groups, or null when none is. The whole group is
        ///     returned as is when it matches itself.
        /// </summary>
        /// <param name="types">The exception types (Python's type or tuple of types); none matches nothing.</param>
        /// <returns>The matching subgroup, this group, or null.</returns>
        /// <exception cref="TypeError"><paramref name="types"/> is null, or one of them is null or not an exception type
        ///     (CPython: <c>expected a function, exception type or tuple of exception types</c>).</exception>
        public BaseExceptionGroup subgroup(params Type[] types)
            => SplitCore(TypeMatcher(types), constructRest: false).match;

        /// <summary>
        ///     Python's <c>subgroup(condition)</c> with a predicate: the members (and nested groups) for which
        ///     <paramref name="condition"/> returns true, or null when none does. The predicate sees this group first, and
        ///     a group it accepts is kept whole.
        /// </summary>
        /// <param name="condition">The predicate, called on this group, then on each member and nested group.</param>
        /// <returns>The matching subgroup, this group, or null.</returns>
        /// <exception cref="TypeError"><paramref name="condition"/> is null (CPython: <c>expected a function, exception
        ///     type or tuple of exception types</c>).</exception>
        public BaseExceptionGroup subgroup(Func<Exception, bool> condition)
            => SplitCore(PredicateMatcher(condition), constructRest: false).match;

        /// <summary>
        ///     Python's <c>split(condition)</c> with exception types: <c>(match, rest)</c>, the members that are instances of
        ///     any of <paramref name="types"/> and the ones that are not, each a group with this group's structure or null
        ///     when empty.
        /// </summary>
        /// <param name="types">The exception types; none matches nothing.</param>
        /// <returns>The matching and the remaining parts.</returns>
        /// <exception cref="TypeError">A null or non-exception type.</exception>
        /// <remarks>Each part is built by <see cref="derive"/> and takes this group's <see cref="Exception.Data"/> (an
        ///     independent copy) and its stack trace, as CPython copies <c>__notes__</c> and <c>__traceback__</c>.</remarks>
        public (BaseExceptionGroup match, BaseExceptionGroup rest) split(params Type[] types)
            => SplitCore(TypeMatcher(types), constructRest: true);

        /// <summary>
        ///     Python's <c>split(condition)</c> with a predicate: <c>(match, rest)</c> by <paramref name="condition"/>, as
        ///     <see cref="split(Type[])"/> does by type.
        /// </summary>
        /// <param name="condition">The predicate, called on this group, then on each member and nested group.</param>
        /// <returns>The matching and the remaining parts.</returns>
        /// <exception cref="TypeError"><paramref name="condition"/> is null.</exception>
        public (BaseExceptionGroup match, BaseExceptionGroup rest) split(Func<Exception, bool> condition)
            => SplitCore(PredicateMatcher(condition), constructRest: true);

        /// <summary>
        ///     Python's <c>derive(excs)</c>: a new group with this group's message and <paramref name="excs"/> as members.
        ///     <see cref="subgroup(Type[])"/> and <see cref="split(Type[])"/> build every part through it, so a subclass
        ///     overrides it to keep its own class and fields in those parts (Python's documented extension point).
        /// </summary>
        /// <param name="excs">The members of the new group.</param>
        /// <returns>The new group: by default <see cref="Create"/>'s choice, a plain <see cref="ExceptionGroup"/> or
        ///     <see cref="BaseExceptionGroup"/>, NOT this group's subclass (as in Python).</returns>
        /// <exception cref="TypeError">A null member sequence.</exception>
        /// <exception cref="ValueError">An empty member sequence, or a null member.</exception>
        public virtual BaseExceptionGroup derive(IEnumerable<Exception> excs) => Create(_message, excs);

        /// <summary>
        ///     Whether Python counts <paramref name="e"/> as an <c>Exception</c> (not only a <c>BaseException</c>): every
        ///     .NET exception but <see cref="KeyboardInterrupt"/>, <see cref="SystemExit"/>, <see cref="GeneratorExit"/> and
        ///     a group that is not an <see cref="ExceptionGroup"/>.
        /// </summary>
        /// <param name="e">The member.</param>
        /// <returns>True for a Python <c>Exception</c>.</returns>
        internal static bool IsPythonException(Exception e)
            => e is not (KeyboardInterrupt or SystemExit or GeneratorExit) && (e is not BaseExceptionGroup || e is ExceptionGroup);

        /// <summary>CPython's <c>"U"</c> check of the message argument.</summary>
        /// <param name="message">The message.</param>
        /// <returns>The message, unchanged.</returns>
        /// <exception cref="TypeError">The message is null.</exception>
        private static string CheckMessage(string message)
            => message ?? throw new TypeError("BaseExceptionGroup.__new__() argument 1 must be str, not None");

        /// <summary>
        ///     CPython's member checks in <c>BaseExceptionGroup.__new__</c>, in its order: a sequence, non-empty, every item
        ///     an exception.
        /// </summary>
        /// <param name="exceptions">The members.</param>
        /// <returns>The members as an array (the sequence enumerated once).</returns>
        /// <exception cref="TypeError">The sequence is null.</exception>
        /// <exception cref="ValueError">It is empty, or holds a null.</exception>
        private static Exception[] CheckExceptions(IEnumerable<Exception> exceptions)
        {
            if (exceptions is null)
                throw new TypeError("second argument (exceptions) must be a sequence");
            var members = new List<Exception>(exceptions);
            if (members.Count == 0)
                throw new ValueError("second argument (exceptions) must be a non-empty sequence");
            for (int i = 0; i < members.Count; i++)
                if (members[i] is null)
                    throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"Item {i} of second argument (exceptions) is not an exception"));
            return members.ToArray();
        }

        /// <summary>CPython's matcher for a type or tuple of types: <c>PyErr_GivenExceptionMatches</c>, an
        ///     <c>isinstance</c> test.</summary>
        /// <param name="types">The types.</param>
        /// <returns>The matcher.</returns>
        /// <exception cref="TypeError">A null array, or a null or non-exception type in it.</exception>
        private static Func<Exception, bool> TypeMatcher(Type[] types)
        {
            if (types is null)
                throw new TypeError("expected a function, exception type or tuple of exception types");
            foreach (Type t in types)
                if (t is null || !typeof(Exception).IsAssignableFrom(t))
                    throw new TypeError("expected a function, exception type or tuple of exception types");
            // A copy: the caller's params array may change while the recursion runs.
            Type[] copy = (Type[])types.Clone();
            return e =>
            {
                foreach (Type t in copy)
                    if (t.IsInstanceOfType(e))
                        return true;
                return false;
            };
        }

        /// <summary>CPython's predicate matcher (any callable that is not a class).</summary>
        /// <param name="condition">The predicate.</param>
        /// <returns>The predicate.</returns>
        /// <exception cref="TypeError">The predicate is null.</exception>
        private static Func<Exception, bool> PredicateMatcher(Func<Exception, bool> condition)
            => condition ?? throw new TypeError("expected a function, exception type or tuple of exception types");

        /// <summary>
        ///     CPython's <c>exceptiongroup_split_recursive</c> from this group.
        /// </summary>
        /// <param name="matcher">The match test.</param>
        /// <param name="constructRest">Whether to build the non-matching part (<c>split</c>) or not (<c>subgroup</c>).</param>
        /// <returns>The matching part and (when asked) the rest, each null when empty.</returns>
        private (BaseExceptionGroup match, BaseExceptionGroup rest) SplitCore(Func<Exception, bool> matcher, bool constructRest)
        {
            var (match, rest) = SplitRecursive(this, matcher, constructRest);
            return ((BaseExceptionGroup)match, (BaseExceptionGroup)rest);
        }

        /// <summary>
        ///     CPython's <c>exceptiongroup_split_recursive</c>: a node that matches is kept whole; a leaf that does not goes to
        ///     the rest; a group that does not is split member by member and rebuilt through <see cref="derive"/> from the
        ///     non-empty parts.
        /// </summary>
        /// <param name="exc">The node: a member or a group.</param>
        /// <param name="matcher">The match test.</param>
        /// <param name="constructRest">Whether to build the non-matching part.</param>
        /// <returns>The node's matching part and rest (null when empty).</returns>
        /// <exception cref="TypeError">A <see cref="derive"/> override returned null.</exception>
        private static (Exception match, Exception rest) SplitRecursive(Exception exc, Func<Exception, bool> matcher, bool constructRest)
        {
            // Full match: the node itself (CPython checks the node before its members, so a matching group stays whole).
            if (matcher(exc))
                return (exc, null);
            if (exc is not BaseExceptionGroup eg)
                return (null, constructRest ? exc : null);

            // Partial match: recurse into the members, keeping their order in both parts.
            var matchList = new List<Exception>();
            var restList = constructRest ? new List<Exception>() : null;
            foreach (Exception member in eg.InnerExceptions)
            {
                var (m, r) = SplitRecursive(member, matcher, constructRest);
                if (m != null)
                    matchList.Add(m);
                if (r != null)
                    restList.Add(r);
            }
            return (Subset(eg, matchList), constructRest ? Subset(eg, restList) : null);
        }

        /// <summary>
        ///     CPython's <c>exceptiongroup_subset</c>: <paramref name="orig"/>'s <see cref="derive"/> of
        ///     <paramref name="excs"/>, with <paramref name="orig"/>'s metadata, or null for no members.
        /// </summary>
        /// <param name="orig">The group being split.</param>
        /// <param name="excs">The part's members.</param>
        /// <returns>The part, or null.</returns>
        /// <exception cref="TypeError"><see cref="derive"/> returned null (CPython: <c>derive must return an instance of
        ///     BaseExceptionGroup</c>).</exception>
        private static BaseExceptionGroup Subset(BaseExceptionGroup orig, List<Exception> excs)
        {
            if (excs.Count == 0)
                return null;
            BaseExceptionGroup eg = orig.derive(excs)
                                    ?? throw new TypeError("derive must return an instance of BaseExceptionGroup");
            // CPython copies __traceback__, __context__, __cause__ and __notes__. .NET's analogues: the stack trace (only
            // onto a group never thrown, which is what derive builds) and Data (an independent copy, as CPython copies the
            // notes list). The cause has no separate slot here: an AggregateException's InnerException is its first member.
            if (orig.StackTrace is string trace && eg.StackTrace is null && !ReferenceEquals(eg, orig))
                ExceptionDispatchInfo.SetRemoteStackTrace(eg, trace);
            foreach (DictionaryEntry entry in orig.Data)
                if (!eg.Data.Contains(entry.Key))
                    eg.Data[entry.Key] = entry.Value;
            return eg;
        }
    }

    /// <summary>
    ///     Python's <c>ExceptionGroup</c>: a <see cref="BaseExceptionGroup"/> whose members are all Python
    ///     <c>Exception</c>s, the group ordinary code raises and catches (<c>except*</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python's MRO is <c>ExceptionGroup → BaseExceptionGroup, Exception</c>; NumSharp's derives from
    ///     <see cref="BaseExceptionGroup"/> (and so from .NET's <see cref="AggregateException"/>).
    ///     </para>
    ///     <para>
    ///     It refuses a <see cref="KeyboardInterrupt"/>, <see cref="SystemExit"/>, <see cref="GeneratorExit"/> or a
    ///     non-<see cref="ExceptionGroup"/> group as a member, as Python's does (<c>Cannot nest BaseExceptions in an
    ///     ExceptionGroup</c>); build a <see cref="BaseExceptionGroup"/> for those, or let
    ///     <see cref="BaseExceptionGroup.Create"/> choose.
    ///     </para>
    /// </remarks>
    public class ExceptionGroup : BaseExceptionGroup
    {
        /// <summary>Python's <c>ExceptionGroup(message, exceptions)</c>.</summary>
        /// <param name="message">The group's message (Python's <c>message</c>).</param>
        /// <param name="exceptions">The members, in order; enumerated once.</param>
        /// <exception cref="TypeError">A null message or member sequence; a member Python counts as a
        ///     <c>BaseException</c> only (CPython: <c>Cannot nest BaseExceptions in an ExceptionGroup</c>, or <c>Cannot nest
        ///     BaseExceptions in 'MyGroup'</c> for a subclass).</exception>
        /// <exception cref="ValueError">An empty member sequence, or a null member.</exception>
        public ExceptionGroup(string message, IEnumerable<Exception> exceptions) : base(message, exceptions)
        {
            // The base constructor validated the members; CPython's class check runs after those checks, in __new__.
            foreach (Exception e in InnerExceptions)
                if (!IsPythonException(e))
                    throw new TypeError(GetType() == typeof(ExceptionGroup)
                        ? "Cannot nest BaseExceptions in an ExceptionGroup"
                        : $"Cannot nest BaseExceptions in '{GetType().Name}'");
        }
    }
}
