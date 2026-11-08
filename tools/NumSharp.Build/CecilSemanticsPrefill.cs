using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using Mono.Cecil;
using CecilMethodSemantics = Mono.Cecil.MethodSemanticsAttributes;
using SrmMetadataReader = System.Reflection.Metadata.MetadataReader;
using SrmMethodHandle = System.Reflection.Metadata.MethodDefinitionHandle;

namespace NumSharp.Build;

/// <summary>
///     Fills in, in linear time, the property and event accessors Mono.Cecil would otherwise read in QUADRATIC time for
///     a type with many properties or events - before anything (the weave, the writer) asks for them.
/// </summary>
/// <remarks>
///     <para><b>The quadratic.</b> Cecil 0.11.6 reads a type's method semantics (which method is which property's getter,
///     which event's adder) all at once, the first time any of them is asked for - and the writer asks for every type.
///     For each accessor method it finds the owning property or event by a LINEAR scan of the type's property (event)
///     list (<c>MetadataReader.GetMember</c>), so a type with P properties costs about P² comparisons: a synthetic type
///     of 50,000 auto-properties took 9-18 s to weave, all of it in that scan, and a designer class with a few thousand
///     resource properties pays tenths of a second.</para>
///     <para><b>The prefill.</b> The same answers, read through System.Reflection.Metadata's accessor API (a binary search
///     of the MethodSemantics table per property or event) and matched to Cecil's objects by row number, are stored with
///     Cecil's own public setters (<c>PropertyDefinition.GetMethod</c> / <c>SetMethod</c>, the event's add / remove /
///     raise methods) and two internal fields of <c>MethodDefinition</c> (<c>sem_attrs</c>, <c>sem_attrs_ready</c>) that
///     Cecil's reader sets itself: once they are set, Cecil's lazy read never runs for the type. The result is the state
///     Cecil's reader would have produced, so every later read and the written assembly are unchanged (a test weaves
///     with and without the prefill and compares the bytes).</para>
///     <para><b>Fallbacks, each to Cecil's own path for the type:</b> a type with fewer than <see cref="Threshold"/>
///     properties and events (Cecil's scan is cheap there, so the common type keeps the exact code it always ran); a
///     Cecil build without the two internal fields (a version change - the weave is then only slower); a type whose
///     semantics Cecil already read; an accessor of the <c>.other</c> kind (C# never emits one); an accessor that
///     belongs to two members or to another type (invalid for C#, but legal metadata Cecil resolves in its own way).</para>
/// </remarks>
internal static class CecilSemanticsPrefill
{
    /// <summary>Properties plus events from which a type is prefilled: below it Cecil's own scan costs at most a few thousand comparisons.</summary>
    internal const int Threshold = 64;

    /// <summary><c>MethodDefinition.sem_attrs</c> (the method's semantics), or null when this Cecil has no such field.</summary>
    private static readonly FieldInfo SemanticsField = Field("sem_attrs", typeof(CecilMethodSemantics));

    /// <summary><c>MethodDefinition.sem_attrs_ready</c> (the semantics were read), or null when this Cecil has no such field.</summary>
    private static readonly FieldInfo SemanticsReadyField = Field("sem_attrs_ready", typeof(bool));

    /// <summary>
    ///     Prefills every type of <paramref name="module"/> that has at least <paramref name="threshold"/> properties and
    ///     events. Call it right after reading the module, before any accessor or <c>SemanticsAttributes</c> is read.
    /// </summary>
    /// <param name="module">The module Cecil read from <paramref name="assemblyPath"/>.</param>
    /// <param name="assemblyPath">The file the module was read from (read again here, through System.Reflection.Metadata).</param>
    /// <param name="threshold">The minimum properties + events of a prefilled type (<see cref="Threshold"/> in the tool; 0 in a test that prefills every type).</param>
    /// <returns>The number of types prefilled.</returns>
    /// <exception cref="IOException">The assembly file cannot be read.</exception>
    /// <exception cref="BadImageFormatException">The file is not a valid PE image with metadata.</exception>
    internal static int Prefill(ModuleDefinition module, string assemblyPath, int threshold = Threshold)
    {
        if (SemanticsField == null || SemanticsReadyField == null)
            return 0; // a Cecil without the fields: its own (slower) path stays correct

        var large = new List<TypeDefinition>();
        foreach (var type in AllTypes(module))
        {
            int members = (type.HasProperties ? type.Properties.Count : 0) + (type.HasEvents ? type.Events.Count : 0);
            if (members > 0 && members >= threshold)
                large.Add(type);
        }

        if (large.Count == 0)
            return 0; // the common assembly: no type is large, and the file is not read again

        // The module was read InMemory (the file handle is released), so the file can be opened again; the bytes are
        // the ones Cecil read.
        using var stream = new FileStream(assemblyPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var pe = new PEReader(stream);
        var metadata = System.Reflection.Metadata.PEReaderExtensions.GetMetadataReader(pe);

        int prefilled = 0;
        foreach (var type in large)
            if (TryPrefill(type, metadata))
                prefilled++;
        return prefilled;
    }

    /// <summary>Every type of the module, nested types included (Cecil's <c>Types</c> lists top-level types only).</summary>
    /// <param name="module">The module.</param>
    /// <returns>The types, outer before nested.</returns>
    private static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
    {
        var stack = new Stack<TypeDefinition>(module.Types);
        while (stack.Count > 0)
        {
            var type = stack.Pop();
            yield return type;
            if (type.HasNestedTypes)
                foreach (var nested in type.NestedTypes)
                    stack.Push(nested);
        }
    }

    /// <summary>
    ///     Computes one type's accessors from the metadata and, when every check passes, stores them in Cecil's objects.
    ///     Nothing is stored for a type that falls back: the checks all run before the first write.
    /// </summary>
    /// <param name="type">A type with many properties or events.</param>
    /// <param name="metadata">The assembly's metadata, read by System.Reflection.Metadata.</param>
    /// <returns>True when the type was prefilled; false when it keeps Cecil's own path.</returns>
    private static bool TryPrefill(TypeDefinition type, SrmMetadataReader metadata)
    {
        var methods = new Dictionary<int, MethodDefinition>(type.Methods.Count);
        foreach (var method in type.Methods)
        {
            // Already read (by anything before us): Cecil's state is final for the type; leave it alone.
            if ((bool)SemanticsReadyField.GetValue(method))
                return false;
            methods[(int)method.MetadataToken.RID] = method;
        }

        var semantics = new Dictionary<MethodDefinition, CecilMethodSemantics>();
        var getters = new List<(PropertyDefinition Property, MethodDefinition Getter, MethodDefinition Setter)>();
        var eventAccessors = new List<(EventDefinition Event, MethodDefinition Adder, MethodDefinition Remover, MethodDefinition Raiser)>();
        var typeHandle = MetadataTokens.TypeDefinitionHandle((int)type.MetadataToken.RID);
        var typeDefinition = metadata.GetTypeDefinition(typeHandle);

        if (type.HasProperties)
        {
            var properties = new Dictionary<int, PropertyDefinition>(type.Properties.Count);
            foreach (var property in type.Properties)
                properties[(int)property.MetadataToken.RID] = property;

            foreach (var handle in typeDefinition.GetProperties())
            {
                var accessors = metadata.GetPropertyDefinition(handle).GetAccessors();
                if (!accessors.Others.IsEmpty)
                    return false; // .other accessors: Cecil's own path (it keeps them in a collection this does not fill)
                if (!properties.TryGetValue(MetadataTokens.GetRowNumber(handle), out var property))
                    return false;
                var getter = Resolve(accessors.Getter, methods);
                var setter = Resolve(accessors.Setter, methods);
                if ((!accessors.Getter.IsNil && getter == null) || (!accessors.Setter.IsNil && setter == null))
                    return false; // an accessor of another type
                if (!Claim(semantics, getter, CecilMethodSemantics.Getter) || !Claim(semantics, setter, CecilMethodSemantics.Setter))
                    return false; // one method, two roles
                getters.Add((property, getter, setter));
            }
        }

        if (type.HasEvents)
        {
            var events = new Dictionary<int, EventDefinition>(type.Events.Count);
            foreach (var @event in type.Events)
                events[(int)@event.MetadataToken.RID] = @event;

            foreach (var handle in typeDefinition.GetEvents())
            {
                var accessors = metadata.GetEventDefinition(handle).GetAccessors();
                if (!accessors.Others.IsEmpty)
                    return false;
                if (!events.TryGetValue(MetadataTokens.GetRowNumber(handle), out var @event))
                    return false;
                var adder = Resolve(accessors.Adder, methods);
                var remover = Resolve(accessors.Remover, methods);
                var raiser = Resolve(accessors.Raiser, methods);
                if ((!accessors.Adder.IsNil && adder == null) || (!accessors.Remover.IsNil && remover == null) ||
                    (!accessors.Raiser.IsNil && raiser == null))
                    return false;
                if (!Claim(semantics, adder, CecilMethodSemantics.AddOn) || !Claim(semantics, remover, CecilMethodSemantics.RemoveOn) ||
                    !Claim(semantics, raiser, CecilMethodSemantics.Fire))
                    return false;
                eventAccessors.Add((@event, adder, remover, raiser));
            }
        }

        // Every check passed: store. The accessors first (Cecil's InitializeMethods returns early once one is set),
        // then every method's semantics with its ready flag, the pair Cecil's ReadAllSemantics writes - a method that
        // is no accessor gets None, as Cecil gives it.
        foreach (var (property, getter, setter) in getters)
        {
            if (getter != null)
                property.GetMethod = getter;
            if (setter != null)
                property.SetMethod = setter;
        }

        foreach (var (@event, adder, remover, raiser) in eventAccessors)
        {
            if (adder != null)
                @event.AddMethod = adder;
            if (remover != null)
                @event.RemoveMethod = remover;
            if (raiser != null)
                @event.InvokeMethod = raiser;
        }

        foreach (var method in type.Methods)
        {
            SemanticsField.SetValue(method, semantics.TryGetValue(method, out var s) ? s : CecilMethodSemantics.None);
            SemanticsReadyField.SetValue(method, true);
        }

        return true;
    }

    /// <summary>The Cecil method of an accessor handle, or null for a nil handle or a method of another type.</summary>
    /// <param name="handle">The accessor's method handle.</param>
    /// <param name="methods">The type's methods by row number.</param>
    /// <returns>The method, or null.</returns>
    private static MethodDefinition Resolve(SrmMethodHandle handle, Dictionary<int, MethodDefinition> methods)
        => !handle.IsNil && methods.TryGetValue(MetadataTokens.GetRowNumber(handle), out var method) ? method : null;

    /// <summary>Records a method's role; false when the method already has one (an accessor of two members).</summary>
    /// <param name="semantics">The roles recorded so far.</param>
    /// <param name="method">The accessor, or null (nothing to record).</param>
    /// <param name="role">Its role.</param>
    /// <returns>False on a second role.</returns>
    private static bool Claim(Dictionary<MethodDefinition, CecilMethodSemantics> semantics, MethodDefinition method, CecilMethodSemantics role)
    {
        if (method == null)
            return true;
        if (semantics.ContainsKey(method))
            return false;
        semantics.Add(method, role);
        return true;
    }

    /// <summary>One of <c>MethodDefinition</c>'s internal fields, or null when it is missing or of another type.</summary>
    /// <param name="name">The field name.</param>
    /// <param name="fieldType">Its expected type.</param>
    /// <returns>The field, or null.</returns>
    private static FieldInfo Field(string name, Type fieldType)
    {
        var field = typeof(MethodDefinition).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        return field != null && field.FieldType == fieldType ? field : null;
    }
}
