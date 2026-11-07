#:package Mono.Cecil@0.11.6
#:property PublishAot=false
#:property Nullable=enable
#:property ImplicitUsings=enable

// =============================================================================
// SurfaceMap.cs - NumSharp.dll's internal surface, and what it closes to a consumer who extends or inherits
// =============================================================================
//
// Reads the BUILT assemblies with Mono.Cecil (metadata, portable PDB, XML doc file) instead of grepping the source, so
// every accessibility answer is the compiler's own: the effective reach of a nested member (a public member of an
// internal type is internal), partial types spread over many files, property accessors with their own access, and
// compiler-generated code (lambdas, iterators, backing fields, file-local types) told apart from declared code.
//
// Writes, into --out:
//   internal-types.tsv            one row per type no consumer can name (declared internal, or nested in one)
//   internal-members.tsv          one row per internal member: declared internal / private protected on a visible type,
//                                 or a non-private member of an internal type
//   extensibility-findings.tsv    what closes a visible type to a consumer's subclass or implementation
//   virtuality.tsv                per visible non-sealed class: how much of its visible instance surface can be overridden
//   friend-internals-usage.tsv    the internals NumSharp's own InternalsVisibleTo packages reach - what a third-party
//                                 package of the same kind cannot reach
//   exact-type-checks.tsv         GetType() equality tests in Core's source (an exact test treats a subclass as foreign)
//   tables.md                     the summary tables of all of the above, with source links
//
// Usage, from the repository root after building NumSharp.Core and the src/ packages for the chosen framework:
//   dotnet run tools/internal-surface-map/SurfaceMap.cs -- --repo . --out docs/plans/internal-surface --label "<commit>"
// =============================================================================

using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

return SurfaceMap.Program.Run(args);

namespace SurfaceMap
{
    /// <summary>
    ///     Who outside NumSharp.dll can name a type or member. Ordered from the widest to the narrowest, so the reach of a
    ///     nested declaration is the <see cref="Math.Max(int, int)"/> of its own and its container's.
    /// </summary>
    internal enum Reach
    {
        /// <summary>Any consumer: the member and every type around it are public.</summary>
        Public,

        /// <summary>Only a consumer's subclass: <c>protected</c> or <c>protected internal</c> somewhere in the chain.</summary>
        Protected,

        /// <summary>Only NumSharp.dll and its InternalsVisibleTo friends: <c>internal</c> / <c>private protected</c> in the chain.</summary>
        Internal,

        /// <summary>Only the declaring type: <c>private</c> somewhere in the chain.</summary>
        Private,
    }

    /// <summary>The command line of the map.</summary>
    internal sealed class Options
    {
        /// <summary>The repository root: <c>src/NumSharp.Core</c> and the packages' <c>src/&lt;name&gt;</c> folders live under it.</summary>
        public string Repo = "";

        /// <summary>The output folder; created when missing, its files overwritten.</summary>
        public string Out = "";

        /// <summary>The build configuration whose output is read (<c>Debug</c> by default: it always has a portable PDB).</summary>
        public string Configuration = "Debug";

        /// <summary>The target framework whose output is read.</summary>
        public string Tfm = "net10.0";

        /// <summary>A label for the header of <c>tables.md</c> (the branch and commit the build came from).</summary>
        public string Label = "";

        /// <summary>The built NumSharp.dll (the assembly name of NumSharp.Core).</summary>
        public string CoreDll => Path.Combine(Repo, "src", "NumSharp.Core", "bin", Configuration, Tfm, "NumSharp.dll");

        /// <summary>
        ///     Parses <c>--repo</c>, <c>--out</c>, <c>--config</c>, <c>--tfm</c> and <c>--label</c>.
        /// </summary>
        /// <param name="args">The arguments after <c>--</c>.</param>
        /// <returns>The options, with <see cref="Repo"/> and <see cref="Out"/> made absolute.</returns>
        /// <exception cref="ArgumentException">An option is unknown or lacks its value, <c>--repo</c> / <c>--out</c> is
        ///     missing, or NumSharp.dll is not built at <see cref="CoreDll"/>.</exception>
        public static Options Parse(string[] args)
        {
            var o = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                // Every option takes one value; a trailing option without one is a usage error, not a default.
                if (i + 1 >= args.Length)
                    throw new ArgumentException($"Option {args[i]} needs a value.");
                string value = args[++i];
                switch (args[i - 1])
                {
                    case "--repo": o.Repo = Path.GetFullPath(value); break;
                    case "--out": o.Out = Path.GetFullPath(value); break;
                    case "--config": o.Configuration = value; break;
                    case "--tfm": o.Tfm = value; break;
                    case "--label": o.Label = value; break;
                    default: throw new ArgumentException($"Unknown option {args[i - 1]}.");
                }
            }

            if (o.Repo.Length == 0 || o.Out.Length == 0)
                throw new ArgumentException("Usage: SurfaceMap.cs -- --repo <root> --out <dir> [--config Debug] [--tfm net10.0] [--label text]");
            if (!File.Exists(o.CoreDll))
                throw new ArgumentException($"NumSharp.dll is not built at {o.CoreDll} (build src/NumSharp.Core for {o.Tfm} / {o.Configuration} first).");
            return o;
        }
    }

    /// <summary>
    ///     One type of NumSharp.dll as the map sees it: its C# spelling, its reach, where it is declared and its members.
    /// </summary>
    internal sealed class TypeInfo
    {
        /// <summary>The Cecil definition.</summary>
        public required TypeDefinition Def;

        /// <summary>The namespace of the outermost declaring type (a nested type has none of its own in metadata).</summary>
        public required string Namespace;

        /// <summary>The C# spelling: <c>Outer&lt;T&gt;.Inner</c>.</summary>
        public required string Display;

        /// <summary>class / static class / abstract class / sealed class / record ... / struct / interface / enum / delegate.</summary>
        public required string Kind;

        /// <summary>The declared C# access of the type itself.</summary>
        public required string Access;

        /// <summary>The effective reach (the narrowest along the declaring-type chain).</summary>
        public required Reach Reach;

        /// <summary>The repository-relative file that holds most of the type's located members (its declaration for a
        ///     memberless type); empty when nothing locates it.</summary>
        public string File = "";

        /// <summary>The declaration line in <see cref="File"/>, or the first located member's line; 0 when unknown.</summary>
        public int Line;

        /// <summary>How many distinct source files the type's located members come from (a partial type spans several).</summary>
        public int FileCount;

        /// <summary>
        ///     Every file the type is declared or located in (each part of a partial type), searched for the members the PDB
        ///     cannot place. <see cref="File"/> first.
        /// </summary>
        public List<string> Files = new();

        /// <summary>The <see cref="File"/>'s folder under <c>src/NumSharp.Core</c>, two levels deep (<c>Backends/Kernels</c>).</summary>
        public string Area = "";

        /// <summary>The first sentence of the type's XML <c>summary</c>; empty when undocumented.</summary>
        public string Summary = "";

        /// <summary>The non-private, non-compiler-generated members.</summary>
        public List<MemberInfo> Members = new();

        /// <summary>The number of private members (counted, never listed: they are the type's own business).</summary>
        public int PrivateMembers;

        /// <summary>For an enum, its value count (values are not listed as members).</summary>
        public int EnumValues;

        /// <summary>The friend packages whose IL names the type (only recorded for an internal type).</summary>
        public SortedSet<string> Friends = new(StringComparer.Ordinal);
    }

    /// <summary>One non-private member of a type: its C# spelling, access, reach, modifiers and location.</summary>
    internal sealed class MemberInfo
    {
        /// <summary>The declaring type.</summary>
        public required TypeInfo Owner;

        /// <summary>The Cecil definition (method, property, field or event).</summary>
        public required IMemberDefinition Def;

        /// <summary>ctor / method / operator / conversion / property / indexer / field / const / event.</summary>
        public required string Kind;

        /// <summary>The member's C# name (<c>operator +</c> for an operator, the type's name for a constructor).</summary>
        public required string Name;

        /// <summary>The C# signature without modifiers.</summary>
        public required string Signature;

        /// <summary>The declared C# access (for a property / event: its most accessible accessor's).</summary>
        public required string Access;

        /// <summary>The effective reach (the member's access combined with the declaring type's reach).</summary>
        public required Reach Reach;

        /// <summary>static / abstract / virtual / override / sealed override / extern / readonly / const, space separated.</summary>
        public required string Modifiers;

        /// <summary>True for an instance member a subclass can override (C# <c>virtual</c> / <c>abstract</c> / unsealed
        ///     <c>override</c>); an implicit interface implementation's <c>virtual final</c> slot is not one.</summary>
        public bool Overridable;

        /// <summary>True when this declaration opens the virtual slot (<c>virtual</c> / <c>abstract</c>, not <c>override</c>).</summary>
        public bool IntroducesSlot;

        /// <summary>True for an <c>abstract</c> member (it must be implemented by every concrete subclass).</summary>
        public bool IsAbstract;

        /// <summary>True for a static member.</summary>
        public bool IsStatic;

        /// <summary>For a property / event whose accessors differ in access: the narrower one, e.g. <c>internal set</c>.</summary>
        public string NarrowAccessor = "";

        /// <summary>The repository-relative source file; empty when neither the PDB nor the source search locates it.</summary>
        public string File = "";

        /// <summary>The 1-based line in <see cref="File"/>; 0 when unknown.</summary>
        public int Line;

        /// <summary>The first sentence of the member's XML <c>summary</c>; empty when undocumented.</summary>
        public string Summary = "";

        /// <summary>The friend packages whose IL names the member.</summary>
        public SortedSet<string> Friends = new(StringComparer.Ordinal);
    }

    /// <summary>One way a visible type is closed to a consumer, as a row of <c>extensibility-findings.tsv</c>.</summary>
    /// <param name="Category">The finding's kind (see <see cref="Extensibility"/> for the list).</param>
    /// <param name="Type">The visible type.</param>
    /// <param name="Member">The member concerned, or <c>null</c> for a type-level finding.</param>
    /// <param name="Detail">What closes it, in words.</param>
    internal sealed record Finding(string Category, TypeInfo Type, MemberInfo? Member, string Detail);

    /// <summary>An internal type or member a friend package's IL names, with how often and from where.</summary>
    internal sealed class FriendUse
    {
        /// <summary>The friend package (assembly name).</summary>
        public required string Friend;

        /// <summary>The internal type named (for a member use: the member's declaring type).</summary>
        public required TypeInfo Type;

        /// <summary>The internal member named, or <c>null</c> for a use of the type itself (a cast, a signature, a base type).</summary>
        public MemberInfo? Member;

        /// <summary>How the friend names it: call / field / type / derives / implements / overrides.</summary>
        public required string How;

        /// <summary>The number of distinct sites (IL instructions, signatures, declarations) that name it.</summary>
        public int Sites;

        /// <summary>The friend's methods / types that name it (<c>Type::Method</c>).</summary>
        public SortedSet<string> Users = new(StringComparer.Ordinal);
    }

    /// <summary>A <c>GetType()</c> equality test found in Core's source.</summary>
    /// <param name="File">The repository-relative file.</param>
    /// <param name="Line">The 1-based line.</param>
    /// <param name="Compared">The type the test compares with (<c>?</c> when both sides are objects).</param>
    /// <param name="Text">The trimmed source line.</param>
    internal sealed record ExactTypeCheck(string File, int Line, string Compared, string Text);

    /// <summary>
    ///     A Cecil resolver that hands out the one already-read NumSharp.dll, so a friend's member reference resolves to the
    ///     very <see cref="MethodDefinition"/> the model indexed (identity lookups instead of signature matching).
    /// </summary>
    internal sealed class SharedResolver : DefaultAssemblyResolver
    {
        /// <summary>Makes <paramref name="assembly"/> the answer for its own name from now on.</summary>
        /// <param name="assembly">The assembly read by the model.</param>
        public void Share(AssemblyDefinition assembly) => RegisterAssembly(assembly);
    }

    /// <summary>Entry point: options, model, friend scan, source scan, outputs.</summary>
    internal static class Program
    {
        /// <summary>Runs the whole map.</summary>
        /// <param name="args">The command line (see <see cref="Options.Parse"/>).</param>
        /// <returns>0 on success, 1 on a usage error (the message is printed).</returns>
        public static int Run(string[] args)
        {
            Options options;
            try
            {
                options = Options.Parse(args);
            }
            catch (ArgumentException e)
            {
                Console.Error.WriteLine(e.Message);
                return 1;
            }

            var model = Model.Load(options);
            var findings = Extensibility.Analyze(model);
            var friends = FriendScan.Run(model, options);
            var checks = SourceScan.ExactTypeChecks(options);
            Directory.CreateDirectory(options.Out);
            Output.Write(model, findings, friends, checks, options);
            Console.WriteLine($"types {model.Types.Count}, internal types {model.Types.Count(t => t.Reach == Reach.Internal)}, " +
                              $"internal members {model.InternalMembers().Count()}, findings {findings.Count}, " +
                              $"friend uses {friends.Uses.Count}, exact type checks {checks.Count} -> {options.Out}");
            return 0;
        }
    }

    /// <summary>
    ///     NumSharp.dll read into <see cref="TypeInfo"/> / <see cref="MemberInfo"/> rows, with the indexes the friend scan
    ///     and the outputs need.
    /// </summary>
    internal sealed class Model
    {
        /// <summary>The read assembly (kept so friend references resolve into it).</summary>
        public required AssemblyDefinition Assembly;

        /// <summary>The resolver shared with the friend scan.</summary>
        public required SharedResolver Resolver;

        /// <summary>Every declared (not compiler-generated) type, nested ones included.</summary>
        public List<TypeInfo> Types = new();

        /// <summary>Type definition -> its row.</summary>
        public Dictionary<TypeDefinition, TypeInfo> ByType = new();

        /// <summary>Method definition -> the member row it belongs to (an accessor maps to its property / event).</summary>
        public Dictionary<MethodDefinition, MemberInfo> ByMethod = new();

        /// <summary>Field definition -> its member row.</summary>
        public Dictionary<FieldDefinition, MemberInfo> ByField = new();

        /// <summary>The friend assemblies NumSharp.dll names in <c>InternalsVisibleTo</c>, in declaration order.</summary>
        public List<string> FriendNames = new();

        /// <summary>The repository root (to make PDB document paths relative).</summary>
        public required string Repo;

        /// <summary>Every internal member: declared internal on a visible type, or non-private on an internal type.</summary>
        /// <returns>The members whose <see cref="MemberInfo.Reach"/> is <see cref="Reach.Internal"/>.</returns>
        public IEnumerable<MemberInfo> InternalMembers() => Types.SelectMany(t => t.Members).Where(m => m.Reach == Reach.Internal);

        /// <summary>
        ///     Reads NumSharp.dll with its PDB and XML doc file and builds every row.
        /// </summary>
        /// <param name="options">Where the build output and the source live.</param>
        /// <returns>The model.</returns>
        /// <exception cref="BadImageFormatException">NumSharp.dll is not a .NET assembly.</exception>
        public static Model Load(Options options)
        {
            var resolver = new SharedResolver();
            resolver.AddSearchDirectory(Path.GetDirectoryName(options.CoreDll)!);
            // System types (base classes of exceptions / attributes) resolve from the running shared framework; the
            // reference assemblies the build compiled against only forward to it.
            resolver.AddSearchDirectory(RuntimeEnvironment.GetRuntimeDirectory());
            bool symbols = File.Exists(Path.ChangeExtension(options.CoreDll, ".pdb"));
            var assembly = AssemblyDefinition.ReadAssembly(options.CoreDll, new ReaderParameters
            {
                AssemblyResolver = resolver,
                ReadSymbols = symbols,
                // InMemory: the DLL can be rebuilt while the map runs (no file lock held).
                InMemory = true,
            });
            resolver.Share(assembly);

            var model = new Model { Assembly = assembly, Resolver = resolver, Repo = options.Repo };
            foreach (var attr in assembly.CustomAttributes.Where(a => a.AttributeType.Name == "InternalsVisibleToAttribute"))
                model.FriendNames.Add(((string)attr.ConstructorArguments[0].Value).Split(',')[0].Trim());

            var summaries = Docs.Load(Path.ChangeExtension(options.CoreDll, ".xml"));
            var index = SourceIndex.Build(Path.Combine(options.Repo, "src", "NumSharp.Core"), options.Repo);

            foreach (var td in assembly.MainModule.GetTypes())
            {
                if (td.Name == "<Module>" || Names.IsGenerated(td))
                    continue;
                var ti = new TypeInfo
                {
                    Def = td,
                    Namespace = Names.NamespaceOf(td),
                    Display = Names.TypeDisplay(td),
                    Kind = Names.TypeKind(td),
                    Access = Access.Of(td),
                    Reach = Access.ReachOf(td),
                };
                ti.Summary = summaries.Get(td);
                model.Types.Add(ti);
                model.ByType[td] = ti;
                model.AddMembers(ti, summaries);
                Locator.LocateType(ti, index, options.Repo, td.IsNested ? model.ByType.GetValueOrDefault(td.DeclaringType) : null);
            }

            // Signatures first: the text search for body-less members starts from the type lines, which stay as placed.
            Locator.RefineToSignatures(model, index);
            foreach (var ti in model.Types)
                Locator.LocateMembersWithoutBodies(ti, index);
            return model;
        }

        /// <summary>
        ///     Adds the type's methods, properties, events and fields as rows (private and compiler-generated ones counted or
        ///     skipped), and indexes them for the friend scan.
        /// </summary>
        /// <param name="ti">The type.</param>
        /// <param name="summaries">The XML doc summaries.</param>
        private void AddMembers(TypeInfo ti, Docs summaries)
        {
            var td = ti.Def;
            // A delegate's members are the runtime's Invoke / BeginInvoke / EndInvoke, not declared API.
            if (ti.Kind == "delegate")
                return;
            if (td.IsEnum)
            {
                ti.EnumValues = td.Fields.Count(f => f.IsStatic && f.IsLiteral);
                return;
            }

            foreach (var m in td.Methods)
            {
                // Accessors are reported through their property / event; a static constructor is not callable API; a record's
                // synthesized Equals / GetHashCode / ToString / Deconstruct / == carry [CompilerGenerated] and have no source.
                if (m.IsGetter || m.IsSetter || m.IsAddOn || m.IsRemoveOn || m.IsFire || m.IsOther || (m.IsConstructor && m.IsStatic)
                    || Names.HasAttribute(m.CustomAttributes, "System.Runtime.CompilerServices.CompilerGeneratedAttribute"))
                    continue;
                if (Names.IsGenerated(m) || m.IsPrivate || m.IsCompilerControlled)
                {
                    if (m.IsPrivate && !Names.IsGenerated(m))
                        ti.PrivateMembers++;
                    continue;
                }

                var access = Access.Of(m);
                var mi = new MemberInfo
                {
                    Owner = ti,
                    Def = m,
                    Kind = m.IsConstructor ? "ctor" : Names.IsConversion(m) ? "conversion" : m.IsSpecialName && m.Name.StartsWith("op_", StringComparison.Ordinal) ? "operator" : "method",
                    Name = Names.MethodName(m, ti),
                    Signature = Names.MethodSignature(m, ti),
                    Access = access,
                    Reach = Access.Combine(ti.Reach, access),
                    Modifiers = Names.Modifiers(m, td),
                    Overridable = Access.IsOverridable(m, td),
                    IntroducesSlot = Access.IsOverridable(m, td) && m.IsNewSlot,
                    IsAbstract = m.IsAbstract && !td.IsInterface,
                    IsStatic = m.IsStatic,
                };
                (mi.File, mi.Line) = Locator.Locate(m, Repo);
                mi.Summary = summaries.Get(m);
                ti.Members.Add(mi);
                ByMethod[m] = mi;
            }

            foreach (var p in td.Properties)
            {
                var accessors = new[] { p.GetMethod, p.SetMethod }.Where(a => a is not null).Cast<MethodDefinition>().ToArray();
                if (Names.IsGenerated(p) || accessors.Length == 0)
                    continue;
                // A property is as visible as its most visible accessor; a narrower second accessor is its own fact
                // ("public get; internal set;" - the setter is hidden from every consumer and subclass).
                var visible = accessors.Where(a => !a.IsPrivate).OrderBy(a => (int)Access.ReachOfAccess(Access.Of(a))).ToArray();
                if (visible.Length == 0)
                {
                    ti.PrivateMembers++;
                    continue;
                }

                var main = visible[0];
                var access = Access.Of(main);
                var mi = new MemberInfo
                {
                    Owner = ti,
                    Def = p,
                    Kind = p.HasParameters ? "indexer" : "property",
                    Name = p.HasParameters ? "this[]" : p.Name,
                    Signature = Names.PropertySignature(p, access),
                    Access = access,
                    Reach = Access.Combine(ti.Reach, access),
                    Modifiers = Names.Modifiers(main, td),
                    Overridable = Access.IsOverridable(main, td),
                    IntroducesSlot = Access.IsOverridable(main, td) && main.IsNewSlot,
                    IsAbstract = main.IsAbstract && !td.IsInterface,
                    IsStatic = main.IsStatic,
                };
                foreach (var other in accessors.Where(x => x != main && Access.Of(x) != access))
                    mi.NarrowAccessor = $"{Access.Of(other)} {(other == p.GetMethod ? "get" : "set")}";
                // The first accessor with a body places the property (an abstract one has none: the text search does it later).
                (mi.File, mi.Line) = accessors.Select(a => Locator.Locate(a, Repo)).Where(l => l.Line > 0).DefaultIfEmpty(("", 0)).First();
                mi.Summary = summaries.Get(p);
                ti.Members.Add(mi);
                foreach (var a in accessors)
                    ByMethod[a] = mi;
            }

            foreach (var e in td.Events)
            {
                if (Names.IsGenerated(e) || e.AddMethod is null)
                    continue;
                if (e.AddMethod.IsPrivate)
                {
                    ti.PrivateMembers++;
                    continue;
                }

                var access = Access.Of(e.AddMethod);
                var mi = new MemberInfo
                {
                    Owner = ti,
                    Def = e,
                    Kind = "event",
                    Name = e.Name,
                    Signature = $"event {Names.TypeName(e.EventType)} {e.Name}",
                    Access = access,
                    Reach = Access.Combine(ti.Reach, access),
                    Modifiers = Names.Modifiers(e.AddMethod, td),
                    Overridable = Access.IsOverridable(e.AddMethod, td),
                    IntroducesSlot = Access.IsOverridable(e.AddMethod, td) && e.AddMethod.IsNewSlot,
                    IsAbstract = e.AddMethod.IsAbstract && !td.IsInterface,
                    IsStatic = e.AddMethod.IsStatic,
                };
                (mi.File, mi.Line) = Locator.Locate(e.AddMethod, Repo);
                mi.Summary = summaries.Get(e);
                ti.Members.Add(mi);
                ByMethod[e.AddMethod] = mi;
                if (e.RemoveMethod is not null)
                    ByMethod[e.RemoveMethod] = mi;
            }

            foreach (var f in td.Fields)
            {
                // Backing fields, fixed-buffer helpers and an event's delegate field are compiler-generated or private.
                if (Names.IsGenerated(f) || f.IsPrivate || f.IsCompilerControlled)
                {
                    if (f.IsPrivate && !Names.IsGenerated(f) && !td.Events.Any(e => e.Name == f.Name))
                        ti.PrivateMembers++;
                    continue;
                }

                var access = Access.Of(f);
                var mi = new MemberInfo
                {
                    Owner = ti,
                    Def = f,
                    Kind = f.IsLiteral ? "const" : "field",
                    Name = f.Name,
                    Signature = $"{Names.TypeName(f.FieldType)} {f.Name}",
                    Access = access,
                    Reach = Access.Combine(ti.Reach, access),
                    Modifiers = Names.FieldModifiers(f),
                    IsStatic = f.IsStatic,
                };
                mi.Summary = summaries.Get(f);
                ti.Members.Add(mi);
                ByField[f] = mi;
            }
        }
    }

    /// <summary>C# access spellings and the reach arithmetic.</summary>
    internal static class Access
    {
        /// <summary>The declared C# access of a type.</summary>
        /// <param name="t">The type.</param>
        /// <returns><c>public</c>, <c>internal</c>, <c>protected</c>, <c>protected internal</c>, <c>private protected</c> or
        ///     <c>private</c>.</returns>
        public static string Of(TypeDefinition t)
        {
            if (!t.IsNested)
                return t.IsPublic ? "public" : "internal";
            if (t.IsNestedPublic) return "public";
            if (t.IsNestedFamilyOrAssembly) return "protected internal";
            if (t.IsNestedFamily) return "protected";
            if (t.IsNestedAssembly) return "internal";
            if (t.IsNestedFamilyAndAssembly) return "private protected";
            return "private";
        }

        /// <summary>The declared C# access of a method (or accessor).</summary>
        /// <param name="m">The method.</param>
        /// <returns>The C# access keyword(s).</returns>
        public static string Of(MethodDefinition m)
        {
            if (m.IsPublic) return "public";
            if (m.IsFamilyOrAssembly) return "protected internal";
            if (m.IsFamily) return "protected";
            if (m.IsAssembly) return "internal";
            if (m.IsFamilyAndAssembly) return "private protected";
            return "private";
        }

        /// <summary>The declared C# access of a field.</summary>
        /// <param name="f">The field.</param>
        /// <returns>The C# access keyword(s).</returns>
        public static string Of(FieldDefinition f)
        {
            if (f.IsPublic) return "public";
            if (f.IsFamilyOrAssembly) return "protected internal";
            if (f.IsFamily) return "protected";
            if (f.IsAssembly) return "internal";
            if (f.IsFamilyAndAssembly) return "private protected";
            return "private";
        }

        /// <summary>What a declared access alone lets an outsider do.</summary>
        /// <param name="access">A C# access spelling from one of the <c>Of</c> overloads.</param>
        /// <returns>The reach of that access in isolation.</returns>
        /// <remarks><c>private protected</c> is <see cref="Reach.Internal"/>: only a subclass INSIDE NumSharp.dll sees it,
        ///     which is exactly what a consumer's subclass cannot be.</remarks>
        public static Reach ReachOfAccess(string access) => access switch
        {
            "public" => Reach.Public,
            "protected" or "protected internal" => Reach.Protected,
            "internal" or "private protected" => Reach.Internal,
            _ => Reach.Private,
        };

        /// <summary>The effective reach of a type: the narrowest along its declaring-type chain.</summary>
        /// <param name="t">The type.</param>
        /// <returns>The reach.</returns>
        public static Reach ReachOf(TypeDefinition t)
        {
            var own = ReachOfAccess(Of(t));
            return t.IsNested ? (Reach)Math.Max((int)ReachOf(t.DeclaringType), (int)own) : own;
        }

        /// <summary>The effective reach of a member declared with <paramref name="access"/> in a type of reach <paramref name="owner"/>.</summary>
        /// <param name="owner">The declaring type's reach.</param>
        /// <param name="access">The member's declared access.</param>
        /// <returns>The narrower of the two.</returns>
        public static Reach Combine(Reach owner, string access) => (Reach)Math.Max((int)owner, (int)ReachOfAccess(access));

        /// <summary>
        ///     Whether a subclass can override the method: C# <c>virtual</c>, <c>abstract</c> or an unsealed <c>override</c>.
        /// </summary>
        /// <param name="m">The method (or accessor).</param>
        /// <param name="owner">The declaring type.</param>
        /// <returns>False for an interface member, a static member, and the <c>virtual final</c> slot the compiler gives a
        ///     non-virtual method that implements an interface.</returns>
        public static bool IsOverridable(MethodDefinition m, TypeDefinition owner)
            => !owner.IsInterface && !m.IsStatic && m.IsVirtual && !m.IsFinal;
    }

    /// <summary>C# spellings of Cecil types, members and modifiers.</summary>
    internal static class Names
    {
        /// <summary>CLR names the C# keyword spells.</summary>
        private static readonly Dictionary<string, string> Keywords = new(StringComparer.Ordinal)
        {
            ["System.Void"] = "void", ["System.Boolean"] = "bool", ["System.Byte"] = "byte", ["System.SByte"] = "sbyte",
            ["System.Int16"] = "short", ["System.UInt16"] = "ushort", ["System.Int32"] = "int", ["System.UInt32"] = "uint",
            ["System.Int64"] = "long", ["System.UInt64"] = "ulong", ["System.Char"] = "char", ["System.Single"] = "float",
            ["System.Double"] = "double", ["System.Decimal"] = "decimal", ["System.String"] = "string",
            ["System.Object"] = "object", ["System.IntPtr"] = "nint", ["System.UIntPtr"] = "nuint",
        };

        /// <summary>Operator method names -> their C# token.</summary>
        private static readonly Dictionary<string, string> Operators = new(StringComparer.Ordinal)
        {
            ["op_Addition"] = "+", ["op_Subtraction"] = "-", ["op_Multiply"] = "*", ["op_Division"] = "/", ["op_Modulus"] = "%",
            ["op_BitwiseAnd"] = "&", ["op_BitwiseOr"] = "|", ["op_ExclusiveOr"] = "^", ["op_LeftShift"] = "<<",
            ["op_RightShift"] = ">>", ["op_UnsignedRightShift"] = ">>>", ["op_Equality"] = "==", ["op_Inequality"] = "!=",
            ["op_LessThan"] = "<", ["op_GreaterThan"] = ">", ["op_LessThanOrEqual"] = "<=", ["op_GreaterThanOrEqual"] = ">=",
            ["op_UnaryNegation"] = "-", ["op_UnaryPlus"] = "+", ["op_LogicalNot"] = "!", ["op_OnesComplement"] = "~",
            ["op_Increment"] = "++", ["op_Decrement"] = "--", ["op_True"] = "true", ["op_False"] = "false",
        };

        /// <summary>
        ///     Whether the compiler (or a generator) made the member rather than a person: its name holds a character C# names
        ///     cannot (<c>&lt;</c>, <c>&gt;</c>, <c>$</c>), or it carries <c>[CompilerGenerated]</c> on a type.
        /// </summary>
        /// <param name="m">The type or member.</param>
        /// <returns>True for lambdas' closures, iterators, backing fields, file-local and anonymous types and the like.</returns>
        public static bool IsGenerated(IMemberDefinition m)
        {
            if (m.Name.IndexOfAny(['<', '>', '$']) >= 0)
                return true;
            if (m is TypeDefinition t)
                return t.CustomAttributes.Any(a => a.AttributeType.FullName == "System.Runtime.CompilerServices.CompilerGeneratedAttribute")
                       || (t.IsNested && IsGenerated(t.DeclaringType));
            return false;
        }

        /// <summary>The namespace of the outermost declaring type.</summary>
        /// <param name="t">The type.</param>
        /// <returns>The namespace, or <c>(global)</c>.</returns>
        public static string NamespaceOf(TypeDefinition t)
        {
            while (t.IsNested)
                t = t.DeclaringType;
            return t.Namespace.Length == 0 ? "(global)" : t.Namespace;
        }

        /// <summary>A type definition's C# spelling: <c>Outer&lt;T&gt;.Inner&lt;U&gt;</c> (namespace omitted).</summary>
        /// <param name="t">The type.</param>
        /// <returns>The spelling.</returns>
        public static string TypeDisplay(TypeDefinition t)
        {
            string own = StripArity(t.Name);
            // A nested type repeats its container's generic parameters in metadata; only the rest are its own.
            int inherited = t.IsNested ? t.DeclaringType.GenericParameters.Count : 0;
            var mine = t.GenericParameters.Skip(inherited).Select(g => g.Name).ToArray();
            if (mine.Length > 0)
                own += "<" + string.Join(", ", mine) + ">";
            return t.IsNested ? TypeDisplay(t.DeclaringType) + "." + own : own;
        }

        /// <summary>The C# spelling of a type reference as it appears in a signature.</summary>
        /// <param name="t">The type reference (any Cecil shape).</param>
        /// <returns>The spelling (<c>int</c>, <c>NDArray[]</c>, <c>Span&lt;T&gt;</c>, <c>(int, long)</c>, <c>T?</c>, <c>byte*</c>).</returns>
        public static string TypeName(TypeReference t)
        {
            switch (t)
            {
                case RequiredModifierType rm: return TypeName(rm.ElementType);
                case OptionalModifierType om: return TypeName(om.ElementType);
                case PinnedType pt: return TypeName(pt.ElementType);
                case ByReferenceType br: return "ref " + TypeName(br.ElementType);
                case PointerType pt: return TypeName(pt.ElementType) + "*";
                case ArrayType at: return TypeName(at.ElementType) + "[" + new string(',', Math.Max(0, at.Rank - 1)) + "]";
                case GenericParameter gp: return gp.Name;
                case FunctionPointerType fp:
                    return "delegate*<" + string.Join(", ", fp.Parameters.Select(p => TypeName(p.ParameterType)).Append(TypeName(fp.ReturnType))) + ">";
                case GenericInstanceType gi:
                    var args = gi.GenericArguments.Select(TypeName).ToArray();
                    if (gi.ElementType.FullName == "System.Nullable`1")
                        return args[0] + "?";
                    if (gi.ElementType.Namespace == "System" && gi.ElementType.Name.StartsWith("ValueTuple`", StringComparison.Ordinal))
                        return "(" + string.Join(", ", args) + ")";
                    return Plain(gi.ElementType, args);
                default:
                    return Plain(t, []);
            }
        }

        /// <summary>A non-composite type's spelling, with <paramref name="args"/> spread over its nesting levels.</summary>
        /// <param name="t">The (element) type.</param>
        /// <param name="args">The generic arguments of the instantiation, outermost first; empty for an open spelling.</param>
        /// <returns>The spelling.</returns>
        private static string Plain(TypeReference t, string[] args)
        {
            if (Keywords.TryGetValue(t.FullName, out var kw))
                return kw;
            // Generic arity is per nesting level in metadata (Outer`1/Inner`1), so the arguments are dealt out the same way.
            var chain = new List<TypeReference>();
            for (var c = t; c is not null; c = c.DeclaringType)
                chain.Insert(0, c);
            var parts = new List<string>();
            int used = 0;
            foreach (var c in chain)
            {
                int arity = Arity(c.Name);
                string name = StripArity(c.Name);
                if (arity > 0)
                {
                    var mine = args.Length >= used + arity
                        ? args.Skip(used).Take(arity)
                        : Enumerable.Range(0, arity).Select(i => c.GenericParameters.Count > i ? c.GenericParameters[i].Name : "T");
                    name += "<" + string.Join(", ", mine) + ">";
                    used += arity;
                }

                parts.Add(name);
            }

            return string.Join(".", parts);
        }

        /// <summary>The generic arity in a metadata name (<c>List`1</c> -> 1).</summary>
        /// <param name="name">The metadata name.</param>
        /// <returns>The arity, 0 when the name has no backtick.</returns>
        private static int Arity(string name)
        {
            int tick = name.IndexOf('`');
            return tick >= 0 && int.TryParse(name.AsSpan(tick + 1), out var n) ? n : 0;
        }

        /// <summary>A metadata name without its arity suffix.</summary>
        /// <param name="name">The metadata name.</param>
        /// <returns><c>List</c> for <c>List`1</c>.</returns>
        public static string StripArity(string name)
        {
            int tick = name.IndexOf('`');
            return tick >= 0 ? name[..tick] : name;
        }

        /// <summary>The C# kind of a type.</summary>
        /// <param name="t">The type.</param>
        /// <returns>interface / enum / delegate / [readonly ][ref ]struct / static class / abstract class / sealed class /
        ///     class (record instead of class for a record).</returns>
        public static string TypeKind(TypeDefinition t)
        {
            if (t.IsInterface) return "interface";
            if (t.IsEnum) return "enum";
            if (t.BaseType?.FullName == "System.MulticastDelegate") return "delegate";
            if (t.IsValueType)
            {
                bool ro = HasAttribute(t.CustomAttributes, "System.Runtime.CompilerServices.IsReadOnlyAttribute");
                bool rf = HasAttribute(t.CustomAttributes, "System.Runtime.CompilerServices.IsByRefLikeAttribute");
                return (ro ? "readonly " : "") + (rf ? "ref " : "") + "struct";
            }

            string noun = t.Methods.Any(m => m.Name == "<Clone>$") ? "record" : "class";
            if (t.IsAbstract && t.IsSealed) return "static class";
            if (t.IsAbstract) return "abstract " + noun;
            if (t.IsSealed) return "sealed " + noun;
            return noun;
        }

        /// <summary>Whether an attribute list holds an attribute of the given full name.</summary>
        /// <param name="attributes">The attributes.</param>
        /// <param name="fullName">The attribute type's full name.</param>
        /// <returns>True when present.</returns>
        public static bool HasAttribute(IEnumerable<CustomAttribute> attributes, string fullName)
            => attributes.Any(a => a.AttributeType.FullName == fullName);

        /// <summary>Whether a method is a C# finalizer: the parameterless <c>void</c> override of <c>object.Finalize</c>.</summary>
        /// <param name="m">The method.</param>
        /// <returns>True for <c>~Name()</c>.</returns>
        public static bool IsFinalizer(MethodDefinition m)
            => m.Name == "Finalize" && m.IsVirtual && !m.IsNewSlot && !m.HasParameters && m.ReturnType.FullName == "System.Void";

        /// <summary>Whether a method is a user-defined conversion (<c>op_Implicit</c> / <c>op_Explicit</c>).</summary>
        /// <param name="m">The method.</param>
        /// <returns>True for a conversion operator.</returns>
        public static bool IsConversion(MethodDefinition m)
            => m.IsSpecialName && m.Name is "op_Implicit" or "op_Explicit" or "op_CheckedExplicit";

        /// <summary>The C# name of a method: the type's name for a constructor, <c>operator +</c> for an operator.</summary>
        /// <param name="m">The method.</param>
        /// <param name="owner">The declaring type.</param>
        /// <returns>The name.</returns>
        public static string MethodName(MethodDefinition m, TypeInfo owner)
        {
            if (m.IsConstructor)
                return StripArity(owner.Def.Name);
            // C# spells an override of object.Finalize as a finalizer (`~Name()`); metadata only knows "Finalize".
            if (IsFinalizer(m))
                return "~" + StripArity(owner.Def.Name);
            if (m.IsSpecialName && m.Name == "op_Implicit") return "implicit operator";
            if (m.IsSpecialName && m.Name is "op_Explicit" or "op_CheckedExplicit") return "explicit operator";
            if (m.IsSpecialName && m.Name.StartsWith("op_", StringComparison.Ordinal))
            {
                // C# 11 checked operators share the token of their unchecked twin.
                string key = m.Name.Replace("op_Checked", "op_", StringComparison.Ordinal);
                if (Operators.TryGetValue(key, out var token))
                    return "operator " + token;
            }

            return m.Name;
        }

        /// <summary>A method's C# signature: return type, name, generic parameters and parameters (no modifiers).</summary>
        /// <param name="m">The method.</param>
        /// <param name="owner">The declaring type.</param>
        /// <returns>The signature.</returns>
        public static string MethodSignature(MethodDefinition m, TypeInfo owner)
        {
            var sb = new StringBuilder();
            string name = MethodName(m, owner);
            if (m.IsConstructor || IsFinalizer(m))
                sb.Append(name);
            else if (IsConversion(m))
                sb.Append(name).Append(' ').Append(TypeName(m.ReturnType));
            else
                sb.Append(TypeName(m.ReturnType)).Append(' ').Append(name);
            if (m.HasGenericParameters)
                sb.Append('<').Append(string.Join(", ", m.GenericParameters.Select(g => g.Name))).Append('>');
            sb.Append('(').Append(string.Join(", ", m.Parameters.Select((p, i) => ParameterText(m, p, i)))).Append(')');
            return sb.ToString();
        }

        /// <summary>A property's or indexer's C# signature, an accessor's access written only where it narrows the property's.</summary>
        /// <param name="p">The property.</param>
        /// <param name="propertyAccess">The property's own access (its most accessible accessor's).</param>
        /// <returns><c>Shape shape { get; internal set; }</c> / <c>NDArray this[int index] { get; set; }</c>.</returns>
        public static string PropertySignature(PropertyDefinition p, string propertyAccess)
        {
            string name = p.HasParameters
                ? "this[" + string.Join(", ", p.Parameters.Select(x => TypeName(x.ParameterType) + " " + x.Name)) + "]"
                : p.Name;
            var accessors = new List<string>();
            if (p.GetMethod is { } g) accessors.Add(AccessorText(g, "get", propertyAccess));
            if (p.SetMethod is { } s) accessors.Add(AccessorText(s, IsInitOnly(s) ? "init" : "set", propertyAccess));
            return $"{TypeName(p.PropertyType)} {name} {{ {string.Join(" ", accessors)} }}";
        }

        /// <summary>One accessor's text: its keyword, prefixed by its access when that differs from the property's (C#'s rule).</summary>
        /// <param name="a">The accessor method.</param>
        /// <param name="keyword">get / set / init.</param>
        /// <param name="propertyAccess">The property's own access.</param>
        /// <returns><c>get;</c> or <c>internal set;</c>.</returns>
        private static string AccessorText(MethodDefinition a, string keyword, string propertyAccess)
        {
            string access = Access.Of(a);
            return (access == propertyAccess ? "" : access + " ") + keyword + ";";
        }

        /// <summary>Whether a setter is an <c>init</c> accessor (its return type carries <c>modreq(IsExternalInit)</c>).</summary>
        /// <param name="setter">The setter.</param>
        /// <returns>True for <c>init</c>.</returns>
        private static bool IsInitOnly(MethodDefinition setter)
            => setter.ReturnType is RequiredModifierType rm && rm.ModifierType.FullName == "System.Runtime.CompilerServices.IsExternalInit";

        /// <summary>One parameter's C# text: <c>this</c> / <c>ref</c> / <c>out</c> / <c>in</c> / <c>params</c>, type, name.</summary>
        /// <param name="m">The method.</param>
        /// <param name="p">The parameter.</param>
        /// <param name="i">Its position (the first parameter of an extension method gets <c>this</c>).</param>
        /// <returns>The parameter text.</returns>
        private static string ParameterText(MethodDefinition m, ParameterDefinition p, int i)
        {
            var prefix = new StringBuilder();
            if (i == 0 && m.IsStatic && HasAttribute(m.CustomAttributes, "System.Runtime.CompilerServices.ExtensionAttribute"))
                prefix.Append("this ");
            var t = p.ParameterType;
            // An `in` parameter of a virtual method is modreq(InAttribute) around the by-ref type.
            while (t is RequiredModifierType rm)
                t = rm.ElementType;
            if (t is ByReferenceType br)
            {
                bool readOnly = p.IsIn || HasAttribute(p.CustomAttributes, "System.Runtime.CompilerServices.IsReadOnlyAttribute");
                prefix.Append(p.IsOut ? "out " : readOnly ? "in " : "ref ");
                t = br.ElementType;
            }

            if (HasAttribute(p.CustomAttributes, "System.ParamArrayAttribute"))
                prefix.Append("params ");
            return prefix + TypeName(t) + " " + p.Name;
        }

        /// <summary>A method's (or accessor's) C# modifiers.</summary>
        /// <param name="m">The method.</param>
        /// <param name="owner">The declaring type.</param>
        /// <returns>Space-separated modifiers: static, abstract, virtual, override, sealed override, extern.</returns>
        public static string Modifiers(MethodDefinition m, TypeDefinition owner)
        {
            var parts = new List<string>();
            if (m.IsStatic) parts.Add("static");
            if (owner.IsInterface)
            {
                // An interface member is implicitly abstract; a body makes it a default implementation.
                if (m.IsStatic && m.IsAbstract) parts.Add("abstract");
                else if (!m.IsStatic && !m.IsAbstract) parts.Add("default-impl");
            }
            else if (m.IsAbstract) parts.Add(m.IsNewSlot ? "abstract" : "abstract override");
            else if (m.IsVirtual && !m.IsFinal) parts.Add(m.IsNewSlot ? "virtual" : "override");
            else if (m.IsVirtual && m.IsFinal && !m.IsNewSlot) parts.Add("sealed override");
            if (m.IsPInvokeImpl || m.IsInternalCall) parts.Add("extern");
            return string.Join(" ", parts);
        }

        /// <summary>A field's C# modifiers.</summary>
        /// <param name="f">The field.</param>
        /// <returns>Space-separated modifiers: const, static, readonly, volatile.</returns>
        public static string FieldModifiers(FieldDefinition f)
        {
            if (f.IsLiteral)
                return "const";
            var parts = new List<string>();
            if (f.IsStatic) parts.Add("static");
            if (f.IsInitOnly) parts.Add("readonly");
            if (f.FieldType is RequiredModifierType rm && rm.ModifierType.FullName == "System.Runtime.CompilerServices.IsVolatile") parts.Add("volatile");
            return string.Join(" ", parts);
        }
    }

    /// <summary>The XML doc file's summaries, keyed by documentation-comment id.</summary>
    internal sealed class Docs
    {
        /// <summary>Doc id (<c>T:</c>, <c>M:</c>, <c>P:</c>, <c>F:</c>, <c>E:</c>) -> the summary's first sentence.</summary>
        private readonly Dictionary<string, string> _byId = new(StringComparer.Ordinal);

        /// <summary>Reads the XML doc file; a missing file gives an empty map (every summary empty), not an error.</summary>
        /// <param name="path">The XML doc file beside the DLL.</param>
        /// <returns>The summaries.</returns>
        /// <exception cref="System.Xml.XmlException">The file exists but is not well-formed XML.</exception>
        public static Docs Load(string path)
        {
            var docs = new Docs();
            if (!File.Exists(path))
                return docs;
            foreach (var member in XDocument.Load(path).Descendants("member"))
            {
                var id = (string?)member.Attribute("name");
                var summary = member.Element("summary");
                if (id is not null && summary is not null)
                    docs._byId[id] = FirstSentence(Flatten(summary));
            }

            return docs;
        }

        /// <summary>The summary of a type or member, or empty.</summary>
        /// <param name="member">The definition.</param>
        /// <returns>The first sentence of its summary.</returns>
        /// <remarks>Cecil's id builder can throw on exotic signatures (function pointers); such a member reads as
        ///     undocumented rather than stopping the map.</remarks>
        public string Get(IMemberDefinition member)
        {
            try
            {
                return _byId.TryGetValue(DocCommentId.GetDocCommentId(member), out var s) ? s : "";
            }
            catch (Exception)
            {
                return "";
            }
        }

        /// <summary>The plain text of a doc element: <c>see</c> / <c>paramref</c> become the names they point at.</summary>
        /// <param name="e">The element.</param>
        /// <returns>The text with whitespace collapsed.</returns>
        private static string Flatten(XElement e)
        {
            var sb = new StringBuilder();
            Append(e, sb);
            return Regex.Replace(sb.ToString(), @"\s+", " ").Trim();
        }

        /// <summary>Appends one doc node's text.</summary>
        /// <param name="node">The node.</param>
        /// <param name="sb">The text so far.</param>
        private static void Append(XNode node, StringBuilder sb)
        {
            if (node is XText text)
            {
                sb.Append(text.Value);
                return;
            }

            if (node is not XElement el)
                return;
            string local = el.Name.LocalName;
            if (local is "see" or "seealso" && !el.Nodes().Any())
            {
                sb.Append((string?)el.Attribute("langword") ?? ShortCref((string?)el.Attribute("cref")) ?? (string?)el.Attribute("href") ?? "");
                return;
            }

            if (local is "paramref" or "typeparamref")
            {
                sb.Append((string?)el.Attribute("name"));
                return;
            }

            foreach (var child in el.Nodes())
                Append(child, sb);
            if (local is "para" or "br")
                sb.Append(' ');
        }

        /// <summary>A cref as a reader would write it: the type for <c>T:</c>, <c>Type.Member</c> otherwise.</summary>
        /// <param name="cref">The cref attribute.</param>
        /// <returns>The short name, or <c>null</c> when there is no cref.</returns>
        private static string? ShortCref(string? cref)
        {
            if (cref is null)
                return null;
            bool isType = cref.StartsWith("T:", StringComparison.Ordinal);
            string s = cref.Length > 2 && cref[1] == ':' ? cref[2..] : cref;
            int paren = s.IndexOf('(');
            if (paren >= 0)
                s = s[..paren];
            var parts = s.Split('.').Select(Names.StripArity).ToArray();
            if (parts[^1] == "#ctor" && parts.Length > 1)
                return parts[^2];
            return isType || parts.Length < 2 ? parts[^1] : parts[^2] + "." + parts[^1];
        }

        /// <summary>The first sentence, capped at 220 characters.</summary>
        /// <param name="text">The flattened summary.</param>
        /// <returns>The sentence.</returns>
        /// <remarks>A period ends a sentence only before whitespace and a capital (or the end), and never after e.g. / i.e.
        ///     / vs / etc. / cf., so <c>np.sum</c> or <c>e.g. NDArray</c> does not cut it short.</remarks>
        private static string FirstSentence(string text)
        {
            var m = Regex.Match(text, @"^(.+?(?<!\b(?:e\.g|i\.e|vs|etc|cf))[.!?])(?=\s+[A-Z(""]|$)");
            string s = m.Success ? m.Groups[1].Value : text;
            return s.Length > 220 ? s[..217].TrimEnd() + "..." : s;
        }
    }

    /// <summary>Type declarations found by scanning Core's source text (for types and members the PDB cannot place).</summary>
    internal sealed class SourceIndex
    {
        /// <summary>Simple type name (no arity) -> the files and lines declaring a type of that name.</summary>
        public Dictionary<string, List<(string File, int Line)>> Declarations = new(StringComparer.Ordinal);

        /// <summary>Repository-relative file -> its lines (read once, searched for members without bodies).</summary>
        public Dictionary<string, string[]> Lines = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>A type declaration: <c>class X</c>, <c>struct X</c>, <c>interface X</c>, <c>enum X</c>, <c>record X</c>.</summary>
        private static readonly Regex TypeDeclaration = new(@"\b(?:class|struct|interface|enum|record)\s+(?<name>[A-Za-z_]\w*)", RegexOptions.Compiled);

        /// <summary>A delegate declaration: <c>delegate R X(</c> / <c>delegate R X&lt;</c>.</summary>
        private static readonly Regex DelegateDeclaration = new(@"\bdelegate\s+[\w<>\[\],.?* ]+?\s+(?<name>[A-Za-z_]\w*)\s*[<(]", RegexOptions.Compiled);

        /// <summary>Scans every <c>.cs</c> file under <paramref name="root"/> (bin / obj excluded).</summary>
        /// <param name="root">The project folder.</param>
        /// <param name="repo">The repository root the stored paths are relative to.</param>
        /// <returns>The index.</returns>
        public static SourceIndex Build(string root, string repo)
        {
            var index = new SourceIndex();
            foreach (var path in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories))
            {
                string rel = Locator.Relative(path, repo);
                if (rel.Contains("/bin/", StringComparison.Ordinal) || rel.Contains("/obj/", StringComparison.Ordinal))
                    continue;
                var lines = File.ReadAllLines(path);
                index.Lines[rel] = lines;
                for (int i = 0; i < lines.Length; i++)
                {
                    string code = lines[i].TrimStart();
                    // Comments and doc comments name types all the time; only code lines declare them.
                    if (code.StartsWith("//", StringComparison.Ordinal) || code.StartsWith("*", StringComparison.Ordinal))
                        continue;
                    foreach (Match m in TypeDeclaration.Matches(code))
                        index.Add(m.Groups["name"].Value, rel, i + 1);
                    foreach (Match m in DelegateDeclaration.Matches(code))
                        index.Add(m.Groups["name"].Value, rel, i + 1);
                }
            }

            return index;
        }

        /// <summary>File -> the namespaces it opens (block or file-scoped), computed on first ask.</summary>
        private readonly Dictionary<string, HashSet<string>> _namespaces = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        ///     Whether a file opens the namespace (<c>namespace X</c> or <c>namespace X;</c>), so a declaration found in it can
        ///     belong to a top-level type of that namespace.
        /// </summary>
        /// <param name="file">The repository-relative file.</param>
        /// <param name="ns">The namespace, or <c>(global)</c>.</param>
        /// <returns>True when the file opens <paramref name="ns"/>, or opens none and <paramref name="ns"/> is the global one.</returns>
        public bool DeclaresNamespace(string file, string ns)
        {
            if (!_namespaces.TryGetValue(file, out var set))
            {
                set = new HashSet<string>(StringComparer.Ordinal);
                if (Lines.TryGetValue(file, out var lines))
                    foreach (var line in lines)
                    {
                        var m = Regex.Match(line, @"^\s*namespace\s+(?<ns>[\w.]+)");
                        if (m.Success)
                            set.Add(m.Groups["ns"].Value);
                    }

                _namespaces[file] = set;
            }

            return ns == "(global)" ? set.Count == 0 : set.Contains(ns);
        }

        /// <summary>Records one declaration.</summary>
        /// <param name="name">The simple type name.</param>
        /// <param name="file">The repository-relative file.</param>
        /// <param name="line">The 1-based line.</param>
        private void Add(string name, string file, int line)
        {
            if (!Declarations.TryGetValue(name, out var list))
                Declarations[name] = list = new List<(string, int)>();
            list.Add((file, line));
        }
    }

    /// <summary>Source locations from the PDB, with a source-text fallback.</summary>
    internal static class Locator
    {
        /// <summary>A repository-relative path with forward slashes.</summary>
        /// <param name="path">An absolute path (a PDB document, a scanned file).</param>
        /// <param name="repo">The repository root.</param>
        /// <returns>The relative path; a path outside the repo keeps everything from its <c>src/</c> segment on (a PDB built
        ///     elsewhere, or with a path map).</returns>
        public static string Relative(string path, string repo)
        {
            string p = path.Replace('\\', '/');
            string r = repo.Replace('\\', '/').TrimEnd('/') + "/";
            if (p.StartsWith(r, StringComparison.OrdinalIgnoreCase))
                return p[r.Length..];
            int src = p.IndexOf("/src/", StringComparison.OrdinalIgnoreCase);
            return src >= 0 ? p[(src + 1)..] : p;
        }

        /// <summary>The first visible sequence point of a method body.</summary>
        /// <param name="m">The method.</param>
        /// <param name="repo">The repository root.</param>
        /// <returns>The file and line, or <c>("", 0)</c> for an abstract / extern method or a build without symbols.</returns>
        /// <remarks>A class constructor runs its type's field initializers first, each with a sequence point on its field's
        ///     line, so its first point is a field; the point of its base / this constructor call is the constructor's own
        ///     (the <c>: base(...)</c> text, or the declaration for an implicit one), and is taken instead.</remarks>
        public static (string File, int Line) Locate(MethodDefinition m, string repo)
        {
            if (!m.HasBody || m.DebugInformation is null || !m.DebugInformation.HasSequencePoints)
                return ("", 0);
            var points = m.DebugInformation.SequencePoints;
            SequencePoint? chosen = null;
            if (m.IsConstructor && !m.IsStatic && m.DeclaringType.BaseType is { } baseType)
            {
                // The chaining call: a constructor of the base class or of this class (`: this(...)`), on `this`.
                string own = m.DeclaringType.FullName, parent = baseType.GetElementType().FullName;
                var chain = m.Body.Instructions.FirstOrDefault(i => i.OpCode == OpCodes.Call && i.Operand is MethodReference { Name: ".ctor" } r
                    && r.DeclaringType.GetElementType().FullName is var t && (t == parent || t == own));
                if (chain is not null)
                    chosen = points.Where(sp => !sp.IsHidden && sp.Offset <= chain.Offset).OrderBy(sp => sp.Offset).LastOrDefault();
            }

            chosen ??= points.FirstOrDefault(sp => !sp.IsHidden);
            if (chosen is null)
                return ("", 0);
            string file = Relative(chosen.Document.Url, repo);
            // A source generator's body (a [GeneratedRegex] partial method) lives under obj/; the declaration a reader wants
            // is the partial one in the type's own file, which the text search finds instead.
            return file.Contains("/obj/", StringComparison.OrdinalIgnoreCase) ? ("", 0) : (file, chosen.StartLine);
        }

        /// <summary>
        ///     Places a type: the file most of its located members come from, and its declaration line in that file (or any
        ///     declaration of its name when no member is located); also collects every file of the type into
        ///     <see cref="TypeInfo.Files"/>.
        /// </summary>
        /// <param name="ti">The type (its members already located).</param>
        /// <param name="index">The source index.</param>
        /// <param name="repo">The repository root.</param>
        /// <param name="outer">The declaring type of a nested type (already placed: Cecil lists a type before its nested
        ///     types), or <c>null</c> for a top-level type.</param>
        public static void LocateType(TypeInfo ti, SourceIndex index, string repo, TypeInfo? outer)
        {
            var located = ti.Members.Where(m => m.File.Length > 0).Select(m => (m.File, m.Line)).ToList();
            // A type whose visible members have no body (only private helpers, an interface with defaults, a field-only
            // struct) is still placed by its method bodies: count every one, private ones included, before the text search.
            if (located.Count == 0)
                located = ti.Def.Methods.Select(m => Locate(m, repo)).Where(l => l.Line > 0).ToList();
            var files = located.GroupBy(l => l.File).OrderByDescending(g => g.Count()).ThenBy(g => g.Key.Length).ToList();
            ti.FileCount = files.Count;
            string name = Names.StripArity(ti.Def.Name);
            index.Declarations.TryGetValue(name, out var found);
            // A simple name is not unique (every BitGenerator has a nested `State`): a nested type's declarations count only
            // inside its declaring type's files, a top-level type's only in files that open its namespace.
            var declarations = found?.Where(d => outer is not null
                ? outer.Files.Contains(d.File, StringComparer.OrdinalIgnoreCase)
                : index.DeclaresNamespace(d.File, ti.Namespace)).ToList();
            ti.Files.AddRange(files.Select(g => g.Key));
            foreach (var d in declarations ?? [])
                if (!ti.Files.Contains(d.File, StringComparer.OrdinalIgnoreCase))
                    ti.Files.Add(d.File);
            if (files.Count > 0)
            {
                ti.File = files[0].Key;
                var inFile = declarations?.Where(d => d.File.Equals(ti.File, StringComparison.OrdinalIgnoreCase)).ToList();
                int firstMember = files[0].Min(l => l.Line);
                ti.Line = inFile is { Count: > 0 }
                    // Several same-named declarations in one file (nested twins): the last one above the first member wins.
                    ? inFile.Where(d => d.Line <= firstMember).Select(d => d.Line).DefaultIfEmpty(inFile[0].Line).Max()
                    : firstMember;
            }
            else if (declarations is { Count: > 0 })
            {
                // Prefer a declaration whose folder matches the namespace's last segment; otherwise the first one.
                string last = ti.Namespace.Split('.')[^1];
                var pick = declarations.FirstOrDefault(d => d.File.Contains("/" + last + "/", StringComparison.OrdinalIgnoreCase));
                if (pick.File is null)
                    pick = declarations[0];
                (ti.File, ti.Line) = pick;
                ti.FileCount = declarations.Select(d => d.File).Distinct(StringComparer.OrdinalIgnoreCase).Count();
                // The chosen declaration's file leads the search list, as the primary file does when members place it.
                ti.Files.Remove(ti.File);
                ti.Files.Insert(0, ti.File);
            }

            ti.Area = AreaOf(ti.File);
        }

        /// <summary>
        ///     Places members the PDB cannot (fields, constants, abstract and interface members) by searching every file of
        ///     the declaring type - the primary file from the type's declaration on, then each other part of a partial type -
        ///     for the member's name in a declaration position.
        /// </summary>
        /// <param name="ti">The type.</param>
        /// <param name="index">The source index.</param>
        /// <remarks>
        ///     Two passes. The strict one also requires the declaration's own keywords on the line (its access, and
        ///     <c>abstract</c> / <c>virtual</c> / <c>const</c> / <c>static</c>), because a partial type's other files are full
        ///     of same-named overrides in nested node classes and calls; the loose one (the name alone) catches what the strict
        ///     one misses (a declaration split over lines). A member found nowhere keeps the primary file with line 0 (the
        ///     outputs then link the type's declaration).
        /// </remarks>
        public static void LocateMembersWithoutBodies(TypeInfo ti, SourceIndex index)
        {
            if (ti.Files.Count == 0)
                return;
            foreach (var m in ti.Members.Where(m => m.Line == 0))
            {
                var rx = new Regex(DeclarationPattern(m));
                var words = DeclarationWords(m).Select(w => new Regex(@"\b" + w + @"\b")).ToArray();
                foreach (bool strict in new[] { true, false })
                {
                    foreach (var file in ti.Files)
                    {
                        if (!index.Lines.TryGetValue(file, out var lines))
                            continue;
                        // In the primary file the declaration line is known, so mentions above the type (another type's
                        // members) are skipped; the other parts are searched whole.
                        int start = file.Equals(ti.File, StringComparison.OrdinalIgnoreCase) ? Math.Max(0, ti.Line - 1) : 0;
                        for (int i = start; i < lines.Length && m.Line == 0; i++)
                        {
                            string code = lines[i].TrimStart();
                            if (code.StartsWith("//", StringComparison.Ordinal) || code.StartsWith("*", StringComparison.Ordinal))
                                continue;
                            if (rx.IsMatch(code) && (!strict || words.All(w => w.IsMatch(code))))
                                (m.File, m.Line) = (file, i + 1);
                        }

                        if (m.Line > 0)
                            break;
                    }

                    if (m.Line > 0)
                        break;
                }

                if (m.Line == 0)
                {
                    // A parameterless constructor found nowhere is the one C# supplies when a type declares none: marked as
                    // such and linked to its type's declaration, like the implicit ones RefineToSignatures meets.
                    if (m.Kind == "ctor" && m.Def is MethodDefinition { HasParameters: false })
                    {
                        m.Modifiers = (m.Modifiers + " implicit").Trim();
                        (m.File, m.Line) = (ti.File, ti.Line);
                    }
                    else
                    {
                        m.File = ti.File;
                    }
                }
            }
        }

        /// <summary>
        ///     The regex of a member's name in DECLARATION position, shared by the signature refinement and the text search:
        ///     a method or constructor name before its parameter list (a body's <c>nameof(setitem)</c> is not one), a
        ///     property or event name ending the line or opening its accessors, a field name before its initializer or end.
        /// </summary>
        /// <param name="m">The member.</param>
        /// <returns>The pattern; a constructor's excludes object creations (<c>new Name(</c> / <c>new Name {</c>).</returns>
        private static string DeclarationPattern(MemberInfo m) => m.Kind switch
        {
            "ctor" => @"(?<!\bnew\s+)\b" + Regex.Escape(Names.StripArity(m.Owner.Def.Name)) + @"\s*\(",
            "operator" or "conversion" => @"\boperator\b",
            "indexer" => @"\bthis\s*\[",
            // A finalizer is spelled `~Name(`: no word boundary sits before the tilde.
            "method" when m.Name.StartsWith('~') => Regex.Escape(m.Name) + @"\s*\(",
            "method" => @"\b" + Regex.Escape(m.Name) + @"\s*[<(]",
            "field" or "const" => @"\b" + Regex.Escape(m.Name) + @"\s*(?:[=;,]|$)",
            _ => @"\b" + Regex.Escape(m.Name) + @"\s*(?:$|\{|=>|;)",
        };

        /// <summary>
        ///     Moves each PDB-placed member from its first sequence point to its signature line. A block body's first point
        ///     is its opening brace and an expression body's is the expression, both below the signature (a long signature
        ///     spans several lines), so the nearest line above that names the member is the declaration.
        /// </summary>
        /// <param name="model">The model (every member already placed).</param>
        /// <param name="index">The source index (the files' lines).</param>
        /// <remarks>Searches at most 40 lines up and stops at the first line naming the member; a line that already names it
        ///     is kept, so an expression-bodied one-liner does not move.</remarks>
        public static void RefineToSignatures(Model model, SourceIndex index)
        {
            foreach (var m in model.Types.SelectMany(t => t.Members))
            {
                if (m.Line <= 0 || m.Kind is "field" or "const" || !index.Lines.TryGetValue(m.File, out var lines) || m.Line > lines.Length)
                    continue;
                var rx = new Regex(DeclarationPattern(m));
                bool found = false;
                for (int i = m.Line - 1; i >= 0 && i >= m.Line - 41; i--)
                {
                    string code = lines[i].TrimStart();
                    // Comments, attributes and body lines (an expression body's `=> Name(...)` forwarding to an overload, a
                    // block's braces, a returned or thrown call) never declare the member.
                    if (code.StartsWith("//", StringComparison.Ordinal) || code.StartsWith("*", StringComparison.Ordinal) || code.StartsWith("[", StringComparison.Ordinal)
                        || code.StartsWith("=>", StringComparison.Ordinal) || code.StartsWith("{", StringComparison.Ordinal) || code.StartsWith("}", StringComparison.Ordinal)
                        || code.StartsWith("return ", StringComparison.Ordinal) || code.StartsWith("throw ", StringComparison.Ordinal))
                        continue;
                    if (rx.IsMatch(code))
                    {
                        m.Line = i + 1;
                        found = true;
                        break;
                    }
                }

                // A parameterless constructor whose only sequence points are field initializers was never written: C#
                // supplied it, so it is marked as such and linked to its type's declaration instead of a field line.
                if (!found && m.Kind == "ctor" && m.Def is MethodDefinition { HasParameters: false } && m.Owner.Line > 0)
                {
                    m.Modifiers = (m.Modifiers + " implicit").Trim();
                    (m.File, m.Line) = (m.Owner.File, m.Owner.Line);
                }
            }
        }

        /// <summary>The keywords a member's declaration line must carry in the strict search pass.</summary>
        /// <param name="m">The member.</param>
        /// <returns>Its access words (none for an interface member, whose access is implicit) and its distinguishing
        ///     modifier: <c>abstract</c>, <c>virtual</c>, <c>override</c>, <c>const</c> or <c>static</c>.</returns>
        private static IEnumerable<string> DeclarationWords(MemberInfo m)
        {
            if (!m.Owner.Def.IsInterface)
                foreach (var word in m.Access.Split(' '))
                    yield return word;
            var modifiers = m.Modifiers.Split(' ');
            if (m.IsAbstract)
                yield return "abstract";
            else if (modifiers.Contains("virtual"))
                yield return "virtual";
            else if (modifiers.Contains("override"))
                yield return "override";
            if (m.Kind == "const")
                yield return "const";
            else if (m.IsStatic)
                yield return "static";
        }

        /// <summary>The area of a Core file: its folder under <c>src/NumSharp.Core</c>, at most two levels deep.</summary>
        /// <param name="file">The repository-relative file.</param>
        /// <returns><c>Backends/Kernels</c>, <c>Utilities</c>, <c>(root)</c>, or <c>(unknown)</c>.</returns>
        public static string AreaOf(string file)
        {
            const string core = "src/NumSharp.Core/";
            if (!file.StartsWith(core, StringComparison.OrdinalIgnoreCase))
                return "(unknown)";
            var parts = file[core.Length..].Split('/');
            return parts.Length switch
            {
                1 => "(root)",
                2 => parts[0],
                _ => parts[0] + "/" + parts[1],
            };
        }
    }

    /// <summary>
    ///     The ways a visible type is closed to a consumer who subclasses or implements it. Categories:
    ///     <list type="bullet">
    ///         <item><c>ctor-closed</c>: a non-sealed class with no public / protected instance constructor.</item>
    ///         <item><c>abstract-closed</c>: an abstract member a consumer cannot see, so no consumer subclass compiles.</item>
    ///         <item><c>interface-closed</c>: an interface member that is not public, so no consumer type can implement it.</item>
    ///         <item><c>hidden-hook</c>: an <c>internal virtual</c> slot - an override point only NumSharp.dll's own classes reach.</item>
    ///         <item><c>hidden-hook-override</c>: an internal override of such a slot (shows which classes use the hook).</item>
    ///         <item><c>private-protected</c>: a member for subclasses inside NumSharp.dll only.</item>
    ///         <item><c>narrow-accessor</c>: a visible property / event whose other accessor is internal.</item>
    ///         <item><c>internal-state</c>: an internal instance field or property of a non-sealed class (state a subclass needs
    ///             and cannot reach).</item>
    ///         <item><c>internal-interface</c>: a visible type implementing an internal interface (library code that dispatches
    ///             on that interface never sees a consumer's type).</item>
    ///         <item><c>sealed-class</c> / <c>static-class</c> / <c>value-type</c>: closed by declaration.</item>
    ///     </list>
    /// </summary>
    internal static class Extensibility
    {
        /// <summary>Collects every finding over the visible types.</summary>
        /// <param name="model">The model.</param>
        /// <returns>The findings, type by type.</returns>
        public static List<Finding> Analyze(Model model)
        {
            var findings = new List<Finding>();
            foreach (var ti in model.Types.Where(t => t.Reach is Reach.Public or Reach.Protected).OrderBy(t => t.Namespace).ThenBy(t => t.Display))
            {
                var td = ti.Def;
                if (ti.Kind == "interface")
                {
                    foreach (var m in ti.Members.Where(m => m.Access != "public"))
                        findings.Add(new Finding("interface-closed", ti, m, $"{m.Access} interface member: no type outside NumSharp.dll can implement {ti.Display}"));
                    continue;
                }

                if (ti.Kind is "enum" or "delegate")
                    continue;
                if (td.IsValueType)
                {
                    findings.Add(new Finding("value-type", ti, null, $"{ti.Kind}: not inheritable by nature"));
                    continue;
                }

                if (ti.Kind == "static class")
                {
                    findings.Add(new Finding("static-class", ti, null, "static: extended with extension methods, never inherited"));
                    continue;
                }

                if (td.IsSealed)
                {
                    findings.Add(new Finding("sealed-class", ti, null, $"sealed ({SealedGroup(td)})"));
                    continue;
                }

                var ctors = ti.Members.Where(m => m.Kind == "ctor" && !m.IsStatic).ToList();
                if (ctors.Count > 0 && ctors.All(c => c.Reach == Reach.Internal))
                    findings.Add(new Finding("ctor-closed", ti, null,
                        "every instance constructor is internal: " + string.Join("; ", ctors.Select(c => $"{c.Access} {c.Signature}"))));
                else if (ctors.Count == 0 && td.Methods.Any(m => m.IsConstructor && !m.IsStatic))
                    findings.Add(new Finding("ctor-closed", ti, null, "every instance constructor is private"));

                foreach (var m in ti.Members.Where(m => m.Reach == Reach.Internal))
                {
                    if (m.IsAbstract)
                        findings.Add(new Finding("abstract-closed", ti, m, $"{m.Access} abstract: a subclass outside NumSharp.dll cannot implement it, so it cannot derive from {ti.Display}"));
                    else if (m.Overridable && m.IntroducesSlot)
                        findings.Add(new Finding("hidden-hook", ti, m, $"{m.Access} virtual: an override point only NumSharp.dll's classes can use"));
                    else if (m.Overridable || m.Modifiers.Contains("override", StringComparison.Ordinal))
                        findings.Add(new Finding("hidden-hook-override", ti, m, $"{m.Access} {m.Modifiers}"));
                    if (m.Access == "private protected")
                        findings.Add(new Finding("private-protected", ti, m, "visible to subclasses inside NumSharp.dll only"));
                    else if (!m.IsStatic && m.Kind is "field" or "property" or "indexer" && !m.Overridable)
                        findings.Add(new Finding("internal-state", ti, m, $"{m.Access} {m.Kind}: hidden from a consumer's subclass"));
                }

                foreach (var m in ti.Members.Where(m => m.Reach != Reach.Internal && m.NarrowAccessor.Length > 0 && Access.ReachOfAccess(m.NarrowAccessor.Split(' ')[0]) == Reach.Internal))
                    findings.Add(new Finding("narrow-accessor", ti, m, $"{m.Access} {m.Kind} with {m.NarrowAccessor}"));
            }

            // Internal interfaces on visible types (structs included: library code can test a boxed value against them).
            foreach (var ti in model.Types.Where(t => t.Reach is Reach.Public or Reach.Protected))
                foreach (var impl in ti.Def.Interfaces)
                {
                    var itd = SafeResolve(impl.InterfaceType);
                    if (itd is not null && model.ByType.TryGetValue(itd, out var iti) && iti.Reach == Reach.Internal)
                        findings.Add(new Finding("internal-interface", ti, null, $"implements internal {iti.Namespace}.{iti.Display}"));
                }

            return findings;
        }

        /// <summary>What a sealed class is, by its base chain: an exception, an attribute, or anything else.</summary>
        /// <param name="td">The sealed class.</param>
        /// <returns><c>exception</c>, <c>attribute</c>, <c>event args</c> or <c>other</c>.</returns>
        public static string SealedGroup(TypeDefinition td)
        {
            for (var b = td.BaseType; b is not null;)
            {
                switch (b.FullName)
                {
                    case "System.Exception": return "exception";
                    case "System.Attribute": return "attribute";
                    case "System.EventArgs": return "event args";
                }

                var resolved = SafeResolve(b);
                if (resolved is null)
                    break;
                b = resolved.BaseType;
            }

            return "other";
        }

        /// <summary>Resolves a type reference, or <c>null</c> when its assembly is not found (the map only needs best effort here).</summary>
        /// <param name="t">The reference.</param>
        /// <returns>The definition, or <c>null</c>.</returns>
        public static TypeDefinition? SafeResolve(TypeReference t)
        {
            try
            {
                return t.Resolve();
            }
            catch (AssemblyResolutionException)
            {
                return null;
            }
        }
    }

    /// <summary>The internals NumSharp's own InternalsVisibleTo packages reach, read from their IL.</summary>
    internal sealed class FriendScan
    {
        /// <summary>Every (friend, internal type or member, way of naming it) with its sites and users.</summary>
        public List<FriendUse> Uses = new();

        /// <summary>The friend assemblies that were scanned, with their paths.</summary>
        public List<(string Name, string Path)> Scanned = new();

        /// <summary>The friend names with no build output under <c>src/</c> (test projects, other repositories).</summary>
        public List<string> NotScanned = new();

        /// <summary>The lookup of <see cref="Uses"/> by (friend, target, how).</summary>
        private readonly Dictionary<(string, object, string), FriendUse> _byKey = new();

        /// <summary>The model whose internals are being matched.</summary>
        private Model _model = null!;

        /// <summary>
        ///     Scans every friend that has a build output under <c>src/&lt;name&gt;/bin/&lt;config&gt;/&lt;tfm&gt;</c>.
        /// </summary>
        /// <param name="model">The model (its resolver hands out the same NumSharp.dll).</param>
        /// <param name="options">Where the build outputs live.</param>
        /// <returns>The uses, sorted by friend, then by sites (most first).</returns>
        public static FriendScan Run(Model model, Options options)
        {
            var scan = new FriendScan { _model = model };
            foreach (var name in model.FriendNames.Distinct())
            {
                string path = Path.Combine(options.Repo, "src", name, "bin", options.Configuration, options.Tfm, name + ".dll");
                if (!File.Exists(path))
                {
                    scan.NotScanned.Add(name);
                    continue;
                }

                // A friend can reference another friend (MLNet -> OnnxRuntime): its own folder resolves them.
                model.Resolver.AddSearchDirectory(Path.GetDirectoryName(path)!);
                var module = ModuleDefinition.ReadModule(path, new ReaderParameters { AssemblyResolver = model.Resolver, InMemory = true });
                scan.Scanned.Add((name, Locator.Relative(path, options.Repo)));
                scan.ScanModule(name, module);
            }

            scan.Uses = scan.Uses.OrderBy(u => u.Friend, StringComparer.Ordinal).ThenByDescending(u => u.Sites).ThenBy(u => u.Type.Display).ToList();
            return scan;
        }

        /// <summary>Walks every declaration and IL instruction of one friend module.</summary>
        /// <param name="friend">The friend's assembly name.</param>
        /// <param name="module">The friend module.</param>
        private void ScanModule(string friend, ModuleDefinition module)
        {
            foreach (var type in module.GetTypes())
            {
                string typeUser = type.FullName;
                if (type.BaseType is not null)
                    UseType(friend, type.BaseType, typeUser, "derives");
                foreach (var impl in type.Interfaces)
                    UseType(friend, impl.InterfaceType, typeUser, "implements");
                foreach (var f in type.Fields)
                    UseType(friend, f.FieldType, typeUser, "type");
                foreach (var p in type.Properties)
                    UseType(friend, p.PropertyType, typeUser, "type");
                foreach (var m in type.Methods)
                {
                    string user = $"{type.FullName}::{m.Name}";
                    UseType(friend, m.ReturnType, user, "type");
                    foreach (var p in m.Parameters)
                        UseType(friend, p.ParameterType, user, "type");
                    foreach (var a in m.CustomAttributes)
                        UseMethod(friend, a.Constructor, user);
                    // An override of an internal virtual slot is the strongest use: the friend extends NumSharp through it.
                    if (m.IsVirtual && !m.IsNewSlot)
                    {
                        var baseMethod = SafeBase(m);
                        if (baseMethod is not null && baseMethod != m && _model.ByMethod.TryGetValue(baseMethod, out var hook) && hook.Reach == Reach.Internal)
                            Record(friend, hook.Owner, hook, "overrides", user);
                    }

                    foreach (var o in m.Overrides)
                        UseMethod(friend, o, user, "overrides");
                    if (!m.HasBody)
                        continue;
                    foreach (var v in m.Body.Variables)
                        UseType(friend, v.VariableType, user, "type");
                    foreach (var ins in m.Body.Instructions)
                    {
                        switch (ins.Operand)
                        {
                            case MethodReference mr: UseMethod(friend, mr, user); break;
                            case FieldReference fr: UseField(friend, fr, user); break;
                            case TypeReference tr: UseType(friend, tr, user, "type"); break;
                        }
                    }
                }

                foreach (var a in type.CustomAttributes)
                    UseMethod(friend, a.Constructor, typeUser);
            }
        }

        /// <summary>The method a friend override overrides, or <c>null</c> when its base chain does not resolve.</summary>
        /// <param name="m">The overriding method.</param>
        /// <returns>The base method definition.</returns>
        private static MethodDefinition? SafeBase(MethodDefinition m)
        {
            try
            {
                return m.GetBaseMethod();
            }
            catch (AssemblyResolutionException)
            {
                return null;
            }
        }

        /// <summary>Records every internal NumSharp type a type reference names (its element and generic arguments too).</summary>
        /// <param name="friend">The friend.</param>
        /// <param name="t">The type reference.</param>
        /// <param name="user">The friend member naming it.</param>
        /// <param name="how">derives / implements / type.</param>
        private void UseType(string friend, TypeReference? t, string user, string how)
        {
            foreach (var part in Parts(t))
            {
                if (!IsCore(part))
                    continue;
                var td = Extensibility.SafeResolve(part);
                if (td is not null && _model.ByType.TryGetValue(td, out var ti) && ti.Reach == Reach.Internal)
                    Record(friend, ti, null, how, user);
            }
        }

        /// <summary>Records a method reference: its generic arguments' types, its declaring type, and the method itself when internal.</summary>
        /// <param name="friend">The friend.</param>
        /// <param name="mr">The method reference.</param>
        /// <param name="user">The friend member naming it.</param>
        /// <param name="how">call (default) or overrides.</param>
        private void UseMethod(string friend, MethodReference mr, string user, string how = "call")
        {
            if (mr is GenericInstanceMethod gim)
            {
                foreach (var arg in gim.GenericArguments)
                    UseType(friend, arg, user, "type");
                mr = gim.ElementMethod;
            }

            UseType(friend, mr.DeclaringType, user, "type");
            if (!IsCore(mr.DeclaringType))
                return;
            MethodDefinition? md;
            try
            {
                md = mr.Resolve();
            }
            catch (AssemblyResolutionException)
            {
                return;
            }

            if (md is not null && _model.ByMethod.TryGetValue(md, out var mi) && mi.Reach == Reach.Internal)
                Record(friend, mi.Owner, mi, how, user);
            // An accessor narrower than its property (public get; internal set): the use is of the hidden accessor.
            else if (md is not null && _model.ByMethod.TryGetValue(md, out var prop) && Access.ReachOfAccess(Access.Of(md)) == Reach.Internal)
                Record(friend, prop.Owner, prop, how + " (" + Access.Of(md) + " accessor)", user);
        }

        /// <summary>Records a field reference when the field is internal.</summary>
        /// <param name="friend">The friend.</param>
        /// <param name="fr">The field reference.</param>
        /// <param name="user">The friend member naming it.</param>
        private void UseField(string friend, FieldReference fr, string user)
        {
            UseType(friend, fr.DeclaringType, user, "type");
            if (!IsCore(fr.DeclaringType))
                return;
            FieldDefinition? fd;
            try
            {
                fd = fr.Resolve();
            }
            catch (AssemblyResolutionException)
            {
                return;
            }

            if (fd is not null && _model.ByField.TryGetValue(fd, out var mi) && mi.Reach == Reach.Internal)
                Record(friend, mi.Owner, mi, "field", user);
        }

        /// <summary>Adds one site to a (friend, target, how) use.</summary>
        /// <param name="friend">The friend.</param>
        /// <param name="ti">The internal type, or the internal member's declaring type.</param>
        /// <param name="mi">The internal member, or <c>null</c> for a use of the type.</param>
        /// <param name="how">How it is named.</param>
        /// <param name="user">The friend member naming it.</param>
        private void Record(string friend, TypeInfo ti, MemberInfo? mi, string how, string user)
        {
            var key = (friend, (object?)mi ?? ti, how);
            if (!_byKey.TryGetValue(key, out var use))
            {
                use = new FriendUse { Friend = friend, Type = ti, Member = mi, How = how };
                _byKey[key] = use;
                Uses.Add(use);
            }

            use.Sites++;
            use.Users.Add(user);
            if (mi is not null)
                mi.Friends.Add(friend);
            else
                ti.Friends.Add(friend);
        }

        /// <summary>A type reference and every type inside it (element types, generic arguments), recursively.</summary>
        /// <param name="t">The reference.</param>
        /// <returns>The parts, the reference itself first.</returns>
        private static IEnumerable<TypeReference> Parts(TypeReference? t)
        {
            if (t is null)
                yield break;
            yield return t;
            if (t is GenericInstanceType gi)
            {
                foreach (var p in Parts(gi.ElementType))
                    yield return p;
                foreach (var a in gi.GenericArguments)
                foreach (var p in Parts(a))
                    yield return p;
            }
            else if (t is TypeSpecification spec)
            {
                foreach (var p in Parts(spec.ElementType))
                    yield return p;
            }
        }

        /// <summary>Whether a (non-composite) type reference points into NumSharp.dll.</summary>
        /// <param name="t">The reference.</param>
        /// <returns>True when its outermost declaring type's scope is the NumSharp assembly.</returns>
        private bool IsCore(TypeReference t)
        {
            if (t is TypeSpecification or GenericParameter)
                return false;
            while (t.DeclaringType is not null)
                t = t.DeclaringType;
            return t.Scope switch
            {
                AssemblyNameReference anr => anr.Name == _model.Assembly.Name.Name,
                ModuleDefinition md => md == _model.Assembly.MainModule,
                _ => false,
            };
        }
    }

    /// <summary>Source-text scans of Core that metadata cannot answer.</summary>
    internal static class SourceScan
    {
        /// <summary><c>x.GetType() == typeof(T)</c> / <c>!=</c>, and the mirrored order.</summary>
        private static readonly Regex AgainstTypeof = new(
            @"GetType\(\)\s*(?:==|!=)\s*typeof\(\s*(?<t>[^()]+?)\s*\)|typeof\(\s*(?<t>[^()]+?)\s*\)\s*(?:==|!=)\s*[\w.\[\]]+(?:\(\))?\.GetType\(\)",
            RegexOptions.Compiled);

        /// <summary><c>a.GetType() == b.GetType()</c>: an exact class match between two objects.</summary>
        private static readonly Regex AgainstObject = new(@"GetType\(\)\s*(?:==|!=)\s*[\w.\[\]]+(?:\(\))?\.GetType\(\)", RegexOptions.Compiled);

        /// <summary>A local typed from <c>GetType()</c> then compared: <c>Type t = a.GetType(); if (t == typeof(X))</c>.</summary>
        private static readonly Regex TypeLocalCompare = new(@"\b(?<v>[a-z]\w*)\s*(?:==|!=)\s*typeof\(\s*(?<t>[^()]+?)\s*\)", RegexOptions.Compiled);

        /// <summary>
        ///     Every exact class test in Core's code lines (comments skipped). An exact test answers "is this precisely X",
        ///     so a subclass of X - a consumer's included - takes the other branch.
        /// </summary>
        /// <param name="options">Where the source lives.</param>
        /// <returns>The tests, by file and line.</returns>
        public static List<ExactTypeCheck> ExactTypeChecks(Options options)
        {
            var found = new List<ExactTypeCheck>();
            string root = Path.Combine(options.Repo, "src", "NumSharp.Core");
            foreach (var path in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories).OrderBy(p => p, StringComparer.Ordinal))
            {
                string rel = Locator.Relative(path, options.Repo);
                if (rel.Contains("/bin/", StringComparison.Ordinal) || rel.Contains("/obj/", StringComparison.Ordinal))
                    continue;
                var lines = File.ReadAllLines(path);
                // Locals assigned from GetType() in this file: `Type t = a.GetType();` makes a later `t == typeof(X)` exact.
                var typeLocals = new HashSet<string>(StringComparer.Ordinal);
                for (int i = 0; i < lines.Length; i++)
                {
                    string code = lines[i].Trim();
                    if (code.StartsWith("//", StringComparison.Ordinal) || code.StartsWith("*", StringComparison.Ordinal))
                        continue;
                    foreach (Match m in Regex.Matches(code, @"\b(?:Type|var)\s+(?<v>[a-z]\w*)\s*=\s*[\w.\[\]()]+\.GetType\(\)"))
                        typeLocals.Add(m.Groups["v"].Value);
                    bool hit = false;
                    foreach (Match m in AgainstTypeof.Matches(code))
                    {
                        found.Add(new ExactTypeCheck(rel, i + 1, m.Groups["t"].Value.Trim(), code));
                        hit = true;
                    }

                    if (!hit && AgainstObject.IsMatch(code))
                    {
                        found.Add(new ExactTypeCheck(rel, i + 1, "?", code));
                        hit = true;
                    }

                    if (!hit)
                        foreach (Match m in TypeLocalCompare.Matches(code))
                            if (typeLocals.Contains(m.Groups["v"].Value))
                                found.Add(new ExactTypeCheck(rel, i + 1, m.Groups["t"].Value.Trim(), code));
                }
            }

            return found;
        }
    }

    /// <summary>Writes the TSV files and <c>tables.md</c>.</summary>
    internal static class Output
    {
        /// <summary>Writes every output file.</summary>
        /// <param name="model">The model.</param>
        /// <param name="findings">The extensibility findings.</param>
        /// <param name="friends">The friend scan.</param>
        /// <param name="checks">The exact type checks.</param>
        /// <param name="options">The output folder and label.</param>
        /// <exception cref="IOException">An output file cannot be written.</exception>
        public static void Write(Model model, List<Finding> findings, FriendScan friends, List<ExactTypeCheck> checks, Options options)
        {
            var internalTypes = model.Types.Where(t => t.Reach == Reach.Internal).OrderBy(t => t.Area).ThenBy(t => t.Namespace).ThenBy(t => t.Display, StringComparer.Ordinal).ToList();
            var internalMembers = model.InternalMembers().OrderBy(m => m.Owner.Area).ThenBy(m => m.Owner.Namespace).ThenBy(m => m.Owner.Display, StringComparer.Ordinal)
                .ThenBy(m => KindOrder(m.Kind)).ThenBy(m => m.Name, StringComparer.Ordinal).ThenBy(m => m.Signature, StringComparer.Ordinal).ToList();

            Tsv(Path.Combine(options.Out, "internal-types.tsv"),
                ["area", "namespace", "type", "kind", "access", "nested_in", "file", "line", "files", "methods", "properties", "fields", "ctors", "private_members", "used_by_friends", "summary"],
                internalTypes.Select(t => new[]
                {
                    t.Area, t.Namespace, t.Display, t.Kind, t.Access, t.Def.IsNested ? Names.TypeDisplay(t.Def.DeclaringType) : "",
                    t.File, Num(t.Line), Num(t.FileCount),
                    Num(t.Members.Count(m => m.Kind is "method" or "operator" or "conversion")),
                    Num(t.Members.Count(m => m.Kind is "property" or "indexer" or "event")),
                    Num(t.Members.Count(m => m.Kind is "field" or "const") + t.EnumValues),
                    Num(t.Members.Count(m => m.Kind == "ctor")), Num(t.PrivateMembers),
                    string.Join(",", t.Friends.Concat(t.Members.SelectMany(m => m.Friends)).Distinct().Order(StringComparer.Ordinal)), t.Summary,
                }));

            Tsv(Path.Combine(options.Out, "internal-members.tsv"),
                ["area", "namespace", "type", "type_reach", "member_kind", "access", "modifiers", "signature", "file", "line", "used_by_friends", "summary"],
                internalMembers.Select(m => new[]
                {
                    m.Owner.Area, m.Owner.Namespace, m.Owner.Display, m.Owner.Reach.ToString().ToLowerInvariant(), m.Kind, m.Access, m.Modifiers,
                    m.Signature, m.File, Num(m.Line), string.Join(",", m.Friends), m.Summary,
                }));

            Tsv(Path.Combine(options.Out, "extensibility-findings.tsv"),
                ["category", "namespace", "type", "type_kind", "member_kind", "member", "detail", "file", "line"],
                findings.Select(f => new[]
                {
                    f.Category, f.Type.Namespace, f.Type.Display, f.Type.Kind, f.Member?.Kind ?? "", f.Member?.Signature ?? "", f.Detail,
                    f.Member?.File is { Length: > 0 } mf ? mf : f.Type.File, Num(f.Member?.Line > 0 ? f.Member.Line : f.Type.Line),
                }));

            var virtuality = Virtuality(model);
            Tsv(Path.Combine(options.Out, "virtuality.tsv"),
                ["namespace", "type", "kind", "visible_ctors", "visible_instance_members", "overridable", "non_virtual", "pct_overridable", "internal_members", "internal_virtual", "file", "line", "overridable_members", "internal_virtual_members"],
                virtuality.Select(v => new[]
                {
                    v.Type.Namespace, v.Type.Display, v.Type.Kind, Num(v.VisibleCtors), Num(v.Visible), Num(v.Overridable), Num(v.Visible - v.Overridable),
                    v.Visible == 0 ? "" : (100.0 * v.Overridable / v.Visible).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture),
                    Num(v.Internal), Num(v.InternalVirtual), v.Type.File, Num(v.Type.Line),
                    MemberNames(v.Type.Members.Where(m => !m.IsStatic && m.Reach != Reach.Internal && m.Overridable)),
                    MemberNames(v.Type.Members.Where(m => m.Reach == Reach.Internal && (m.Overridable || m.IsAbstract))),
                }));

            Tsv(Path.Combine(options.Out, "friend-internals-usage.tsv"),
                ["friend", "how", "namespace", "type", "type_kind", "member_kind", "member", "access", "sites", "users"],
                friends.Uses.Select(u => new[]
                {
                    u.Friend, u.How, u.Type.Namespace, u.Type.Display, u.Type.Kind, u.Member?.Kind ?? "type", u.Member?.Signature ?? "",
                    u.Member?.Access ?? u.Type.Access, Num(u.Sites), string.Join("; ", u.Users.Take(6)) + (u.Users.Count > 6 ? $"; +{u.Users.Count - 6} more" : ""),
                }));

            Tsv(Path.Combine(options.Out, "exact-type-checks.tsv"),
                ["file", "line", "compared_type", "subclass_relevant", "code"],
                checks.Select(c => new[] { c.File, Num(c.Line), c.Compared, SubclassRelevant(c, model) ? "yes" : "no", c.Text }));

            Tables.Write(model, internalTypes, internalMembers, findings, virtuality, friends, checks, options);
        }

        /// <summary>The display order of member kinds inside one type.</summary>
        /// <param name="kind">A member kind.</param>
        /// <returns>Its rank: constructors first, then methods, operators, properties, events, fields.</returns>
        public static int KindOrder(string kind) => kind switch
        {
            "ctor" => 0, "method" => 1, "operator" => 2, "conversion" => 3, "property" => 4, "indexer" => 5, "event" => 6, "field" => 7, "const" => 8, _ => 9,
        };

        /// <summary>One row of <c>virtuality.tsv</c>.</summary>
        /// <param name="Type">The visible non-sealed, non-static class.</param>
        /// <param name="VisibleCtors">Public / protected instance constructors.</param>
        /// <param name="Visible">Visible declared instance methods, properties and events.</param>
        /// <param name="Overridable">How many of those a subclass can override.</param>
        /// <param name="Internal">Declared internal members (all kinds).</param>
        /// <param name="InternalVirtual">Internal members that are virtual or abstract (override points a consumer cannot use).</param>
        public sealed record VirtualityRow(TypeInfo Type, int VisibleCtors, int Visible, int Overridable, int Internal, int InternalVirtual);

        /// <summary>Computes <see cref="VirtualityRow"/> for every visible class a consumer could subclass by declaration.</summary>
        /// <param name="model">The model.</param>
        /// <returns>The rows, largest visible surface first.</returns>
        public static List<VirtualityRow> Virtuality(Model model) => model.Types
            .Where(t => t.Reach is Reach.Public or Reach.Protected && !t.Def.IsValueType && !t.Def.IsInterface && !t.Def.IsSealed && t.Kind is not ("delegate" or "enum"))
            .Select(t => new VirtualityRow(
                t,
                t.Members.Count(m => m.Kind == "ctor" && m.Reach != Reach.Internal),
                t.Members.Count(m => !m.IsStatic && m.Reach != Reach.Internal && m.Kind is "method" or "property" or "indexer" or "event"),
                t.Members.Count(m => !m.IsStatic && m.Reach != Reach.Internal && m.Overridable),
                t.Members.Count(m => m.Reach == Reach.Internal),
                t.Members.Count(m => m.Reach == Reach.Internal && (m.Overridable || m.IsAbstract))))
            .OrderByDescending(v => v.Visible).ThenBy(v => v.Type.Display, StringComparer.Ordinal).ToList();

        /// <summary>Distinct member names, comma separated, capped at 40 (a TSV cell, not a list to read in full).</summary>
        /// <param name="members">The members.</param>
        /// <returns>The names, with <c>+N more</c> when capped.</returns>
        public static string MemberNames(IEnumerable<MemberInfo> members)
        {
            var names = members.Select(m => m.Name).Distinct(StringComparer.Ordinal).ToList();
            return string.Join(", ", names.Take(40)) + (names.Count > 40 ? $", +{names.Count - 40} more" : "");
        }

        /// <summary>
        ///     Whether an exact type test can treat a consumer's subclass differently from <c>is</c>: the compared type is a
        ///     visible, non-sealed NumSharp class (a subclass of it can exist), or the test compares two objects' classes.
        ///     A test against a CLR primitive, a sealed class or a struct means the same as <c>is</c>.
        /// </summary>
        /// <param name="check">The test.</param>
        /// <param name="model">The model (to look the compared type up).</param>
        /// <returns>True when the exactness matters to a subclass.</returns>
        public static bool SubclassRelevant(ExactTypeCheck check, Model model)
        {
            if (check.Compared == "?")
                return true;
            string name = check.Compared.StartsWith("NumSharp.", StringComparison.Ordinal) ? check.Compared["NumSharp.".Length..] : check.Compared;
            return model.Types.Any(t => t.Display == name && t.Reach is Reach.Public or Reach.Protected
                                        && !t.Def.IsSealed && !t.Def.IsValueType && !t.Def.IsInterface && t.Kind != "delegate");
        }

        /// <summary>A count as invariant text.</summary>
        /// <param name="n">The number.</param>
        /// <returns>Its digits.</returns>
        public static string Num(int n) => n.ToString(System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>Writes a TSV file; tabs and line breaks inside a cell become spaces so every row stays one line.</summary>
        /// <param name="path">The file.</param>
        /// <param name="header">The column names.</param>
        /// <param name="rows">The rows.</param>
        /// <exception cref="IOException">The file cannot be written.</exception>
        private static void Tsv(string path, string[] header, IEnumerable<string[]> rows)
        {
            var sb = new StringBuilder();
            sb.Append(string.Join('\t', header)).Append('\n');
            foreach (var row in rows)
                sb.Append(string.Join('\t', row.Select(c => Regex.Replace(c ?? "", @"[\t\r\n]+", " ")))).Append('\n');
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }
    }

    /// <summary>The Markdown summary: counts, per-area lists, findings, friend usage, exact type tests.</summary>
    internal static class Tables
    {
        /// <summary>Writes <c>tables.md</c>.</summary>
        /// <param name="model">The model.</param>
        /// <param name="internalTypes">The internal types, sorted.</param>
        /// <param name="internalMembers">The internal members, sorted.</param>
        /// <param name="findings">The findings.</param>
        /// <param name="virtuality">The virtuality rows.</param>
        /// <param name="friends">The friend scan.</param>
        /// <param name="checks">The exact type checks.</param>
        /// <param name="options">The output folder (links are made relative to it) and the label.</param>
        /// <exception cref="IOException">The file cannot be written.</exception>
        public static void Write(Model model, List<TypeInfo> internalTypes, List<MemberInfo> internalMembers, List<Finding> findings,
            List<Output.VirtualityRow> virtuality, FriendScan friends, List<ExactTypeCheck> checks, Options options)
        {
            string up = Path.GetRelativePath(options.Out, options.Repo).Replace('\\', '/');
            string Link(string file, int line)
            {
                if (file.Length == 0)
                    return "";
                string label = Path.GetFileName(file) + (line > 0 ? ":" + line : "");
                return $"[{label}]({up}/{file}{(line > 0 ? "#L" + line : "")})";
            }

            var md = new StringBuilder();
            md.AppendLine("# NumSharp.dll internal surface - generated tables");
            md.AppendLine();
            md.AppendLine($"Generated by `tools/internal-surface-map/SurfaceMap.cs` from the {options.Configuration} / {options.Tfm} build" +
                          (options.Label.Length > 0 ? $" of {options.Label}" : "") + ". Do not edit by hand: re-run the tool. The analysis is in [README.md](README.md).");
            md.AppendLine();

            // ---- Totals
            md.AppendLine("## Totals");
            md.AppendLine();
            md.AppendLine("Declared types (compiler-generated ones excluded), by effective reach:");
            md.AppendLine();
            md.AppendLine("| Kind | Public | Protected (nested) | Internal | Private (nested) | Total |");
            md.AppendLine("|---|---:|---:|---:|---:|---:|");
            foreach (var group in model.Types.GroupBy(t => KindGroup(t.Kind)).OrderBy(g => g.Key))
                md.AppendLine($"| {group.Key} | {Count(group, Reach.Public)} | {Count(group, Reach.Protected)} | {Count(group, Reach.Internal)} | {Count(group, Reach.Private)} | {group.Count()} |");
            md.AppendLine($"| **all** | **{Count(model.Types, Reach.Public)}** | **{Count(model.Types, Reach.Protected)}** | **{Count(model.Types, Reach.Internal)}** | **{Count(model.Types, Reach.Private)}** | **{model.Types.Count}** |");
            md.AppendLine();
            md.AppendLine("Internal members (non-private, non-generated), by where they are declared:");
            md.AppendLine();
            md.AppendLine("| Member kind | On a visible type (declared `internal` / `private protected`) | On an internal type | Total |");
            md.AppendLine("|---|---:|---:|---:|");
            foreach (var group in internalMembers.GroupBy(m => m.Kind).OrderBy(g => Output.KindOrder(g.Key)))
                md.AppendLine($"| {group.Key} | {group.Count(m => m.Owner.Reach != Reach.Internal)} | {group.Count(m => m.Owner.Reach == Reach.Internal)} | {group.Count()} |");
            md.AppendLine($"| **all** | **{internalMembers.Count(m => m.Owner.Reach != Reach.Internal)}** | **{internalMembers.Count(m => m.Owner.Reach == Reach.Internal)}** | **{internalMembers.Count}** |");
            md.AppendLine();

            // ---- Areas
            md.AppendLine("## Internal types by area");
            md.AppendLine();
            md.AppendLine("Area = the folder under `src/NumSharp.Core` holding most of the type's members. Members = non-private members of the internal types; on visible types = members declared internal on the visible types whose main file is in the area.");
            md.AppendLine();
            md.AppendLine("| Area | Internal types | Their members | Internal members on visible types | Named by a friend package |");
            md.AppendLine("|---|---:|---:|---:|---:|");
            var areas = model.Types.Select(t => t.Area).Distinct().OrderBy(a => a, StringComparer.Ordinal).ToList();
            foreach (var area in areas)
            {
                var types = internalTypes.Where(t => t.Area == area).ToList();
                int onVisible = internalMembers.Count(m => m.Owner.Reach != Reach.Internal && m.Owner.Area == area);
                if (types.Count == 0 && onVisible == 0)
                    continue;
                int friendNamed = types.Count(t => t.Friends.Count > 0 || t.Members.Any(m => m.Friends.Count > 0));
                md.AppendLine($"| {area} | {types.Count} | {types.Sum(t => t.Members.Count)} | {onVisible} | {friendNamed} |");
            }

            md.AppendLine();

            // ---- Visible types with the most internal members
            md.AppendLine("## Visible types with internal members");
            md.AppendLine();
            md.AppendLine("Every public (or protected nested) type that declares internal / private protected members, most first. Visible = its declared public / protected members, for scale. The members themselves are in `internal-members.tsv` (filter `type_reach` = public / protected).");
            md.AppendLine();
            md.AppendLine("| Type | Kind | Visible | Internal | ctors | methods | properties | fields | of which virtual / abstract | friend-used | Main file |");
            md.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");
            foreach (var t in model.Types.Where(t => t.Reach != Reach.Internal && t.Reach != Reach.Private)
                         .Select(t => (Type: t, Hidden: t.Members.Where(m => m.Reach == Reach.Internal).ToList()))
                         .Where(x => x.Hidden.Count > 0).OrderByDescending(x => x.Hidden.Count).ThenBy(x => x.Type.Display, StringComparer.Ordinal))
            {
                var h = t.Hidden;
                md.AppendLine($"| `{Esc(t.Type.Namespace)}.{Esc(t.Type.Display)}` | {t.Type.Kind} | {t.Type.Members.Count - h.Count} | {h.Count} | {h.Count(m => m.Kind == "ctor")} | " +
                              $"{h.Count(m => m.Kind is "method" or "operator" or "conversion")} | {h.Count(m => m.Kind is "property" or "indexer" or "event")} | " +
                              $"{h.Count(m => m.Kind is "field" or "const")} | {h.Count(m => m.Overridable || m.IsAbstract)} | {h.Count(m => m.Friends.Count > 0)} | {Link(t.Type.File, t.Type.Line)} |");
            }

            md.AppendLine();

            // ---- Findings
            md.AppendLine("## Extensibility findings");
            md.AppendLine();
            md.AppendLine("Every visible type checked for what stops a consumer's subclass or implementation (categories in `extensibility-findings.tsv`).");
            md.AppendLine();
            md.AppendLine("| Category | Findings | Types |");
            md.AppendLine("|---|---:|---:|");
            foreach (var g in findings.GroupBy(f => f.Category).OrderBy(g => CategoryOrder(g.Key)))
                md.AppendLine($"| {g.Key} | {g.Count()} | {g.Select(f => f.Type).Distinct().Count()} |");
            md.AppendLine();
            foreach (var category in new[] { "abstract-closed", "interface-closed", "ctor-closed", "hidden-hook", "hidden-hook-override", "private-protected", "narrow-accessor", "internal-interface", "internal-state", "sealed-class" })
            {
                var list = findings.Where(f => f.Category == category).ToList();
                if (list.Count == 0)
                    continue;
                md.AppendLine($"### {category} ({list.Count})");
                md.AppendLine();
                if (category == "sealed-class")
                {
                    foreach (var g in list.GroupBy(f => f.Detail).OrderBy(g => g.Key))
                        md.AppendLine($"- **{Text(g.Key)}** ({g.Count()}): " + string.Join(", ", g.Select(f => $"`{Esc(f.Type.Display)}`")));
                    md.AppendLine();
                    continue;
                }

                if (category == "internal-state")
                {
                    md.AppendLine("| Type | internal instance fields / properties |");
                    md.AppendLine("|---|---|");
                    foreach (var g in list.GroupBy(f => f.Type).OrderByDescending(g => g.Count()).ThenBy(g => g.Key.Display, StringComparer.Ordinal))
                        md.AppendLine($"| `{Esc(g.Key.Namespace)}.{Esc(g.Key.Display)}` | " + string.Join(", ", g.Select(f => $"`{Esc(f.Member!.Name)}`")) + " |");
                    md.AppendLine();
                    continue;
                }

                md.AppendLine("| Type | Member | Detail | Where |");
                md.AppendLine("|---|---|---|---|");
                foreach (var f in list)
                    md.AppendLine($"| `{Esc(f.Type.Namespace)}.{Esc(f.Type.Display)}` | {(f.Member is null ? "" : "`" + Esc(f.Member.Signature) + "`")} | {Text(f.Detail)} | " +
                                  $"{Link(f.Member?.File is { Length: > 0 } mf ? mf : f.Type.File, f.Member?.Line > 0 ? f.Member.Line : f.Type.Line)} |");
                md.AppendLine();
            }

            // ---- Virtuality
            md.AppendLine("## How much of each inheritable class can be overridden");
            md.AppendLine();
            md.AppendLine("Visible, non-sealed, non-static classes with at least 5 visible instance members (the full list is `virtuality.tsv`). Overridable = virtual, abstract or unsealed override.");
            md.AppendLine();
            md.AppendLine("| Class | Kind | Visible ctors | Visible instance members | Overridable | % | Internal members | Internal virtual / abstract | Overridable (when few) |");
            md.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|---|");
            foreach (var v in virtuality.Where(v => v.Visible >= 5))
            {
                // Name the overridable members only where they are the exception (a mostly-virtual engine would list 250).
                string few = v.Overridable is > 0 and <= 12
                    ? string.Join(", ", v.Type.Members.Where(m => !m.IsStatic && m.Reach != Reach.Internal && m.Overridable).Select(m => "`" + Esc(m.Name) + "`").Distinct())
                    : "";
                md.AppendLine($"| `{Esc(v.Type.Namespace)}.{Esc(v.Type.Display)}` | {v.Type.Kind} | {v.VisibleCtors} | {v.Visible} | {v.Overridable} | " +
                              $"{(v.Visible == 0 ? "" : (100.0 * v.Overridable / v.Visible).ToString("0", System.Globalization.CultureInfo.InvariantCulture))} | {v.Internal} | {v.InternalVirtual} | {few} |");
            }

            md.AppendLine();

            // ---- Friends
            md.AppendLine("## Internals NumSharp's own packages reach (InternalsVisibleTo)");
            md.AppendLine();
            md.AppendLine("A third-party package of the same kind cannot reach these: each row is an internal type or member the friend's IL names. Friends without a `src/` build output are listed as not scanned.");
            md.AppendLine();
            md.AppendLine("| Friend | Internal members named | Internal types named | Sites |");
            md.AppendLine("|---|---:|---:|---:|");
            foreach (var (name, _) in friends.Scanned)
            {
                var uses = friends.Uses.Where(u => u.Friend == name).ToList();
                md.AppendLine($"| {name} | {uses.Where(u => u.Member is not null).Select(u => u.Member).Distinct().Count()} | {uses.Select(u => u.Type).Distinct().Count()} | {uses.Sum(u => u.Sites)} |");
            }

            if (friends.NotScanned.Count > 0)
                md.AppendLine($"\nNot scanned (no build output under `src/`): {string.Join(", ", friends.NotScanned.Select(n => "`" + n + "`"))}.");
            md.AppendLine();
            foreach (var (name, _) in friends.Scanned)
            {
                var uses = friends.Uses.Where(u => u.Friend == name).ToList();
                if (uses.Count == 0)
                    continue;
                md.AppendLine($"### {name}");
                md.AppendLine();
                md.AppendLine("| How | Type | Member | Access | Sites | Where declared |");
                md.AppendLine("|---|---|---|---|---:|---|");
                foreach (var u in uses)
                    md.AppendLine($"| {u.How} | `{Esc(u.Type.Namespace)}.{Esc(u.Type.Display)}` | {(u.Member is null ? "(the type)" : "`" + Esc(u.Member.Signature) + "`")} | " +
                                  $"{u.Member?.Access ?? u.Type.Access} | {u.Sites} | {Link(u.Member?.File is { Length: > 0 } f ? f : u.Type.File, u.Member?.Line > 0 ? u.Member.Line : u.Type.Line)} |");
                md.AppendLine();
            }

            // ---- Exact type checks
            md.AppendLine("## Exact type tests in Core's source");
            md.AppendLine();
            md.AppendLine("`GetType() == typeof(X)` (and its variants) answers \"is this precisely X\": a subclass of X, a consumer's included, takes the other branch. Only a test against a visible non-sealed NumSharp class, or between two objects' classes, can treat a subclass differently from `is`; those are listed in full. Every site is in `exact-type-checks.tsv`.");
            md.AppendLine();
            var relevant = checks.Where(c => Output.SubclassRelevant(c, model)).ToList();
            md.AppendLine("| Compared type | Code | Where |");
            md.AppendLine("|---|---|---|");
            foreach (var c in relevant)
                md.AppendLine($"| `{Esc(c.Compared)}` | `{Esc(c.Text)}` | {Link(c.File, c.Line)} |");
            md.AppendLine();
            var other = checks.Where(c => !Output.SubclassRelevant(c, model)).ToList();
            md.AppendLine($"The other {other.Count} tests compare with CLR primitives, arrays, sealed types or generic parameters " +
                          $"(element-type dispatch, where exact and `is` agree): " +
                          string.Join(", ", other.GroupBy(c => c.Compared).OrderByDescending(g => g.Count()).ThenBy(g => g.Key, StringComparer.Ordinal)
                              .Select(g => $"`{Esc(g.Key)}` {g.Count()}")) + ".");
            md.AppendLine();

            // ---- Per-area type lists
            md.AppendLine("## Every internal type, by area");
            md.AppendLine();
            md.AppendLine("Members = non-private members (methods / properties / fields / ctors); the member rows are in `internal-members.tsv`.");
            md.AppendLine();
            foreach (var area in internalTypes.Select(t => t.Area).Distinct())
            {
                var types = internalTypes.Where(t => t.Area == area).ToList();
                md.AppendLine($"### {area} ({types.Count})");
                md.AppendLine();
                md.AppendLine("| Type | Kind | Members (m/p/f/c) | Friend-used | Declared | Summary |");
                md.AppendLine("|---|---|---|---|---|---|");
                foreach (var t in types)
                {
                    int methods = t.Members.Count(m => m.Kind is "method" or "operator" or "conversion");
                    int props = t.Members.Count(m => m.Kind is "property" or "indexer" or "event");
                    int fields = t.Members.Count(m => m.Kind is "field" or "const") + t.EnumValues;
                    int ctors = t.Members.Count(m => m.Kind == "ctor");
                    var used = t.Friends.Concat(t.Members.SelectMany(m => m.Friends)).Distinct().Select(ShortFriend).Order(StringComparer.Ordinal);
                    md.AppendLine($"| `{Esc(t.Namespace)}.{Esc(t.Display)}` | {t.Kind} | {methods}/{props}/{fields}/{ctors} | {string.Join(", ", used)} | {Link(t.File, t.Line)} | {Text(t.Summary)} |");
                }

                md.AppendLine();
            }

            File.WriteAllText(Path.Combine(options.Out, "tables.md"), md.ToString().Replace("\r\n", "\n"), new UTF8Encoding(false));
        }

        /// <summary>A friend assembly name without the shared <c>NumSharp.Interop.</c> prefix (table width).</summary>
        /// <param name="name">The assembly name.</param>
        /// <returns>The short name.</returns>
        private static string ShortFriend(string name) => name.StartsWith("NumSharp.Interop.", StringComparison.Ordinal) ? name["NumSharp.Interop.".Length..] : name;

        /// <summary>The order findings categories are listed in: the hard blockers first.</summary>
        /// <param name="category">The category.</param>
        /// <returns>Its rank.</returns>
        private static int CategoryOrder(string category) => category switch
        {
            "abstract-closed" => 0, "interface-closed" => 1, "ctor-closed" => 2, "hidden-hook" => 3, "hidden-hook-override" => 4,
            "private-protected" => 5, "narrow-accessor" => 6, "internal-interface" => 7, "internal-state" => 8,
            "sealed-class" => 9, "static-class" => 10, "value-type" => 11, _ => 12,
        };

        /// <summary>A type kind folded into the totals table's rows.</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>class (incl. abstract / sealed / record), static class, struct (all flavours), interface, enum, delegate.</returns>
        private static string KindGroup(string kind) => kind switch
        {
            "static class" => "static class",
            "interface" or "enum" or "delegate" => kind,
            _ when kind.EndsWith("struct", StringComparison.Ordinal) => "struct",
            _ => "class",
        };

        /// <summary>How many types of a group have the reach.</summary>
        /// <param name="types">The group.</param>
        /// <param name="reach">The reach.</param>
        /// <returns>The count.</returns>
        private static int Count(IEnumerable<TypeInfo> types, Reach reach) => types.Count(t => t.Reach == reach);

        /// <summary>Escapes text that goes inside a code span of a table cell: only the pipe, which would end the cell.</summary>
        /// <param name="s">The cell text.</param>
        /// <returns>The escaped text.</returns>
        private static string Esc(string s) => s.Replace("|", "\\|");

        /// <summary>
        ///     Escapes prose for a table cell: the pipe, and angle brackets, which GitHub would otherwise read as HTML tags
        ///     and drop (<c>NDArray&lt;T&gt;</c> in a summary would lose its <c>&lt;T&gt;</c>).
        /// </summary>
        /// <param name="s">The cell text.</param>
        /// <returns>The escaped text.</returns>
        private static string Text(string s) => s.Replace("|", "\\|").Replace("<", "&lt;").Replace(">", "&gt;");
    }
}
