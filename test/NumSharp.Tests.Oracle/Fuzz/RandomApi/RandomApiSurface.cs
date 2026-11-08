using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     The reflected public surface of NumSharp's random world — every public constructor, method overload, property
    ///     accessor, field and operator of the random types — with ONE canonical signature string per member. The signature
    ///     is the join key of the whole random-API oracle: <c>test/oracle/gen_random_oracle.py</c> names the exact overload
    ///     each case exercises by it, the replay harness resolves it back to the member by reflection and invokes exactly
    ///     that member, and the coverage gates compare the corpus against this inventory.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The inventory is committed as <c>test/oracle/random_surface.json</c> so the Python generator can check its binding
    ///     table against it before writing a corpus; <see cref="RandomApiSurfaceTests"/> fails when the committed file no
    ///     longer equals the reflection (a member added, removed or re-signatured without regenerating), which is what keeps
    ///     "every overload" honest when the surface changes.
    ///     </para>
    ///     <para>
    ///     Canonical signature grammar (stable; the Python side parses nothing, it only compares strings):
    ///     <list type="bullet">
    ///         <item>method: <c>Type.name(p1,p2)</c>, generic methods <c>Type.name&lt;T&gt;(...)</c>, a <c>params</c> array
    ///             as <c>params long[]</c>;</item>
    ///         <item>constructor: <c>Type.ctor(p1,p2)</c>;</item>
    ///         <item>property accessors: <c>Type.name.get</c> / <c>Type.name.set</c>;</item>
    ///         <item>field: <c>Type.name.field</c>; operator: <c>Type.op_Implicit(p1)</c>;</item>
    ///         <item>types by C# keyword for primitives, by simple name otherwise, nested types as <c>Outer.Inner</c>,
    ///             nullable as <c>T?</c>, arrays as <c>T[]</c> / <c>T[,]</c>, generics as <c>Name&lt;A,B&gt;</c>.</item>
    ///     </list>
    ///     Members are listed per DECLARING type (<see cref="BindingFlags.DeclaredOnly"/>): an inherited member such as
    ///     <c>BitGenerator.random_raw</c> appears once, under its declaring type, and the engine-coverage gate is what
    ///     demands it on every engine.
    ///     </para>
    /// </remarks>
    internal static class RandomApiSurface
    {
        /// <summary>
        ///     The random world: every public type the oracle must cover, in a fixed order (the JSON lists members sorted
        ///     by signature, so this order only documents the scope).
        /// </summary>
        internal static readonly Type[] Types =
        {
            typeof(NumPyRandom), typeof(NumPyRandom.State), typeof(NativeRandomState), typeof(Generator),
            typeof(BitGenerator), typeof(BitGeneratorState),
            typeof(PCG64), typeof(PCG64.State), typeof(PCG64DXSM), typeof(PCG64DXSM.State),
            typeof(Philox), typeof(Philox.State), typeof(SFC64), typeof(SFC64.State),
            typeof(MT19937), typeof(MT19937.State),
            typeof(SeedSequence), typeof(SeedlessSeedSequence), typeof(ISeedSequence), typeof(ISpawnableSeedSequence),
        };

        /// <summary>The committed inventory's path relative to the repository root.</summary>
        internal const string CommittedRelativePath = "test/oracle/random_surface.json";

        /// <summary>
        ///     The environment variable that turns <see cref="RandomApiSurfaceTests"/> into the inventory's writer: set to
        ///     <c>1</c>, the test rewrites the committed file from reflection instead of comparing against it.
        /// </summary>
        internal const string WriteVariable = "NUMSHARP_WRITE_RANDOM_SURFACE";

        /// <summary>One parameter of a member, as the parameter-coverage gate and the Python generator read it.</summary>
        /// <param name="Name">The parameter name (the corpus names arguments by it).</param>
        /// <param name="Type">The canonical type name (see the class remarks).</param>
        /// <param name="Optional">Whether the parameter has a default (a call may omit it).</param>
        /// <param name="Default">The default as canonical text (<c>null</c>, <c>default</c>, a number, a quoted string),
        ///     or null when the parameter is required.</param>
        /// <param name="IsParams">Whether the parameter is a <c>params</c> array.</param>
        internal sealed record Parameter(string Name, string Type, bool Optional, string Default, bool IsParams);

        /// <summary>One public member of the random world.</summary>
        /// <param name="Sig">The canonical signature (the join key).</param>
        /// <param name="DeclaringType">The canonical name of the declaring type (<c>PCG64.State</c>).</param>
        /// <param name="Name">The member name (<c>normal</c>, <c>ctor</c>, the property / field name).</param>
        /// <param name="Kind"><c>method</c>, <c>ctor</c>, <c>get</c>, <c>set</c>, <c>field</c> or <c>operator</c>.</param>
        /// <param name="IsStatic">Whether the member is static (constant fields, operators).</param>
        /// <param name="Returns">The canonical return / property / field type (<c>void</c> for setters and ctors).</param>
        /// <param name="Generic">The generic method parameters (<c>T</c>), empty when not generic.</param>
        /// <param name="Parameters">The parameters in declaration order.</param>
        internal sealed record Member(string Sig, string DeclaringType, string Name, string Kind, bool IsStatic, string Returns,
            IReadOnlyList<string> Generic, IReadOnlyList<Parameter> Parameters)
        {
            /// <summary>The reflected member itself (not serialized) — what the harness invokes.</summary>
            internal MemberInfo Info { get; init; }
        }

        /// <summary>
        ///     Reflects the whole random surface: every public constructor, method (each overload), property accessor,
        ///     field and operator declared by <see cref="Types"/>.
        /// </summary>
        /// <returns>The members, sorted by signature (ordinal), each carrying its reflected <see cref="MemberInfo"/>.</returns>
        /// <exception cref="InvalidOperationException">Two members canonicalize to the same signature — the grammar would
        ///     then be unable to address one of them, so this refuses rather than silently dropping it.</exception>
        internal static IReadOnlyList<Member> Reflect()
        {
            const BindingFlags declared = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
            var members = new List<Member>();
            foreach (var t in Types)
            {
                string owner = TypeName(t);
                foreach (var c in t.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                    members.Add(new Member($"{owner}.ctor({ParamList(c)})", owner, "ctor", "ctor", false, "void",
                        Array.Empty<string>(), Params(c)) { Info = c });
                foreach (var m in t.GetMethods(declared))
                {
                    // Property accessors are listed through their properties below; operators keep their op_ name.
                    if (m.IsSpecialName && !m.Name.StartsWith("op_", StringComparison.Ordinal))
                        continue;
                    var generic = m.IsGenericMethodDefinition ? m.GetGenericArguments().Select(TypeName).ToArray() : Array.Empty<string>();
                    string gen = generic.Length > 0 ? "<" + string.Join(",", generic) + ">" : "";
                    string kind = m.IsSpecialName ? "operator" : "method";
                    members.Add(new Member($"{owner}.{m.Name}{gen}({ParamList(m)})", owner, m.Name, kind, m.IsStatic,
                        TypeName(m.ReturnType), generic, Params(m)) { Info = m });
                }
                foreach (var p in t.GetProperties(declared))
                {
                    if (p.GetIndexParameters().Length > 0)
                        throw new InvalidOperationException($"{owner}.{p.Name}: indexers are outside the canonical grammar.");
                    if (p.GetMethod is { IsPublic: true } g)
                        members.Add(new Member($"{owner}.{p.Name}.get", owner, p.Name, "get", g.IsStatic, TypeName(p.PropertyType),
                            Array.Empty<string>(), Array.Empty<Parameter>()) { Info = p });
                    if (p.SetMethod is { IsPublic: true } s)
                        members.Add(new Member($"{owner}.{p.Name}.set", owner, p.Name, "set", s.IsStatic, "void",
                            Array.Empty<string>(), new[] { new Parameter("value", TypeName(p.PropertyType), false, null, false) }) { Info = p });
                }
                foreach (var f in t.GetFields(declared))
                    members.Add(new Member($"{owner}.{f.Name}.field", owner, f.Name, "field", f.IsStatic, TypeName(f.FieldType),
                        Array.Empty<string>(), Array.Empty<Parameter>()) { Info = f });
            }

            var duplicate = members.GroupBy(m => m.Sig, StringComparer.Ordinal).FirstOrDefault(g => g.Count() > 1);
            if (duplicate != null)
                throw new InvalidOperationException($"Two random members canonicalize to '{duplicate.Key}'.");
            members.Sort((a, b) => string.CompareOrdinal(a.Sig, b.Sig));
            return members;
        }

        /// <summary>
        ///     The canonical name of a type (see the class remarks): C# keywords for primitives, simple names otherwise,
        ///     <c>Outer.Inner</c> for nested types, <c>T?</c>, <c>T[]</c> / <c>T[,]</c>, <c>Name&lt;A,B&gt;</c>.
        /// </summary>
        /// <param name="t">The type (a generic method parameter prints as its name).</param>
        /// <returns>The canonical name.</returns>
        internal static string TypeName(Type t)
        {
            if (t.IsByRef)
                return "ref " + TypeName(t.GetElementType()!);
            if (t.IsArray)
                return TypeName(t.GetElementType()!) + "[" + new string(',', t.GetArrayRank() - 1) + "]";
            if (t.IsGenericParameter)
                return t.Name;
            if (t.IsGenericType)
            {
                if (t.GetGenericTypeDefinition() == typeof(Nullable<>))
                    return TypeName(t.GetGenericArguments()[0]) + "?";
                string bare = t.Name.Split('`')[0];
                string outerG = t.IsNested && !t.DeclaringType!.IsGenericType ? TypeName(t.DeclaringType) + "." : "";
                return outerG + bare + "<" + string.Join(",", t.GetGenericArguments().Select(TypeName)) + ">";
            }
            string keyword = Type.GetTypeCode(t) switch
            {
                TypeCode.Boolean => "bool", TypeCode.Byte => "byte", TypeCode.SByte => "sbyte", TypeCode.Int16 => "short",
                TypeCode.UInt16 => "ushort", TypeCode.Int32 => "int", TypeCode.UInt32 => "uint", TypeCode.Int64 => "long",
                TypeCode.UInt64 => "ulong", TypeCode.Single => "float", TypeCode.Double => "double",
                TypeCode.Decimal => "decimal", TypeCode.Char => "char", TypeCode.String => "string",
                _ => null,
            };
            if (keyword != null && !t.IsEnum)
                return keyword;
            if (t == typeof(object))
                return "object";
            if (t == typeof(void))
                return "void";
            return t.IsNested ? TypeName(t.DeclaringType!) + "." + t.Name : t.Name;
        }

        /// <summary>The canonical parameter list of a method or constructor (<c>params</c> arrays marked).</summary>
        /// <param name="m">The method or constructor.</param>
        /// <returns>The comma-separated canonical parameter types.</returns>
        private static string ParamList(MethodBase m)
            => string.Join(",", m.GetParameters().Select(p => (IsParamsArray(p) ? "params " : "") + TypeName(p.ParameterType)));

        /// <summary>Whether a parameter is a <c>params</c> array.</summary>
        /// <param name="p">The parameter.</param>
        /// <returns>True when it carries <see cref="ParamArrayAttribute"/>.</returns>
        private static bool IsParamsArray(ParameterInfo p) => p.GetCustomAttribute<ParamArrayAttribute>() != null;

        /// <summary>The parameters of a method or constructor as <see cref="Parameter"/> records.</summary>
        /// <param name="m">The method or constructor.</param>
        /// <returns>The parameters in declaration order.</returns>
        private static Parameter[] Params(MethodBase m)
            => m.GetParameters().Select(p => new Parameter(p.Name, TypeName(p.ParameterType), p.HasDefaultValue,
                p.HasDefaultValue ? DefaultText(p) : null, IsParamsArray(p))).ToArray();

        /// <summary>
        ///     The canonical text of a parameter default: <c>null</c> for a null reference, <c>default</c> for a defaulted
        ///     struct (reflection reports it as null), <c>true</c>/<c>false</c>, a round-trip invariant number, or a quoted
        ///     string.
        /// </summary>
        /// <param name="p">A parameter that has a default.</param>
        /// <returns>The default's canonical text.</returns>
        private static string DefaultText(ParameterInfo p)
        {
            object v = p.DefaultValue;
            if (v == null)
            {
                var t = p.ParameterType;
                return t.IsValueType && Nullable.GetUnderlyingType(t) == null ? "default" : "null";
            }
            return v switch
            {
                bool b => b ? "true" : "false",
                string s => "\"" + s + "\"",
                double d => d.ToString("R", CultureInfo.InvariantCulture),
                float f => f.ToString("R", CultureInfo.InvariantCulture),
                IFormattable n => n.ToString(null, CultureInfo.InvariantCulture),
                _ => v.ToString(),
            };
        }

        /// <summary>
        ///     Serializes an inventory to the committed JSON form: stable key order, members sorted by signature, LF line
        ///     endings, one member per line (so a surface change shows as a readable one-line diff).
        /// </summary>
        /// <param name="members">The inventory (from <see cref="Reflect"/>).</param>
        /// <returns>The JSON text, ending in a newline.</returns>
        internal static string ToJson(IReadOnlyList<Member> members)
        {
            var sb = new StringBuilder();
            sb.Append("{\n  \"schema\": 1,\n");
            sb.Append("  \"note\": \"Reflected public surface of NumSharp's random world (Fuzz/RandomApi/RandomApiSurface.cs). ");
            sb.Append("Regenerate: set ").Append(WriteVariable).Append("=1 and run the RandomApiSurfaceTests FuzzMatrix test.\",\n");
            sb.Append("  \"types\": [").Append(string.Join(", ", Types.Select(t => JsonSerializer.Serialize(TypeName(t))))).Append("],\n");
            sb.Append("  \"members\": [\n");
            for (int i = 0; i < members.Count; i++)
            {
                var m = members[i];
                sb.Append("    {\"sig\": ").Append(JsonSerializer.Serialize(m.Sig))
                  .Append(", \"type\": ").Append(JsonSerializer.Serialize(m.DeclaringType))
                  .Append(", \"name\": ").Append(JsonSerializer.Serialize(m.Name))
                  .Append(", \"kind\": ").Append(JsonSerializer.Serialize(m.Kind))
                  .Append(", \"static\": ").Append(m.IsStatic ? "true" : "false")
                  .Append(", \"returns\": ").Append(JsonSerializer.Serialize(m.Returns))
                  .Append(", \"generic\": [").Append(string.Join(", ", m.Generic.Select(g => JsonSerializer.Serialize(g)))).Append(']')
                  .Append(", \"params\": [");
                for (int j = 0; j < m.Parameters.Count; j++)
                {
                    var p = m.Parameters[j];
                    if (j > 0)
                        sb.Append(", ");
                    sb.Append("{\"name\": ").Append(JsonSerializer.Serialize(p.Name))
                      .Append(", \"type\": ").Append(JsonSerializer.Serialize(p.Type))
                      .Append(", \"optional\": ").Append(p.Optional ? "true" : "false");
                    if (p.Optional)
                        sb.Append(", \"default\": ").Append(JsonSerializer.Serialize(p.Default));
                    if (p.IsParams)
                        sb.Append(", \"params\": true");
                    sb.Append('}');
                }
                sb.Append("]}").Append(i + 1 < members.Count ? ",\n" : "\n");
            }
            sb.Append("  ]\n}\n");
            return sb.ToString();
        }

        /// <summary>
        ///     The repository root, found by walking up from the test output directory to the folder that holds
        ///     <c>test/oracle/gen_oracle.py</c>.
        /// </summary>
        /// <returns>The root path.</returns>
        /// <exception cref="DirectoryNotFoundException">The tests run outside a checkout (no ancestor holds the oracle
        ///     generator), so neither the committed inventory nor its regeneration target can be found.</exception>
        internal static string RepoRoot()
        {
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "test", "oracle", "gen_oracle.py")))
                    return dir.FullName;
            throw new DirectoryNotFoundException(
                $"No ancestor of {AppContext.BaseDirectory} contains test/oracle/gen_oracle.py; the random surface inventory needs a checkout.");
        }

        /// <summary>The committed inventory's absolute path in this checkout.</summary>
        /// <returns>The path of <c>test/oracle/random_surface.json</c>.</returns>
        /// <exception cref="DirectoryNotFoundException">See <see cref="RepoRoot"/>.</exception>
        internal static string CommittedPath() => Path.Combine(RepoRoot(), "test", "oracle", "random_surface.json");

        /// <summary>
        ///     Resolves a canonical signature to its reflected member — what the replay harness invokes. Built once per
        ///     process from <see cref="Reflect"/>.
        /// </summary>
        /// <param name="sig">The canonical signature a corpus case names.</param>
        /// <returns>The member.</returns>
        /// <exception cref="KeyNotFoundException">No public random member has that signature: the corpus names an overload
        ///     that was removed or re-signatured (regenerate the corpus), or the generator's binding table has a typo.</exception>
        internal static Member Resolve(string sig)
            => BySig.Value.TryGetValue(sig, out var m)
                ? m
                : throw new KeyNotFoundException($"No public random member has the signature '{sig}'.");

        /// <summary>The signature index, reflected lazily once per process (thread-safe).</summary>
        private static readonly Lazy<Dictionary<string, Member>> BySig =
            new(() => Reflect().ToDictionary(m => m.Sig, StringComparer.Ordinal));
    }
}
