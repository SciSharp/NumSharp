using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Examples.MaterialLab.App.Native;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>
    /// A linked GLSL program (vertex + fragment) with a uniform-location cache. Compile and link errors are
    /// turned into exceptions carrying the driver's info log AND the offending source with line numbers,
    /// because a GLSL typo otherwise surfaces as a silent black screen.
    /// </summary>
    internal sealed unsafe class GlProgram : IDisposable
    {
        /// <summary>Uniform locations by name, looked up once (a GL query per frame per uniform is avoidable driver overhead).</summary>
        private readonly Dictionary<string, int> _uniforms = new Dictionary<string, int>(StringComparer.Ordinal);

        /// <summary>The GL program object id.</summary>
        public uint Id { get; private set; }

        /// <summary>A diagnostic name used in error messages.</summary>
        public string Name { get; }

        /// <summary>
        /// Compiles and links a program. Attribute slots are pinned BEFORE linking (via
        /// <paramref name="attributes"/>) so the VAO layouts written in C# can rely on fixed indices.
        /// </summary>
        /// <param name="name">Diagnostic name.</param>
        /// <param name="vertexSource">Vertex shader GLSL (must start with a <c>#version</c> line).</param>
        /// <param name="fragmentSource">Fragment shader GLSL.</param>
        /// <param name="attributes">Vertex attribute names in slot order (slot = array index), or null.</param>
        /// <exception cref="InvalidOperationException">Compilation or linking failed; the message contains the info log and numbered source.</exception>
        public GlProgram(string name, string vertexSource, string fragmentSource, params string[] attributes)
        {
            Name = name;
            uint vs = Compile(Gl.VERTEX_SHADER, vertexSource, name + ".vert");
            uint fs = Compile(Gl.FRAGMENT_SHADER, fragmentSource, name + ".frag");
            Id = Gl.CreateProgram();
            Gl.AttachShader(Id, vs);
            Gl.AttachShader(Id, fs);
            if (attributes != null)
                for (int i = 0; i < attributes.Length; i++)
                {
                    byte[] ascii = AsciiZ(attributes[i]);
                    fixed (byte* p = ascii) Gl.BindAttribLocation(Id, (uint)i, p);
                }
            Gl.LinkProgram(Id);
            int ok;
            Gl.GetProgramiv(Id, Gl.LINK_STATUS, &ok);
            // Shaders can be deleted once linked; the program keeps the compiled code alive.
            Gl.DeleteShader(vs);
            Gl.DeleteShader(fs);
            if (ok == 0)
            {
                string log = ProgramLog(Id);
                Gl.DeleteProgram(Id);
                Id = 0;
                throw new InvalidOperationException($"Linking GL program '{name}' failed:\n{log}");
            }
        }

        /// <summary>Makes this program current for subsequent draws and uniform writes.</summary>
        public void Use() => Gl.UseProgram(Id);

        /// <summary>Cached uniform lookup (−1 when the uniform is absent or optimized out — writes to −1 are ignored by GL).</summary>
        /// <param name="name">Uniform name.</param>
        /// <returns>The location.</returns>
        public int U(string name)
        {
            if (!_uniforms.TryGetValue(name, out int loc))
            {
                loc = Gl.Uniform(Id, name);
                _uniforms[name] = loc;
            }
            return loc;
        }

        /// <summary>Sets an int (or sampler unit) uniform on the CURRENT program.</summary>
        /// <param name="name">Uniform name.</param>
        /// <param name="v">Value.</param>
        public void Set(string name, int v) => Gl.Uniform1i(U(name), v);
        /// <summary>Sets a float uniform on the current program.</summary>
        /// <param name="name">Uniform name.</param>
        /// <param name="v">Value.</param>
        public void Set(string name, float v) => Gl.Uniform1f(U(name), v);
        /// <summary>Sets a vec2 uniform on the current program.</summary>
        /// <param name="name">Uniform name.</param>
        /// <param name="x">X.</param>
        /// <param name="y">Y.</param>
        public void Set(string name, float x, float y) => Gl.Uniform2f(U(name), x, y);
        /// <summary>Sets a vec3 uniform on the current program.</summary>
        /// <param name="name">Uniform name.</param>
        /// <param name="x">X.</param>
        /// <param name="y">Y.</param>
        /// <param name="z">Z.</param>
        public void Set(string name, float x, float y, float z) => Gl.Uniform3f(U(name), x, y, z);
        /// <summary>Sets a vec4 uniform on the current program.</summary>
        /// <param name="name">Uniform name.</param>
        /// <param name="x">X.</param>
        /// <param name="y">Y.</param>
        /// <param name="z">Z.</param>
        /// <param name="w">W.</param>
        public void Set(string name, float x, float y, float z, float w) => Gl.Uniform4f(U(name), x, y, z, w);

        /// <summary>Uploads a vec3 array uniform (e.g. the per-material color table).</summary>
        /// <param name="name">Uniform name (without <c>[0]</c>).</param>
        /// <param name="values">Packed xyz triples.</param>
        public void SetVec3Array(string name, ReadOnlySpan<float> values)
        {
            fixed (float* p = values) Gl.Uniform3fv(U(name), values.Length / 3, p);
        }

        /// <summary>Uploads a vec4 array uniform.</summary>
        /// <param name="name">Uniform name (without <c>[0]</c>).</param>
        /// <param name="values">Packed xyzw quadruples.</param>
        public void SetVec4Array(string name, ReadOnlySpan<float> values)
        {
            fixed (float* p = values) Gl.Uniform4fv(U(name), values.Length / 4, p);
        }

        /// <summary>Compiles one shader stage.</summary>
        /// <param name="stage">VERTEX_SHADER or FRAGMENT_SHADER.</param>
        /// <param name="source">GLSL source.</param>
        /// <param name="label">Diagnostic label.</param>
        /// <returns>The shader id.</returns>
        /// <exception cref="InvalidOperationException">Compilation failed (message has the log + numbered source).</exception>
        private static uint Compile(uint stage, string source, string label)
        {
            uint s = Gl.CreateShader(stage);
            byte[] bytes = Encoding.ASCII.GetBytes(source);
            fixed (byte* p = bytes)
            {
                byte* ptr = p;
                int len = bytes.Length;
                Gl.ShaderSource(s, 1, &ptr, &len);
            }
            Gl.CompileShader(s);
            int ok;
            Gl.GetShaderiv(s, Gl.COMPILE_STATUS, &ok);
            if (ok == 0)
            {
                int logLen;
                Gl.GetShaderiv(s, Gl.INFO_LOG_LENGTH, &logLen);
                var buf = new byte[Math.Max(1, logLen)];
                fixed (byte* b = buf) Gl.GetShaderInfoLog(s, buf.Length, null, b);
                Gl.DeleteShader(s);
                throw new InvalidOperationException($"Compiling {label} failed:\n{Encoding.ASCII.GetString(buf).TrimEnd('\0')}\n{Numbered(source)}");
            }
            return s;
        }

        /// <summary>Reads a program's link log.</summary>
        /// <param name="program">Program id.</param>
        /// <returns>The log text.</returns>
        private static string ProgramLog(uint program)
        {
            int logLen;
            Gl.GetProgramiv(program, Gl.INFO_LOG_LENGTH, &logLen);
            var buf = new byte[Math.Max(1, logLen)];
            fixed (byte* b = buf) Gl.GetProgramInfoLog(program, buf.Length, null, b);
            return Encoding.ASCII.GetString(buf).TrimEnd('\0');
        }

        /// <summary>Prefixes each source line with its number so driver messages ("0(42) : error …") can be matched by eye.</summary>
        /// <param name="source">GLSL source.</param>
        /// <returns>Numbered text.</returns>
        private static string Numbered(string source)
        {
            var sb = new StringBuilder();
            var lines = source.Split('\n');
            for (int i = 0; i < lines.Length; i++) sb.Append((i + 1).ToString().PadLeft(4)).Append("  ").Append(lines[i]).Append('\n');
            return sb.ToString();
        }

        /// <summary>A NUL-terminated ASCII copy of <paramref name="s"/> (GL takes C strings).</summary>
        /// <param name="s">String.</param>
        /// <returns>The bytes including the terminator.</returns>
        private static byte[] AsciiZ(string s)
        {
            var buf = new byte[s.Length + 1];
            for (int i = 0; i < s.Length; i++) buf[i] = (byte)s[i];
            return buf;
        }

        /// <summary>Deletes the program.</summary>
        public void Dispose()
        {
            if (Id != 0) { Gl.DeleteProgram(Id); Id = 0; }
        }
    }
}
