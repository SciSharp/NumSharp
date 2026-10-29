using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp.Examples.MaterialLab.App.Native
{
    /// <summary>
    /// A hand-rolled OpenGL 3.3 core binding: the constants and entry points the renderer uses, loaded as
    /// C# 9 unmanaged function pointers once a context is current. Calling a function pointer is a
    /// direct <c>call</c> — no delegate, no marshalling stub — so the per-frame GL traffic (a few dozen
    /// calls) costs nothing measurable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why two loaders.</b> <c>wglGetProcAddress</c> returns only extension/post-1.1 entry points;
    /// for the OpenGL 1.1 functions that opengl32.dll exports directly it returns null (or the sentinel
    /// values 1, 2, 3, −1 on some drivers). <see cref="Load"/> falls back to <c>GetProcAddress</c> on
    /// opengl32.dll for those, which is the documented way to load both halves.
    /// </para>
    /// <para>
    /// <b>Footgun:</b> every pointer here is only valid while the context it was loaded under exists and
    /// is current on the calling thread. The app creates exactly one context and never switches, so the
    /// pointers are loaded once in <see cref="LoadAll"/>.
    /// </para>
    /// </remarks>
    internal static unsafe class Gl
    {
        // ---------------- constants ----------------

        /// <summary><see cref="Clear"/> mask bit for the color buffer.</summary>
        public const uint COLOR_BUFFER_BIT = 0x4000;
        /// <summary>Capability: blending (additive splats, alpha-blended HUD).</summary>
        public const uint BLEND = 0x0BE2;
        /// <summary>Capability: scissor test.</summary>
        public const uint SCISSOR_TEST = 0x0C11;
        /// <summary>Blend factor 0.</summary>
        public const uint ZERO = 0;
        /// <summary>Blend factor 1 (additive accumulation).</summary>
        public const uint ONE = 1;
        /// <summary>Blend factor: source alpha.</summary>
        public const uint SRC_ALPHA = 0x0302;
        /// <summary>Blend factor: 1 − source alpha (standard "over" compositing).</summary>
        public const uint ONE_MINUS_SRC_ALPHA = 0x0303;
        /// <summary>Blend equation: add (the default).</summary>
        public const uint FUNC_ADD = 0x8006;
        /// <summary>Blend equation: per-channel max (used to keep the fastest particle's speed in a pixel).</summary>
        public const uint MAX = 0x8008;

        /// <summary>Texture target: a 2-D texture.</summary>
        public const uint TEXTURE_2D = 0x0DE1;
        /// <summary>Texture parameter: minification filter.</summary>
        public const uint TEXTURE_MIN_FILTER = 0x2801;
        /// <summary>Texture parameter: magnification filter.</summary>
        public const uint TEXTURE_MAG_FILTER = 0x2800;
        /// <summary>Texture parameter: horizontal wrap mode.</summary>
        public const uint TEXTURE_WRAP_S = 0x2802;
        /// <summary>Texture parameter: vertical wrap mode.</summary>
        public const uint TEXTURE_WRAP_T = 0x2803;
        /// <summary>Filter: bilinear (smooth fields upsample without blocks).</summary>
        public const int LINEAR = 0x2601;
        /// <summary>Filter: nearest texel (crisp glyphs / exact lookups).</summary>
        public const int NEAREST = 0x2600;
        /// <summary>Wrap: clamp to the edge texel (no bleeding from the opposite side).</summary>
        public const int CLAMP_TO_EDGE = 0x812F;
        /// <summary>Texture unit 0; unit n is <c>TEXTURE0 + n</c>.</summary>
        public const uint TEXTURE0 = 0x84C0;

        /// <summary>Client pixel format: RGBA.</summary>
        public const uint RGBA = 0x1908;
        /// <summary>Client pixel format: BGRA (GDI's DIB byte order — uploads the font atlas without swizzling).</summary>
        public const uint BGRA = 0x80E1;
        /// <summary>Client pixel format: single red channel.</summary>
        public const uint RED = 0x1903;
        /// <summary>Internal format: 8-bit RGBA.</summary>
        public const int RGBA8 = 0x8058;
        /// <summary>Internal format: half-float RGBA — HDR accumulation that does not clip when many splats overlap.</summary>
        public const int RGBA16F = 0x881A;
        /// <summary>Internal format: 8-bit single channel (the obstacle mask).</summary>
        public const int R8 = 0x8229;
        /// <summary>Component type: unsigned byte.</summary>
        public const uint UNSIGNED_BYTE = 0x1401;
        /// <summary>Component type: 32-bit float.</summary>
        public const uint FLOAT = 0x1406;
        /// <summary>Pixel-store parameter: row alignment for uploads (set to 1 for tightly packed 8-bit rows).</summary>
        public const uint UNPACK_ALIGNMENT = 0x0CF5;
        /// <summary>Pixel-store parameter: row alignment for readbacks.</summary>
        public const uint PACK_ALIGNMENT = 0x0D05;

        /// <summary>Primitive: triangle strip (the unit quad every sprite and full-screen pass is drawn with).</summary>
        public const uint TRIANGLE_STRIP = 0x0005;
        /// <summary>Primitive: triangles.</summary>
        public const uint TRIANGLES = 0x0004;

        /// <summary>Buffer target: vertex attributes.</summary>
        public const uint ARRAY_BUFFER = 0x8892;
        /// <summary>Usage hint: written once, drawn many times.</summary>
        public const uint STATIC_DRAW = 0x88E4;
        /// <summary>Usage hint: rewritten every frame (the particle and glyph instance streams).</summary>
        public const uint STREAM_DRAW = 0x88E0;

        /// <summary>Shader stage: fragment.</summary>
        public const uint FRAGMENT_SHADER = 0x8B30;
        /// <summary>Shader stage: vertex.</summary>
        public const uint VERTEX_SHADER = 0x8B31;
        /// <summary><see cref="GetShaderiv"/> query: compile succeeded.</summary>
        public const uint COMPILE_STATUS = 0x8B81;
        /// <summary><see cref="GetProgramiv"/> query: link succeeded.</summary>
        public const uint LINK_STATUS = 0x8B82;
        /// <summary>Info-log length query.</summary>
        public const uint INFO_LOG_LENGTH = 0x8B84;

        /// <summary>Framebuffer target (read + draw).</summary>
        public const uint FRAMEBUFFER = 0x8D40;
        /// <summary>First color attachment; attachment n is <c>COLOR_ATTACHMENT0 + n</c>.</summary>
        public const uint COLOR_ATTACHMENT0 = 0x8CE0;
        /// <summary><see cref="CheckFramebufferStatus"/> result: complete.</summary>
        public const uint FRAMEBUFFER_COMPLETE = 0x8CD5;

        /// <summary><see cref="GetString"/>: API version string.</summary>
        public const uint VERSION = 0x1F02;
        /// <summary><see cref="GetString"/>: renderer (GPU) name.</summary>
        public const uint RENDERER = 0x1F01;
        /// <summary><see cref="GetString"/>: vendor name.</summary>
        public const uint VENDOR = 0x1F00;
        /// <summary><see cref="GetError"/> result: no error.</summary>
        public const uint NO_ERROR = 0;

        // WGL context-creation attributes (WGL_ARB_create_context / _profile).
        /// <summary>Requested major version attribute.</summary>
        public const int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
        /// <summary>Requested minor version attribute.</summary>
        public const int WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
        /// <summary>Profile mask attribute.</summary>
        public const int WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126;
        /// <summary>Core profile bit (no deprecated fixed-function state).</summary>
        public const int WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001;

        // ---------------- opengl32.dll exports (WGL) ----------------

        /// <summary>Creates a legacy context — only used to bootstrap <c>wglCreateContextAttribsARB</c>.</summary>
        /// <param name="hdc">Device context with a pixel format already set.</param>
        /// <returns>Context handle, or 0.</returns>
        [DllImport("opengl32.dll", SetLastError = true)]
        public static extern nint wglCreateContext(nint hdc);

        /// <summary>Binds a context to this thread and DC (all GL calls go to the current context).</summary>
        /// <param name="hdc">Device context.</param>
        /// <param name="hglrc">Context (0 = unbind).</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("opengl32.dll", SetLastError = true)]
        public static extern int wglMakeCurrent(nint hdc, nint hglrc);

        /// <summary>Deletes a context.</summary>
        /// <param name="hglrc">Context.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("opengl32.dll", SetLastError = true)]
        public static extern int wglDeleteContext(nint hglrc);

        /// <summary>Looks up a post-1.1 or extension entry point for the CURRENT context.</summary>
        /// <param name="name">ANSI function name.</param>
        /// <returns>Address, or 0 / a small sentinel when the name is a 1.1 export or unknown.</returns>
        [DllImport("opengl32.dll", CharSet = CharSet.Ansi)]
        public static extern nint wglGetProcAddress(string name);

        // ---------------- function pointers (loaded by LoadAll) ----------------

        /// <summary>glViewport(x, y, w, h): maps NDC to the target's pixel rectangle — reset on every target switch.</summary>
        public static delegate* unmanaged<int, int, int, int, void> Viewport;
        /// <summary>glClearColor(r, g, b, a): the value <see cref="Clear"/> writes.</summary>
        public static delegate* unmanaged<float, float, float, float, void> ClearColor;
        /// <summary>glClear(mask): clears the bound target.</summary>
        public static delegate* unmanaged<uint, void> Clear;
        /// <summary>glEnable(cap).</summary>
        public static delegate* unmanaged<uint, void> Enable;
        /// <summary>glDisable(cap).</summary>
        public static delegate* unmanaged<uint, void> Disable;
        /// <summary>glBlendFunc(src, dst).</summary>
        public static delegate* unmanaged<uint, uint, void> BlendFunc;
        /// <summary>glBlendEquation(mode): ADD for accumulation, MAX for per-pixel maxima.</summary>
        public static delegate* unmanaged<uint, void> BlendEquation;
        /// <summary>glScissor(x, y, w, h).</summary>
        public static delegate* unmanaged<int, int, int, int, void> Scissor;
        /// <summary>glGetError(): the oldest queued error flag (checked in debug paths only — it can stall the pipeline).</summary>
        public static delegate* unmanaged<uint> GetError;
        /// <summary>glGetString(name): driver strings for the startup log.</summary>
        public static delegate* unmanaged<uint, byte*> GetString;
        /// <summary>glFinish(): blocks until the GPU is idle — used only when timing or reading back.</summary>
        public static delegate* unmanaged<void> Finish;
        /// <summary>glPixelStorei(pname, value).</summary>
        public static delegate* unmanaged<uint, int, void> PixelStorei;
        /// <summary>glReadPixels(x, y, w, h, format, type, dst): framebuffer readback for screenshots.</summary>
        public static delegate* unmanaged<int, int, int, int, uint, uint, void*, void> ReadPixels;

        /// <summary>glGenTextures(n, out ids).</summary>
        public static delegate* unmanaged<int, uint*, void> GenTextures;
        /// <summary>glDeleteTextures(n, ids).</summary>
        public static delegate* unmanaged<int, uint*, void> DeleteTextures;
        /// <summary>glBindTexture(target, id).</summary>
        public static delegate* unmanaged<uint, uint, void> BindTexture;
        /// <summary>glActiveTexture(unit): selects which unit <see cref="BindTexture"/> affects.</summary>
        public static delegate* unmanaged<uint, void> ActiveTexture;
        /// <summary>glTexImage2D(target, level, internalFormat, w, h, border, format, type, data): (re)allocates storage.</summary>
        public static delegate* unmanaged<uint, int, int, int, int, int, uint, uint, void*, void> TexImage2D;
        /// <summary>glTexSubImage2D(target, level, x, y, w, h, format, type, data): updates a region without reallocating.</summary>
        public static delegate* unmanaged<uint, int, int, int, int, int, uint, uint, void*, void> TexSubImage2D;
        /// <summary>glTexParameteri(target, pname, value).</summary>
        public static delegate* unmanaged<uint, uint, int, void> TexParameteri;

        /// <summary>glCreateShader(stage).</summary>
        public static delegate* unmanaged<uint, uint> CreateShader;
        /// <summary>glShaderSource(shader, count, strings, lengths).</summary>
        public static delegate* unmanaged<uint, int, byte**, int*, void> ShaderSource;
        /// <summary>glCompileShader(shader).</summary>
        public static delegate* unmanaged<uint, void> CompileShader;
        /// <summary>glGetShaderiv(shader, pname, out value).</summary>
        public static delegate* unmanaged<uint, uint, int*, void> GetShaderiv;
        /// <summary>glGetShaderInfoLog(shader, cap, out len, buffer).</summary>
        public static delegate* unmanaged<uint, int, int*, byte*, void> GetShaderInfoLog;
        /// <summary>glDeleteShader(shader).</summary>
        public static delegate* unmanaged<uint, void> DeleteShader;
        /// <summary>glCreateProgram().</summary>
        public static delegate* unmanaged<uint> CreateProgram;
        /// <summary>glAttachShader(program, shader).</summary>
        public static delegate* unmanaged<uint, uint, void> AttachShader;
        /// <summary>glLinkProgram(program).</summary>
        public static delegate* unmanaged<uint, void> LinkProgram;
        /// <summary>glGetProgramiv(program, pname, out value).</summary>
        public static delegate* unmanaged<uint, uint, int*, void> GetProgramiv;
        /// <summary>glGetProgramInfoLog(program, cap, out len, buffer).</summary>
        public static delegate* unmanaged<uint, int, int*, byte*, void> GetProgramInfoLog;
        /// <summary>glUseProgram(program).</summary>
        public static delegate* unmanaged<uint, void> UseProgram;
        /// <summary>glDeleteProgram(program).</summary>
        public static delegate* unmanaged<uint, void> DeleteProgram;
        /// <summary>glGetUniformLocation(program, name): −1 when the uniform was optimized out (setting −1 is a harmless no-op).</summary>
        public static delegate* unmanaged<uint, byte*, int> GetUniformLocation;
        /// <summary>glBindAttribLocation(program, index, name): pins attribute slots before linking.</summary>
        public static delegate* unmanaged<uint, uint, byte*, void> BindAttribLocation;
        /// <summary>glUniform1i(location, value) — also how a sampler is pointed at a texture unit.</summary>
        public static delegate* unmanaged<int, int, void> Uniform1i;
        /// <summary>glUniform1f(location, value).</summary>
        public static delegate* unmanaged<int, float, void> Uniform1f;
        /// <summary>glUniform2f(location, x, y).</summary>
        public static delegate* unmanaged<int, float, float, void> Uniform2f;
        /// <summary>glUniform3f(location, x, y, z).</summary>
        public static delegate* unmanaged<int, float, float, float, void> Uniform3f;
        /// <summary>glUniform4f(location, x, y, z, w).</summary>
        public static delegate* unmanaged<int, float, float, float, float, void> Uniform4f;
        /// <summary>glUniform1fv(location, count, values): uploads a float array uniform.</summary>
        public static delegate* unmanaged<int, int, float*, void> Uniform1fv;
        /// <summary>glUniform3fv(location, count, values): uploads a small per-material table.</summary>
        public static delegate* unmanaged<int, int, float*, void> Uniform3fv;
        /// <summary>glUniform4fv(location, count, values).</summary>
        public static delegate* unmanaged<int, int, float*, void> Uniform4fv;

        /// <summary>glGenVertexArrays(n, out ids): core profile requires a bound VAO for every draw.</summary>
        public static delegate* unmanaged<int, uint*, void> GenVertexArrays;
        /// <summary>glBindVertexArray(id).</summary>
        public static delegate* unmanaged<uint, void> BindVertexArray;
        /// <summary>glDeleteVertexArrays(n, ids).</summary>
        public static delegate* unmanaged<int, uint*, void> DeleteVertexArrays;
        /// <summary>glGenBuffers(n, out ids).</summary>
        public static delegate* unmanaged<int, uint*, void> GenBuffers;
        /// <summary>glBindBuffer(target, id).</summary>
        public static delegate* unmanaged<uint, uint, void> BindBuffer;
        /// <summary>glBufferData(target, bytes, data, usage): (re)allocates — also "orphans" a stream buffer so the driver never stalls on the previous frame's copy.</summary>
        public static delegate* unmanaged<uint, nint, void*, uint, void> BufferData;
        /// <summary>glBufferSubData(target, offset, bytes, data).</summary>
        public static delegate* unmanaged<uint, nint, nint, void*, void> BufferSubData;
        /// <summary>glDeleteBuffers(n, ids).</summary>
        public static delegate* unmanaged<int, uint*, void> DeleteBuffers;
        /// <summary>glVertexAttribPointer(index, size, type, normalized, stride, offset).</summary>
        public static delegate* unmanaged<uint, int, uint, byte, int, nint, void> VertexAttribPointer;
        /// <summary>glEnableVertexAttribArray(index).</summary>
        public static delegate* unmanaged<uint, void> EnableVertexAttribArray;
        /// <summary>glVertexAttribDivisor(index, divisor): 1 = advance once per instance (per particle / per glyph).</summary>
        public static delegate* unmanaged<uint, uint, void> VertexAttribDivisor;
        /// <summary>glDrawArrays(mode, first, count).</summary>
        public static delegate* unmanaged<uint, int, int, void> DrawArrays;
        /// <summary>glDrawArraysInstanced(mode, first, count, instances): one call draws every particle sprite.</summary>
        public static delegate* unmanaged<uint, int, int, int, void> DrawArraysInstanced;

        /// <summary>glGenFramebuffers(n, out ids).</summary>
        public static delegate* unmanaged<int, uint*, void> GenFramebuffers;
        /// <summary>glBindFramebuffer(target, id): 0 = the window.</summary>
        public static delegate* unmanaged<uint, uint, void> BindFramebuffer;
        /// <summary>glFramebufferTexture2D(target, attachment, texTarget, texture, level).</summary>
        public static delegate* unmanaged<uint, uint, uint, uint, int, void> FramebufferTexture2D;
        /// <summary>glCheckFramebufferStatus(target).</summary>
        public static delegate* unmanaged<uint, uint> CheckFramebufferStatus;
        /// <summary>glDrawBuffers(n, attachments): enables multiple render targets for one pass.</summary>
        public static delegate* unmanaged<int, uint*, void> DrawBuffers;
        /// <summary>glDeleteFramebuffers(n, ids).</summary>
        public static delegate* unmanaged<int, uint*, void> DeleteFramebuffers;

        /// <summary>wglSwapIntervalEXT(interval): 1 = vsync, 0 = uncapped, −1 = adaptive (tear only when late). Null if the extension is missing.</summary>
        public static delegate* unmanaged<int, int> SwapIntervalEXT;
        /// <summary>wglGetExtensionsStringEXT(): WGL extension list (to see whether adaptive vsync exists).</summary>
        public static delegate* unmanaged<byte*> GetExtensionsStringEXT;
        /// <summary>wglCreateContextAttribsARB(hdc, share, attribs): creates the 3.3 core context.</summary>
        public static delegate* unmanaged<nint, nint, int*, nint> CreateContextAttribsARB;

        /// <summary>
        /// Module handle of opengl32.dll: the GL 1.1 entry points must come from it, because
        /// <c>wglGetProcAddress</c> returns null (or the sentinels 1, 2, 3, −1) for them.
        /// </summary>
        private static nint _opengl32;

        /// <summary>
        /// Resolves one entry point for the current context, falling back from <c>wglGetProcAddress</c> to
        /// opengl32.dll's own exports (see the class remarks for why both are needed).
        /// </summary>
        /// <param name="name">GL/WGL function name.</param>
        /// <param name="required">When true a missing function throws; when false it yields 0 (optional extensions).</param>
        /// <returns>The function address, or 0 for a missing optional function.</returns>
        /// <exception cref="EntryPointNotFoundException">A required function is not provided by the driver (the GPU/driver cannot run the lab).</exception>
        public static nint Load(string name, bool required = true)
        {
            nint p = wglGetProcAddress(name);
            // Drivers signal "not an extension" with 0 or with the small sentinels 1/2/3/-1.
            if (p == 0 || p == 1 || p == 2 || p == 3 || p == -1)
            {
                if (_opengl32 == 0) _opengl32 = Win32.LoadLibraryW("opengl32.dll");
                p = Win32.GetProcAddress(_opengl32, name);
            }
            if (p == 0 && required)
                throw new EntryPointNotFoundException($"OpenGL function '{name}' is not available — the lab needs an OpenGL 3.3 capable GPU driver.");
            return p;
        }

        /// <summary>
        /// Loads the WGL extension entry points needed to create the real context. Must run while the
        /// bootstrap (legacy) context is current, because WGL extension pointers are resolved through it.
        /// </summary>
        public static void LoadWglBootstrap()
        {
            CreateContextAttribsARB = (delegate* unmanaged<nint, nint, int*, nint>)Load("wglCreateContextAttribsARB", required: false);
        }

        /// <summary>
        /// Loads every function the renderer uses. Call once, after the final (core) context is current.
        /// </summary>
        /// <exception cref="EntryPointNotFoundException">Some required function is missing (driver below GL 3.3).</exception>
        public static void LoadAll()
        {
            Viewport = (delegate* unmanaged<int, int, int, int, void>)Load("glViewport");
            ClearColor = (delegate* unmanaged<float, float, float, float, void>)Load("glClearColor");
            Clear = (delegate* unmanaged<uint, void>)Load("glClear");
            Enable = (delegate* unmanaged<uint, void>)Load("glEnable");
            Disable = (delegate* unmanaged<uint, void>)Load("glDisable");
            BlendFunc = (delegate* unmanaged<uint, uint, void>)Load("glBlendFunc");
            BlendEquation = (delegate* unmanaged<uint, void>)Load("glBlendEquation");
            Scissor = (delegate* unmanaged<int, int, int, int, void>)Load("glScissor");
            GetError = (delegate* unmanaged<uint>)Load("glGetError");
            GetString = (delegate* unmanaged<uint, byte*>)Load("glGetString");
            Finish = (delegate* unmanaged<void>)Load("glFinish");
            PixelStorei = (delegate* unmanaged<uint, int, void>)Load("glPixelStorei");
            ReadPixels = (delegate* unmanaged<int, int, int, int, uint, uint, void*, void>)Load("glReadPixels");

            GenTextures = (delegate* unmanaged<int, uint*, void>)Load("glGenTextures");
            DeleteTextures = (delegate* unmanaged<int, uint*, void>)Load("glDeleteTextures");
            BindTexture = (delegate* unmanaged<uint, uint, void>)Load("glBindTexture");
            ActiveTexture = (delegate* unmanaged<uint, void>)Load("glActiveTexture");
            TexImage2D = (delegate* unmanaged<uint, int, int, int, int, int, uint, uint, void*, void>)Load("glTexImage2D");
            TexSubImage2D = (delegate* unmanaged<uint, int, int, int, int, int, uint, uint, void*, void>)Load("glTexSubImage2D");
            TexParameteri = (delegate* unmanaged<uint, uint, int, void>)Load("glTexParameteri");

            CreateShader = (delegate* unmanaged<uint, uint>)Load("glCreateShader");
            ShaderSource = (delegate* unmanaged<uint, int, byte**, int*, void>)Load("glShaderSource");
            CompileShader = (delegate* unmanaged<uint, void>)Load("glCompileShader");
            GetShaderiv = (delegate* unmanaged<uint, uint, int*, void>)Load("glGetShaderiv");
            GetShaderInfoLog = (delegate* unmanaged<uint, int, int*, byte*, void>)Load("glGetShaderInfoLog");
            DeleteShader = (delegate* unmanaged<uint, void>)Load("glDeleteShader");
            CreateProgram = (delegate* unmanaged<uint>)Load("glCreateProgram");
            AttachShader = (delegate* unmanaged<uint, uint, void>)Load("glAttachShader");
            LinkProgram = (delegate* unmanaged<uint, void>)Load("glLinkProgram");
            GetProgramiv = (delegate* unmanaged<uint, uint, int*, void>)Load("glGetProgramiv");
            GetProgramInfoLog = (delegate* unmanaged<uint, int, int*, byte*, void>)Load("glGetProgramInfoLog");
            UseProgram = (delegate* unmanaged<uint, void>)Load("glUseProgram");
            DeleteProgram = (delegate* unmanaged<uint, void>)Load("glDeleteProgram");
            GetUniformLocation = (delegate* unmanaged<uint, byte*, int>)Load("glGetUniformLocation");
            BindAttribLocation = (delegate* unmanaged<uint, uint, byte*, void>)Load("glBindAttribLocation");
            Uniform1i = (delegate* unmanaged<int, int, void>)Load("glUniform1i");
            Uniform1f = (delegate* unmanaged<int, float, void>)Load("glUniform1f");
            Uniform2f = (delegate* unmanaged<int, float, float, void>)Load("glUniform2f");
            Uniform3f = (delegate* unmanaged<int, float, float, float, void>)Load("glUniform3f");
            Uniform4f = (delegate* unmanaged<int, float, float, float, float, void>)Load("glUniform4f");
            Uniform1fv = (delegate* unmanaged<int, int, float*, void>)Load("glUniform1fv");
            Uniform3fv = (delegate* unmanaged<int, int, float*, void>)Load("glUniform3fv");
            Uniform4fv = (delegate* unmanaged<int, int, float*, void>)Load("glUniform4fv");

            GenVertexArrays = (delegate* unmanaged<int, uint*, void>)Load("glGenVertexArrays");
            BindVertexArray = (delegate* unmanaged<uint, void>)Load("glBindVertexArray");
            DeleteVertexArrays = (delegate* unmanaged<int, uint*, void>)Load("glDeleteVertexArrays");
            GenBuffers = (delegate* unmanaged<int, uint*, void>)Load("glGenBuffers");
            BindBuffer = (delegate* unmanaged<uint, uint, void>)Load("glBindBuffer");
            BufferData = (delegate* unmanaged<uint, nint, void*, uint, void>)Load("glBufferData");
            BufferSubData = (delegate* unmanaged<uint, nint, nint, void*, void>)Load("glBufferSubData");
            DeleteBuffers = (delegate* unmanaged<int, uint*, void>)Load("glDeleteBuffers");
            VertexAttribPointer = (delegate* unmanaged<uint, int, uint, byte, int, nint, void>)Load("glVertexAttribPointer");
            EnableVertexAttribArray = (delegate* unmanaged<uint, void>)Load("glEnableVertexAttribArray");
            VertexAttribDivisor = (delegate* unmanaged<uint, uint, void>)Load("glVertexAttribDivisor");
            DrawArrays = (delegate* unmanaged<uint, int, int, void>)Load("glDrawArrays");
            DrawArraysInstanced = (delegate* unmanaged<uint, int, int, int, void>)Load("glDrawArraysInstanced");

            GenFramebuffers = (delegate* unmanaged<int, uint*, void>)Load("glGenFramebuffers");
            BindFramebuffer = (delegate* unmanaged<uint, uint, void>)Load("glBindFramebuffer");
            FramebufferTexture2D = (delegate* unmanaged<uint, uint, uint, uint, int, void>)Load("glFramebufferTexture2D");
            CheckFramebufferStatus = (delegate* unmanaged<uint, uint>)Load("glCheckFramebufferStatus");
            DrawBuffers = (delegate* unmanaged<int, uint*, void>)Load("glDrawBuffers");
            DeleteFramebuffers = (delegate* unmanaged<int, uint*, void>)Load("glDeleteFramebuffers");

            SwapIntervalEXT = (delegate* unmanaged<int, int>)Load("wglSwapIntervalEXT", required: false);
            GetExtensionsStringEXT = (delegate* unmanaged<byte*>)Load("wglGetExtensionsStringEXT", required: false);
        }

        /// <summary>Reads a driver string (<see cref="VERSION"/>, <see cref="RENDERER"/>, …).</summary>
        /// <param name="name">String id.</param>
        /// <returns>The string, or empty when the driver returns null.</returns>
        public static string GetStr(uint name)
        {
            byte* s = GetString(name);
            return s == null ? "" : Marshal.PtrToStringAnsi((nint)s) ?? "";
        }

        /// <summary>
        /// The WGL extension list, used to decide whether adaptive vsync (<c>WGL_EXT_swap_control_tear</c>)
        /// can be requested.
        /// </summary>
        /// <returns>Space-separated extension names, or empty.</returns>
        public static string WglExtensions()
        {
            if (GetExtensionsStringEXT == null) return "";
            byte* s = GetExtensionsStringEXT();
            return s == null ? "" : Marshal.PtrToStringAnsi((nint)s) ?? "";
        }

        /// <summary>Looks up a uniform by name (NUL-terminated ASCII built on the stack).</summary>
        /// <param name="program">Linked program.</param>
        /// <param name="name">Uniform name.</param>
        /// <returns>The location, or −1 when absent/optimized out.</returns>
        public static int Uniform(uint program, string name)
        {
            int n = Encoding.ASCII.GetByteCount(name);
            byte* buf = stackalloc byte[n + 1];
            Encoding.ASCII.GetBytes(name, new Span<byte>(buf, n));
            buf[n] = 0;
            return GetUniformLocation(program, buf);
        }
    }
}
