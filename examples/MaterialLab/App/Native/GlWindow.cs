using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Examples.MaterialLab.App.Native
{
    /// <summary>
    /// One key press delivered to the app this frame.
    /// </summary>
    /// <param name="VirtualKey">The Win32 virtual-key code (VK_*).</param>
    /// <param name="IsRepeat">True for auto-repeat while the key is held — one-shot actions (scene switch, pause) ignore these; continuous ones (brush size) accept them.</param>
    internal readonly record struct KeyPress(int VirtualKey, bool IsRepeat);

    /// <summary>
    /// The input snapshot the window procedure writes and the game loop reads once per frame. Mouse
    /// coordinates are CLIENT pixels with the origin at the top-left (the renderer flips to its own
    /// y-up simulation space).
    /// </summary>
    internal sealed class InputState
    {
        /// <summary>Cursor x in client pixels.</summary>
        public int MouseX;
        /// <summary>Cursor y in client pixels (0 = top).</summary>
        public int MouseY;
        /// <summary>True once the cursor has moved over the window — before that the position is a meaningless (0,0).</summary>
        public bool MouseSeen;
        /// <summary>Left button held.</summary>
        public bool LeftDown;
        /// <summary>Right button held.</summary>
        public bool RightDown;
        /// <summary>Middle button held.</summary>
        public bool MiddleDown;
        /// <summary>Wheel rotation accumulated since the last <see cref="EndFrame"/>, in notches (positive = away from the user).</summary>
        public float WheelNotches;
        /// <summary>Held state per virtual key (index = VK code).</summary>
        public readonly bool[] KeyDown = new bool[256];
        /// <summary>Presses since the last <see cref="EndFrame"/>, in arrival order.</summary>
        public readonly List<KeyPress> Presses = new List<KeyPress>(16);

        /// <summary>True while either Shift key is held.</summary>
        public bool Shift => KeyDown[0x10];
        /// <summary>True while either Ctrl key is held.</summary>
        public bool Ctrl => KeyDown[0x11];

        /// <summary>Clears the per-frame accumulators (presses, wheel). Held state persists.</summary>
        public void EndFrame()
        {
            Presses.Clear();
            WheelNotches = 0;
        }

        /// <summary>
        /// Drops held-button/key state — called when the window loses focus, because the matching
        /// "up" messages go to whichever window has focus and would otherwise leave a stuck button.
        /// </summary>
        public void ReleaseAll()
        {
            LeftDown = RightDown = MiddleDown = false;
            Array.Clear(KeyDown);
        }
    }

    /// <summary>
    /// Owns the application window and its OpenGL 3.3 core context: creates a window whose CLIENT area
    /// is a chosen fraction of the monitor (2/3 by default — the brief), bootstraps a modern context
    /// through a throwaway legacy one, pumps messages into an <see cref="InputState"/>, and flips between
    /// the framed window and borderless fullscreen (F11) without recreating the context.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>One window per process.</b> The window procedure is a static <see cref="UnmanagedCallersOnlyAttribute"/>
    /// method (no delegate to keep alive against the GC), so it routes to a single static instance.
    /// </para>
    /// <para>
    /// <b>Borderless fullscreen, not exclusive mode.</b> Switching the style to <c>WS_POPUP</c> and covering
    /// the monitor keeps the desktop resolution and the GL context intact, and modern drivers still
    /// present it through the fast "independent flip" path — so F11 is instant and never loses GPU state.
    /// </para>
    /// </remarks>
    internal sealed unsafe class GlWindow : IDisposable
    {
        /// <summary>
        /// The live window, for the static window procedure (an <c>UnmanagedCallersOnly</c> callback cannot
        /// capture <c>this</c>). One window per process.
        /// </summary>
        private static GlWindow s_current;
        /// <summary>
        /// An exception thrown inside the window procedure, rethrown by <see cref="PumpMessages"/>: an
        /// exception may not unwind through native frames, so it is parked here instead.
        /// </summary>
        private static Exception s_callbackError;

        /// <summary>The window handle.</summary>
        public nint Hwnd { get; private set; }
        /// <summary>The window's private device context (valid for the window's lifetime — <c>CS_OWNDC</c>).</summary>
        public nint Hdc { get; private set; }
        /// <summary>The OpenGL rendering context.</summary>
        public nint Hglrc { get; private set; }
        /// <summary>Client width in pixels (updated on <c>WM_SIZE</c>).</summary>
        public int Width { get; private set; }
        /// <summary>Client height in pixels.</summary>
        public int Height { get; private set; }
        /// <summary>Whether the window currently covers its monitor without a frame.</summary>
        public bool IsFullscreen { get; private set; }
        /// <summary>Set when the user closes the window; the game loop exits on it.</summary>
        public bool CloseRequested { get; private set; }
        /// <summary>Set when the client area changed size since the flag was last cleared by the renderer.</summary>
        public bool Resized { get; set; }
        /// <summary>The live input snapshot.</summary>
        public InputState Input { get; } = new InputState();
        /// <summary>The GL_VERSION string of the created context (for the startup log / HUD).</summary>
        public string GlVersion { get; private set; } = "";
        /// <summary>The GPU name (GL_RENDERER).</summary>
        public string GlRenderer { get; private set; } = "";
        /// <summary>The swap interval in effect: 1 vsync, 0 uncapped, −1 adaptive.</summary>
        public int SwapInterval { get; private set; }

        /// <summary>Window rectangle to restore when leaving fullscreen.</summary>
        private Win32.RECT _windowedRect;
        /// <summary>Window style to restore when leaving fullscreen.</summary>
        private uint _windowedStyle;
        /// <summary>Whether WGL_EXT_swap_control_tear exists (adaptive vsync: interval −1).</summary>
        private bool _adaptiveVsyncSupported;

        /// <summary>Instances come from <see cref="Create"/>, which also makes the GL context current.</summary>
        private GlWindow() { }

        /// <summary>
        /// Creates the window and its GL context. The client area is <paramref name="screenFraction"/> of
        /// the primary monitor in each dimension, centered in the monitor's work area; with
        /// <paramref name="fullscreen"/> it starts covering the whole monitor instead.
        /// </summary>
        /// <param name="title">Initial title bar text.</param>
        /// <param name="screenFraction">Fraction of the monitor's width and height for the windowed client area (e.g. 2/3).</param>
        /// <param name="fullscreen">Start in borderless fullscreen.</param>
        /// <param name="hidden">Create without showing (headless screenshot runs — the GL context still renders to offscreen targets).</param>
        /// <returns>The ready window with its context current on the calling thread.</returns>
        /// <exception cref="PlatformNotSupportedException">Not running on Windows.</exception>
        /// <exception cref="InvalidOperationException">A Win32 or WGL step failed (message names the step and the error code).</exception>
        /// <exception cref="EntryPointNotFoundException">The driver lacks an OpenGL 3.3 entry point.</exception>
        public static GlWindow Create(string title, double screenFraction, bool fullscreen, bool hidden = false)
        {
            if (!OperatingSystem.IsWindows())
                throw new PlatformNotSupportedException("The Material Lab front-end is a Win32/OpenGL app; run it on Windows (the Simulation and Verification projects are cross-platform).");
            if (s_current != null)
                throw new InvalidOperationException("Only one GlWindow may exist per process (the window procedure is static).");

            // Real pixels on scaled displays: without this, Windows would bitmap-stretch a smaller surface.
            Win32.SetProcessDpiAwarenessContext(Win32.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            DisableEcoQoS();

            var w = new GlWindow();
            s_current = w;
            nint hInstance = Win32.GetModuleHandleW(null);

            fixed (char* className = "NumSharpMaterialLab")
            {
                var wc = new Win32.WNDCLASSEXW
                {
                    cbSize = (uint)sizeof(Win32.WNDCLASSEXW),
                    style = Win32.CS_OWNDC | Win32.CS_HREDRAW | Win32.CS_VREDRAW,
                    lpfnWndProc = &WndProc,
                    hInstance = hInstance,
                    hCursor = Win32.LoadCursorW(0, Win32.IDC_ARROW),
                    lpszClassName = className,
                };
                if (Win32.RegisterClassExW(&wc) == 0)
                    throw new InvalidOperationException($"RegisterClassExW failed (error {Marshal.GetLastWin32Error()}).");

                var mi = PrimaryMonitor();
                int clientW = Math.Max(320, (int)Math.Round(mi.rcMonitor.Width * screenFraction));
                int clientH = Math.Max(200, (int)Math.Round(mi.rcMonitor.Height * screenFraction));
                uint style = Win32.WS_OVERLAPPEDWINDOW | Win32.WS_CLIPCHILDREN | Win32.WS_CLIPSIBLINGS;
                var r = new Win32.RECT { left = 0, top = 0, right = clientW, bottom = clientH };
                Win32.AdjustWindowRectEx(&r, style, 0, 0);   // grow client → outer so the CLIENT is exactly the fraction
                int outerW = r.Width, outerH = r.Height;
                int x = mi.rcWork.left + (mi.rcWork.Width - outerW) / 2;
                int y = mi.rcWork.top + Math.Max(0, (mi.rcWork.Height - outerH) / 2);

                fixed (char* titlePtr = title)
                    w.Hwnd = Win32.CreateWindowExW(0, className, titlePtr, style, x, y, outerW, outerH, 0, 0, hInstance, 0);
                if (w.Hwnd == 0)
                    throw new InvalidOperationException($"CreateWindowExW failed (error {Marshal.GetLastWin32Error()}).");
            }

            w.Hdc = Win32.GetDC(w.Hwnd);
            w.CreateContext();
            w.UpdateClientSize();
            if (fullscreen) w.ToggleFullscreen();
            if (!hidden) Win32.ShowWindow(w.Hwnd, Win32.SW_SHOW);
            w.UpdateClientSize();
            return w;
        }

        /// <summary>The primary monitor's geometry (the monitor that contains the virtual-screen origin).</summary>
        /// <returns>Monitor rect and work area.</returns>
        private static Win32.MONITORINFO PrimaryMonitor()
        {
            nint mon = Win32.MonitorFromPoint(new Win32.POINT { x = 0, y = 0 }, Win32.MONITOR_DEFAULTTOPRIMARY);
            var mi = new Win32.MONITORINFO { cbSize = (uint)sizeof(Win32.MONITORINFO) };
            Win32.GetMonitorInfoW(mon, &mi);
            return mi;
        }

        /// <summary>
        /// Opts the process out of Windows' EcoQoS execution-speed throttling. On hybrid CPUs a process
        /// Windows deems "background-ish" can be parked on efficiency cores and run ~1.8× slower; a
        /// real-time simulation wants performance cores. Best-effort: silently does nothing on Windows
        /// versions without the API.
        /// </summary>
        private static void DisableEcoQoS()
        {
            var state = new Win32.PROCESS_POWER_THROTTLING_STATE
            {
                Version = Win32.PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                ControlMask = Win32.PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                StateMask = 0,   // controlled bit cleared = throttling OFF
            };
            Win32.SetProcessInformation(Win32.GetCurrentProcess(), Win32.ProcessPowerThrottling, &state, (uint)sizeof(Win32.PROCESS_POWER_THROTTLING_STATE));
        }

        /// <summary>
        /// Sets the pixel format, creates a throwaway legacy context (the only way to reach
        /// <c>wglCreateContextAttribsARB</c>), then creates the real 3.3 core context and loads every entry
        /// point. Falls back to the legacy context — which on current drivers is a compatibility profile
        /// of the newest GL version, fine for <c>#version 330</c> shaders — if the ARB path is missing.
        /// </summary>
        /// <exception cref="InvalidOperationException">A pixel-format or context step failed.</exception>
        private void CreateContext()
        {
            var pfd = new Win32.PIXELFORMATDESCRIPTOR
            {
                nSize = (ushort)sizeof(Win32.PIXELFORMATDESCRIPTOR),
                nVersion = 1,
                dwFlags = Win32.PFD_DRAW_TO_WINDOW | Win32.PFD_SUPPORT_OPENGL | Win32.PFD_DOUBLEBUFFER,
                iPixelType = Win32.PFD_TYPE_RGBA,
                cColorBits = 32,
                cAlphaBits = 8,
                cDepthBits = 0,     // 2-D renderer: no depth buffer needed
                cStencilBits = 0,
            };
            int format = Win32.ChoosePixelFormat(Hdc, &pfd);
            if (format == 0 || Win32.SetPixelFormat(Hdc, format, &pfd) == 0)
                throw new InvalidOperationException($"Setting an OpenGL pixel format failed (error {Marshal.GetLastWin32Error()}).");

            nint legacy = Gl.wglCreateContext(Hdc);
            if (legacy == 0 || Gl.wglMakeCurrent(Hdc, legacy) == 0)
                throw new InvalidOperationException($"Creating the bootstrap OpenGL context failed (error {Marshal.GetLastWin32Error()}).");

            Gl.LoadWglBootstrap();
            Hglrc = legacy;
            if (Gl.CreateContextAttribsARB != null)
            {
                int* attribs = stackalloc int[]
                {
                    Gl.WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
                    Gl.WGL_CONTEXT_MINOR_VERSION_ARB, 3,
                    Gl.WGL_CONTEXT_PROFILE_MASK_ARB, Gl.WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    0,
                };
                nint core = Gl.CreateContextAttribsARB(Hdc, 0, attribs);
                if (core != 0 && Gl.wglMakeCurrent(Hdc, core) != 0)
                {
                    Gl.wglDeleteContext(legacy);
                    Hglrc = core;
                }
                else
                {
                    Gl.wglMakeCurrent(Hdc, legacy);   // keep the working compatibility context
                }
            }

            Gl.LoadAll();
            GlVersion = Gl.GetStr(Gl.VERSION);
            GlRenderer = Gl.GetStr(Gl.RENDERER);
            _adaptiveVsyncSupported = Gl.WglExtensions().Contains("WGL_EXT_swap_control_tear", StringComparison.Ordinal);
            SetVSync(true);
        }

        /// <summary>
        /// Turns vertical sync on or off. "On" prefers ADAPTIVE vsync when the driver supports it: frames
        /// that make the refresh deadline are tear-free, but a late frame is shown immediately instead of
        /// waiting a whole extra refresh — so a 17 ms frame costs ~17 ms, not the 33 ms plain vsync would
        /// quantize it to.
        /// </summary>
        /// <param name="on">True for (adaptive) vsync, false for uncapped presentation.</param>
        public void SetVSync(bool on)
        {
            if (Gl.SwapIntervalEXT == null) { SwapInterval = 0; return; }
            int interval = on ? (_adaptiveVsyncSupported ? -1 : 1) : 0;
            if (Gl.SwapIntervalEXT(interval) == 0 && interval == -1)
            {
                interval = 1;
                Gl.SwapIntervalEXT(1);
            }
            SwapInterval = interval;
        }

        /// <summary>Presents the frame (blocks per the swap interval).</summary>
        public void Swap() => Win32.SwapBuffers(Hdc);

        /// <summary>Asks the frame loop to finish (Esc) — the loop exits at its next check, so disposal stays orderly.</summary>
        public void RequestClose() => CloseRequested = true;

        /// <summary>Updates the title bar (the FPS readout is mirrored there for recordings without the HUD).</summary>
        /// <param name="text">New title.</param>
        public void SetTitle(string text) => Win32.SetWindowTextW(Hwnd, text);

        /// <summary>
        /// Drains the message queue without blocking, routing input into <see cref="Input"/>. Call once per
        /// frame before reading input.
        /// </summary>
        /// <exception cref="Exception">Rethrows an exception raised inside the window procedure (it cannot propagate through native frames, so it is parked and surfaced here).</exception>
        public void PumpMessages()
        {
            Win32.MSG msg;
            while (Win32.PeekMessageW(&msg, 0, 0, 0, Win32.PM_REMOVE) != 0)
            {
                if (msg.message == Win32.WM_QUIT) { CloseRequested = true; break; }
                Win32.TranslateMessage(&msg);
                Win32.DispatchMessageW(&msg);
            }
            if (s_callbackError != null)
            {
                var e = s_callbackError;
                s_callbackError = null;
                throw new InvalidOperationException("Exception inside the window procedure.", e);
            }
        }

        /// <summary>
        /// Flips between the framed window (restoring its previous rect) and borderless fullscreen on the
        /// monitor the window currently occupies. The GL context survives — only the window rect changes.
        /// </summary>
        public void ToggleFullscreen()
        {
            if (!IsFullscreen)
            {
                Win32.RECT wr;
                Win32.GetWindowRect(Hwnd, &wr);
                _windowedRect = wr;
                _windowedStyle = (uint)(long)Win32.GetWindowLongPtrW(Hwnd, Win32.GWL_STYLE);
                nint mon = Win32.MonitorFromWindow(Hwnd, Win32.MONITOR_DEFAULTTONEAREST);
                var mi = new Win32.MONITORINFO { cbSize = (uint)sizeof(Win32.MONITORINFO) };
                Win32.GetMonitorInfoW(mon, &mi);
                Win32.SetWindowLongPtrW(Hwnd, Win32.GWL_STYLE, (nint)(Win32.WS_POPUP | Win32.WS_VISIBLE | Win32.WS_CLIPCHILDREN | Win32.WS_CLIPSIBLINGS));
                Win32.SetWindowPos(Hwnd, 0, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.Width, mi.rcMonitor.Height,
                    Win32.SWP_NOZORDER | Win32.SWP_NOOWNERZORDER | Win32.SWP_FRAMECHANGED | Win32.SWP_SHOWWINDOW);
                IsFullscreen = true;
            }
            else
            {
                Win32.SetWindowLongPtrW(Hwnd, Win32.GWL_STYLE, (nint)_windowedStyle);
                Win32.SetWindowPos(Hwnd, 0, _windowedRect.left, _windowedRect.top, _windowedRect.Width, _windowedRect.Height,
                    Win32.SWP_NOZORDER | Win32.SWP_NOOWNERZORDER | Win32.SWP_FRAMECHANGED | Win32.SWP_SHOWWINDOW);
                IsFullscreen = false;
            }
            UpdateClientSize();
        }

        /// <summary>Re-reads the client size and raises <see cref="Resized"/> when it changed.</summary>
        private void UpdateClientSize()
        {
            Win32.RECT cr;
            Win32.GetClientRect(Hwnd, &cr);
            int nw = Math.Max(1, cr.Width), nh = Math.Max(1, cr.Height);
            if (nw != Width || nh != Height) { Width = nw; Height = nh; Resized = true; }
        }

        /// <summary>
        /// The window procedure. Runs on the UI thread inside <see cref="PumpMessages"/>; any exception is
        /// parked in <see cref="s_callbackError"/> because unwinding through the native dispatcher would
        /// crash the process.
        /// </summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="msg">Message id.</param>
        /// <param name="wParam">First parameter.</param>
        /// <param name="lParam">Second parameter.</param>
        /// <returns>The message result.</returns>
        [UnmanagedCallersOnly]
        private static nint WndProc(nint hwnd, uint msg, nint wParam, nint lParam)
        {
            try
            {
                var w = s_current;
                if (w == null || (w.Hwnd != 0 && hwnd != w.Hwnd))
                    return Win32.DefWindowProcW(hwnd, msg, wParam, lParam);
                var input = w.Input;
                switch (msg)
                {
                    case Win32.WM_CLOSE:
                        w.CloseRequested = true;
                        return 0;
                    case Win32.WM_DESTROY:
                        Win32.PostQuitMessage(0);
                        return 0;
                    case Win32.WM_ERASEBKGND:
                        return 1;   // never let GDI paint over the GL surface
                    case Win32.WM_SIZE:
                        if (w.Hwnd != 0) w.UpdateClientSize();
                        return 0;
                    case 0x0008:    // WM_KILLFOCUS: the "up" events will go elsewhere — drop held state
                        input.ReleaseAll();
                        break;
                    case Win32.WM_MOUSEMOVE:
                        input.MouseX = Win32.LoWordSigned(lParam);
                        input.MouseY = Win32.HiWordSigned(lParam);
                        input.MouseSeen = true;
                        return 0;
                    case Win32.WM_LBUTTONDOWN: input.LeftDown = true; Win32.SetCapture(hwnd); return 0;
                    case Win32.WM_LBUTTONUP: input.LeftDown = false; ReleaseIfIdle(input); return 0;
                    case Win32.WM_RBUTTONDOWN: input.RightDown = true; Win32.SetCapture(hwnd); return 0;
                    case Win32.WM_RBUTTONUP: input.RightDown = false; ReleaseIfIdle(input); return 0;
                    case Win32.WM_MBUTTONDOWN: input.MiddleDown = true; Win32.SetCapture(hwnd); return 0;
                    case Win32.WM_MBUTTONUP: input.MiddleDown = false; ReleaseIfIdle(input); return 0;
                    case Win32.WM_MOUSEWHEEL:
                        input.WheelNotches += (short)(((long)wParam >> 16) & 0xFFFF) / 120f;
                        return 0;
                    case Win32.WM_KEYDOWN:
                    case Win32.WM_SYSKEYDOWN:
                    {
                        int vk = (int)((long)wParam & 0xFF);
                        bool repeat = (((long)lParam >> 30) & 1) != 0;
                        input.KeyDown[vk] = true;
                        input.Presses.Add(new KeyPress(vk, repeat));
                        if (msg == Win32.WM_SYSKEYDOWN && vk != 0x73 /* let Alt+F4 through */) return 0;
                        break;
                    }
                    case Win32.WM_KEYUP:
                    case Win32.WM_SYSKEYUP:
                        input.KeyDown[(int)((long)wParam & 0xFF)] = false;
                        break;
                }
                return Win32.DefWindowProcW(hwnd, msg, wParam, lParam);
            }
            catch (Exception e)
            {
                s_callbackError ??= e;
                return 0;
            }
        }

        /// <summary>Releases the mouse capture once no button is held anymore.</summary>
        /// <param name="input">Current input state.</param>
        private static void ReleaseIfIdle(InputState input)
        {
            if (!input.LeftDown && !input.RightDown && !input.MiddleDown) Win32.ReleaseCapture();
        }

        /// <summary>Destroys the GL context and the window.</summary>
        public void Dispose()
        {
            if (Hglrc != 0)
            {
                Gl.wglMakeCurrent(0, 0);
                Gl.wglDeleteContext(Hglrc);
                Hglrc = 0;
            }
            if (Hwnd != 0)
            {
                Win32.ReleaseDC(Hwnd, Hdc);
                Win32.DestroyWindow(Hwnd);
                Hwnd = 0;
            }
            if (ReferenceEquals(s_current, this)) s_current = null;
        }
    }
}
