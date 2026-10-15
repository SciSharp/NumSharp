using System;
using System.Runtime.InteropServices;

namespace NumSharp.Examples.MaterialLab.App.Native
{
    /// <summary>
    /// The minimal slice of the Win32 API the lab needs to own a real window: register a class, create
    /// a window sized to a fraction of the monitor, pump messages, flip between a framed window and
    /// borderless fullscreen, and render text into a DIB for the HUD's font atlas. Declared by hand
    /// (plain blittable <c>DllImport</c>s, no NuGet) so the example keeps the zero-dependency
    /// character of NumSharp itself.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every signature is blittable (pointers, <c>nint</c>, fixed-size structs), so no marshalling
    /// stubs run per call — which matters for <see cref="PeekMessageW"/>, called every frame.
    /// </para>
    /// <para>
    /// The declarations compile on every OS (so the project builds in cross-platform CI), but they
    /// only resolve on Windows; the app checks <see cref="OperatingSystem.IsWindows"/> before touching
    /// any of them.
    /// </para>
    /// </remarks>
    internal static unsafe class Win32
    {
        // ---------------- window classes & styles ----------------

        /// <summary>Class style: the window keeps a private DC for its whole life — REQUIRED for OpenGL, whose context is bound to that DC.</summary>
        public const uint CS_OWNDC = 0x0020;
        /// <summary>Class style: redraw the whole client area when the width changes (avoids stale strips during a live resize).</summary>
        public const uint CS_HREDRAW = 0x0002;
        /// <summary>Class style: redraw the whole client area when the height changes.</summary>
        public const uint CS_VREDRAW = 0x0001;

        /// <summary>Window style: a normal framed, resizable window with caption and system buttons (the windowed mode).</summary>
        public const uint WS_OVERLAPPEDWINDOW = 0x00CF0000;
        /// <summary>Window style: no frame at all — combined with a monitor-sized rect this is borderless fullscreen.</summary>
        public const uint WS_POPUP = 0x80000000;
        /// <summary>Window style: initially visible.</summary>
        public const uint WS_VISIBLE = 0x10000000;
        /// <summary>Window style: exclude child windows when painting — recommended for GL windows so the swap chain owns every pixel.</summary>
        public const uint WS_CLIPCHILDREN = 0x02000000;
        /// <summary>Window style: exclude sibling windows when painting (paired with <see cref="WS_CLIPCHILDREN"/> for GL).</summary>
        public const uint WS_CLIPSIBLINGS = 0x04000000;
        /// <summary><see cref="GetWindowLongPtrW"/>/<see cref="SetWindowLongPtrW"/> index of the window style.</summary>
        public const int GWL_STYLE = -16;

        // ---------------- messages ----------------

        /// <summary>The window is being destroyed — the app stops its loop.</summary>
        public const uint WM_DESTROY = 0x0002;
        /// <summary>The client area changed size — the renderer resizes its framebuffers.</summary>
        public const uint WM_SIZE = 0x0005;
        /// <summary>The user asked to close the window (Alt+F4, the X button).</summary>
        public const uint WM_CLOSE = 0x0010;
        /// <summary>Posted by <see cref="PostQuitMessage"/>; ends the message pump.</summary>
        public const uint WM_QUIT = 0x0012;
        /// <summary>Background erase request — swallowed so GDI never paints over the GL surface (prevents flicker).</summary>
        public const uint WM_ERASEBKGND = 0x0014;
        /// <summary>A key went down (auto-repeats while held).</summary>
        public const uint WM_KEYDOWN = 0x0100;
        /// <summary>A key was released.</summary>
        public const uint WM_KEYUP = 0x0101;
        /// <summary>A key went down while Alt is held (or F10) — routed like <see cref="WM_KEYDOWN"/> so Alt+Enter works.</summary>
        public const uint WM_SYSKEYDOWN = 0x0104;
        /// <summary>A system key was released.</summary>
        public const uint WM_SYSKEYUP = 0x0105;
        /// <summary>The cursor moved inside the client area; <c>lParam</c> packs client x/y.</summary>
        public const uint WM_MOUSEMOVE = 0x0200;
        /// <summary>Left button pressed.</summary>
        public const uint WM_LBUTTONDOWN = 0x0201;
        /// <summary>Left button released.</summary>
        public const uint WM_LBUTTONUP = 0x0202;
        /// <summary>Right button pressed.</summary>
        public const uint WM_RBUTTONDOWN = 0x0204;
        /// <summary>Right button released.</summary>
        public const uint WM_RBUTTONUP = 0x0205;
        /// <summary>Middle button pressed.</summary>
        public const uint WM_MBUTTONDOWN = 0x0207;
        /// <summary>Middle button released.</summary>
        public const uint WM_MBUTTONUP = 0x0208;
        /// <summary>Wheel rotated; the high word of <c>wParam</c> is the signed delta in 120ths of a notch.</summary>
        public const uint WM_MOUSEWHEEL = 0x020A;
        /// <summary><see cref="PeekMessageW"/> flag: remove the message from the queue.</summary>
        public const uint PM_REMOVE = 0x0001;

        // ---------------- show / position ----------------

        /// <summary><see cref="ShowWindow"/>: activate and display in its current size and position.</summary>
        public const int SW_SHOW = 5;
        /// <summary><see cref="SetWindowPos"/>: keep the current Z order.</summary>
        public const uint SWP_NOZORDER = 0x0004;
        /// <summary><see cref="SetWindowPos"/>: re-evaluate the frame after a style change (required when switching windowed ↔ fullscreen).</summary>
        public const uint SWP_FRAMECHANGED = 0x0020;
        /// <summary><see cref="SetWindowPos"/>: do not change the owner's Z order.</summary>
        public const uint SWP_NOOWNERZORDER = 0x0200;
        /// <summary><see cref="SetWindowPos"/>: show the window.</summary>
        public const uint SWP_SHOWWINDOW = 0x0040;
        /// <summary><see cref="MonitorFromWindow"/>: when the window spans monitors, pick the one it overlaps most.</summary>
        public const uint MONITOR_DEFAULTTONEAREST = 2;
        /// <summary><see cref="MonitorFromPoint"/>: fall back to the primary monitor.</summary>
        public const uint MONITOR_DEFAULTTOPRIMARY = 1;
        /// <summary>Stock arrow cursor resource id for <see cref="LoadCursorW"/>.</summary>
        public const int IDC_ARROW = 32512;
        /// <summary>Per-monitor v2 DPI awareness: the window gets true physical pixels, so "2/3 of the screen" means 2/3 of the real pixels on a scaled display.</summary>
        public static readonly nint DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4;

        // ---------------- pixel format (legacy WGL bootstrap) ----------------

        /// <summary>Pixel-format flag: the format can render to a window.</summary>
        public const uint PFD_DRAW_TO_WINDOW = 0x00000004;
        /// <summary>Pixel-format flag: the format supports OpenGL.</summary>
        public const uint PFD_SUPPORT_OPENGL = 0x00000020;
        /// <summary>Pixel-format flag: double buffered (front/back swap) — mandatory for tear-free animation.</summary>
        public const uint PFD_DOUBLEBUFFER = 0x00000001;
        /// <summary>Pixel type: RGBA (as opposed to color-index).</summary>
        public const byte PFD_TYPE_RGBA = 0;

        // ---------------- GDI (font atlas) ----------------

        /// <summary><see cref="SetBkMode"/>: draw text without filling its background, so the DIB stays black around glyphs.</summary>
        public const int TRANSPARENT = 1;
        /// <summary>DIB color usage: the color table holds literal RGB values.</summary>
        public const uint DIB_RGB_COLORS = 0;
        /// <summary><see cref="CreateFontW"/> quality: grayscale anti-aliasing (NOT ClearType, whose colored fringes would bleed into the alpha channel).</summary>
        public const uint ANTIALIASED_QUALITY = 4;
        /// <summary>Normal font weight.</summary>
        public const int FW_NORMAL = 400;
        /// <summary>Semi-bold font weight (used for the big FPS readout).</summary>
        public const int FW_SEMIBOLD = 600;
        /// <summary>Default charset.</summary>
        public const uint DEFAULT_CHARSET = 1;
        /// <summary>Output precision: prefer TrueType outlines (scalable, hinted).</summary>
        public const uint OUT_TT_PRECIS = 4;
        /// <summary>Clip precision default.</summary>
        public const uint CLIP_DEFAULT_PRECIS = 0;
        /// <summary>Pitch/family default.</summary>
        public const uint DEFAULT_PITCH = 0;

        // ---------------- process QoS ----------------

        /// <summary><see cref="SetProcessInformation"/> class for power throttling (EcoQoS).</summary>
        public const int ProcessPowerThrottling = 4;
        /// <summary>Power-throttling control bit: execution speed (the EcoQoS bit that parks a process on efficiency cores).</summary>
        public const uint PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1;
        /// <summary>Version stamp required in <see cref="PROCESS_POWER_THROTTLING_STATE"/>.</summary>
        public const uint PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1;

        // ---------------- structs ----------------

        /// <summary>Window class registration record (<c>WNDCLASSEXW</c>). The procedure is an unmanaged function pointer, so no delegate has to be kept alive against the GC.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct WNDCLASSEXW
        {
            /// <summary>Size of this struct in bytes (must be set).</summary>
            public uint cbSize;
            /// <summary>Class style bits (<see cref="CS_OWNDC"/> for GL).</summary>
            public uint style;
            /// <summary>The window procedure.</summary>
            public delegate* unmanaged<nint, uint, nint, nint, nint> lpfnWndProc;
            /// <summary>Extra class bytes (unused).</summary>
            public int cbClsExtra;
            /// <summary>Extra window bytes (unused).</summary>
            public int cbWndExtra;
            /// <summary>Module that owns the class.</summary>
            public nint hInstance;
            /// <summary>Large icon (default).</summary>
            public nint hIcon;
            /// <summary>Cursor shown over the client area.</summary>
            public nint hCursor;
            /// <summary>Background brush — left null so Windows never paints over the GL surface.</summary>
            public nint hbrBackground;
            /// <summary>Menu resource name (none).</summary>
            public char* lpszMenuName;
            /// <summary>Class name the window is created from.</summary>
            public char* lpszClassName;
            /// <summary>Small icon (default).</summary>
            public nint hIconSm;
        }

        /// <summary>A queued window message (<c>MSG</c>).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct MSG
        {
            /// <summary>Target window.</summary>
            public nint hwnd;
            /// <summary>Message id.</summary>
            public uint message;
            /// <summary>First parameter.</summary>
            public nint wParam;
            /// <summary>Second parameter.</summary>
            public nint lParam;
            /// <summary>Post time.</summary>
            public uint time;
            /// <summary>Cursor x at post time (screen).</summary>
            public int ptX;
            /// <summary>Cursor y at post time (screen).</summary>
            public int ptY;
            /// <summary>Reserved.</summary>
            public uint lPrivate;
        }

        /// <summary>An integer rectangle (<c>RECT</c>); right/bottom are exclusive.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            /// <summary>Left edge.</summary>
            public int left;
            /// <summary>Top edge.</summary>
            public int top;
            /// <summary>Right edge (exclusive).</summary>
            public int right;
            /// <summary>Bottom edge (exclusive).</summary>
            public int bottom;
            /// <summary>Width in pixels.</summary>
            public readonly int Width => right - left;
            /// <summary>Height in pixels.</summary>
            public readonly int Height => bottom - top;
        }

        /// <summary>A screen point (<c>POINT</c>).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct POINT
        {
            /// <summary>X coordinate.</summary>
            public int x;
            /// <summary>Y coordinate.</summary>
            public int y;
        }

        /// <summary>Monitor geometry (<c>MONITORINFO</c>): the full monitor rect and its work area (minus the taskbar).</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct MONITORINFO
        {
            /// <summary>Size of this struct in bytes (must be set).</summary>
            public uint cbSize;
            /// <summary>Whole monitor, in virtual-screen coordinates (secondary monitors can be negative).</summary>
            public RECT rcMonitor;
            /// <summary>Monitor minus taskbar/docked bars.</summary>
            public RECT rcWork;
            /// <summary>1 for the primary monitor.</summary>
            public uint dwFlags;
        }

        /// <summary>Legacy pixel-format descriptor (<c>PIXELFORMATDESCRIPTOR</c>) used to bootstrap the first GL context.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PIXELFORMATDESCRIPTOR
        {
            /// <summary>Size of this struct.</summary>
            public ushort nSize;
            /// <summary>Always 1.</summary>
            public ushort nVersion;
            /// <summary>PFD_* capability flags.</summary>
            public uint dwFlags;
            /// <summary>RGBA or color-index.</summary>
            public byte iPixelType;
            /// <summary>Color bits excluding alpha.</summary>
            public byte cColorBits;
            /// <summary>Red bits.</summary>
            public byte cRedBits;
            /// <summary>Red shift.</summary>
            public byte cRedShift;
            /// <summary>Green bits.</summary>
            public byte cGreenBits;
            /// <summary>Green shift.</summary>
            public byte cGreenShift;
            /// <summary>Blue bits.</summary>
            public byte cBlueBits;
            /// <summary>Blue shift.</summary>
            public byte cBlueShift;
            /// <summary>Alpha bits.</summary>
            public byte cAlphaBits;
            /// <summary>Alpha shift.</summary>
            public byte cAlphaShift;
            /// <summary>Accumulation bits (unused).</summary>
            public byte cAccumBits;
            /// <summary>Accumulation red bits.</summary>
            public byte cAccumRedBits;
            /// <summary>Accumulation green bits.</summary>
            public byte cAccumGreenBits;
            /// <summary>Accumulation blue bits.</summary>
            public byte cAccumBlueBits;
            /// <summary>Accumulation alpha bits.</summary>
            public byte cAccumAlphaBits;
            /// <summary>Depth bits (the lab renders 2-D, so none are needed).</summary>
            public byte cDepthBits;
            /// <summary>Stencil bits.</summary>
            public byte cStencilBits;
            /// <summary>Aux buffers.</summary>
            public byte cAuxBuffers;
            /// <summary>Layer type (main plane).</summary>
            public byte iLayerType;
            /// <summary>Reserved.</summary>
            public byte bReserved;
            /// <summary>Layer mask.</summary>
            public uint dwLayerMask;
            /// <summary>Visible mask.</summary>
            public uint dwVisibleMask;
            /// <summary>Damage mask.</summary>
            public uint dwDamageMask;
        }

        /// <summary>Header of a device-independent bitmap (<c>BITMAPINFOHEADER</c>); a negative height makes the DIB top-down.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct BITMAPINFOHEADER
        {
            /// <summary>Size of this struct.</summary>
            public uint biSize;
            /// <summary>Width in pixels.</summary>
            public int biWidth;
            /// <summary>Height in pixels (negative = top-down rows).</summary>
            public int biHeight;
            /// <summary>Always 1.</summary>
            public ushort biPlanes;
            /// <summary>Bits per pixel (32 for BGRA).</summary>
            public ushort biBitCount;
            /// <summary>Compression (0 = BI_RGB).</summary>
            public uint biCompression;
            /// <summary>Image size (0 allowed for BI_RGB).</summary>
            public uint biSizeImage;
            /// <summary>Horizontal resolution.</summary>
            public int biXPelsPerMeter;
            /// <summary>Vertical resolution.</summary>
            public int biYPelsPerMeter;
            /// <summary>Palette size.</summary>
            public uint biClrUsed;
            /// <summary>Important colors.</summary>
            public uint biClrImportant;
        }

        /// <summary>Text extent returned by <see cref="GetTextExtentPoint32W"/>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct SIZE
        {
            /// <summary>Width.</summary>
            public int cx;
            /// <summary>Height.</summary>
            public int cy;
        }

        /// <summary>EcoQoS control block for <see cref="SetProcessInformation"/>.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PROCESS_POWER_THROTTLING_STATE
        {
            /// <summary><see cref="PROCESS_POWER_THROTTLING_CURRENT_VERSION"/>.</summary>
            public uint Version;
            /// <summary>Which throttling bits this call controls.</summary>
            public uint ControlMask;
            /// <summary>Desired state of the controlled bits (0 = throttling OFF).</summary>
            public uint StateMask;
        }

        // ---------------- user32 ----------------

        /// <summary>Registers the window class.</summary>
        /// <param name="wc">The class record; <c>cbSize</c> must be set.</param>
        /// <returns>A class atom, or 0 on failure (see <see cref="Marshal.GetLastWin32Error"/>).</returns>
        [DllImport("user32.dll", SetLastError = true)]
        public static extern ushort RegisterClassExW(WNDCLASSEXW* wc);

        /// <summary>Creates the window.</summary>
        /// <param name="exStyle">Extended style bits.</param>
        /// <param name="className">Registered class name.</param>
        /// <param name="windowName">Title bar text.</param>
        /// <param name="style">WS_* style bits.</param>
        /// <param name="x">Left edge (screen).</param>
        /// <param name="y">Top edge (screen).</param>
        /// <param name="width">Outer width including the frame.</param>
        /// <param name="height">Outer height including the frame.</param>
        /// <param name="parent">Parent window (none).</param>
        /// <param name="menu">Menu (none).</param>
        /// <param name="instance">Owning module.</param>
        /// <param name="param">Creation parameter (unused).</param>
        /// <returns>The window handle, or 0 on failure.</returns>
        [DllImport("user32.dll", SetLastError = true)]
        public static extern nint CreateWindowExW(uint exStyle, char* className, char* windowName, uint style,
            int x, int y, int width, int height, nint parent, nint menu, nint instance, nint param);

        /// <summary>Default message handling for everything the app does not intercept.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="msg">Message.</param>
        /// <param name="wParam">First parameter.</param>
        /// <param name="lParam">Second parameter.</param>
        /// <returns>The message result.</returns>
        [DllImport("user32.dll")]
        public static extern nint DefWindowProcW(nint hwnd, uint msg, nint wParam, nint lParam);

        /// <summary>Non-blocking message fetch — the heart of a game loop (never waits, unlike <c>GetMessage</c>).</summary>
        /// <param name="msg">Receives the message.</param>
        /// <param name="hwnd">Filter window (0 = any of this thread's).</param>
        /// <param name="min">Minimum message id (0 = all).</param>
        /// <param name="max">Maximum message id (0 = all).</param>
        /// <param name="removeMsg"><see cref="PM_REMOVE"/> to dequeue.</param>
        /// <returns>Nonzero when a message was retrieved.</returns>
        [DllImport("user32.dll")]
        public static extern int PeekMessageW(MSG* msg, nint hwnd, uint min, uint max, uint removeMsg);

        /// <summary>Translates virtual-key messages into character messages.</summary>
        /// <param name="msg">The message.</param>
        /// <returns>Nonzero if a character message was produced.</returns>
        [DllImport("user32.dll")]
        public static extern int TranslateMessage(MSG* msg);

        /// <summary>Delivers a message to its window procedure.</summary>
        /// <param name="msg">The message.</param>
        /// <returns>The procedure's result.</returns>
        [DllImport("user32.dll")]
        public static extern nint DispatchMessageW(MSG* msg);

        /// <summary>Posts <see cref="WM_QUIT"/> to end the loop.</summary>
        /// <param name="exitCode">Exit code carried by the message.</param>
        [DllImport("user32.dll")]
        public static extern void PostQuitMessage(int exitCode);

        /// <summary>Shows or hides the window.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="cmd">SW_* command.</param>
        /// <returns>Nonzero if the window was previously visible.</returns>
        [DllImport("user32.dll")]
        public static extern int ShowWindow(nint hwnd, int cmd);

        /// <summary>Moves/resizes the window and applies frame changes.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="insertAfter">Z-order anchor (ignored with <see cref="SWP_NOZORDER"/>).</param>
        /// <param name="x">Left (screen).</param>
        /// <param name="y">Top (screen).</param>
        /// <param name="cx">Outer width.</param>
        /// <param name="cy">Outer height.</param>
        /// <param name="flags">SWP_* flags.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll", SetLastError = true)]
        public static extern int SetWindowPos(nint hwnd, nint insertAfter, int x, int y, int cx, int cy, uint flags);

        /// <summary>Reads the client-area size (origin is always 0,0).</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="rect">Receives the client rect.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int GetClientRect(nint hwnd, RECT* rect);

        /// <summary>Reads the outer window rect in screen coordinates.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="rect">Receives the window rect.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int GetWindowRect(nint hwnd, RECT* rect);

        /// <summary>Grows a desired CLIENT rect to the OUTER rect a given style needs — how the window gets an exact 2/3-of-the-screen client area.</summary>
        /// <param name="rect">In: client rect. Out: window rect.</param>
        /// <param name="style">Window style.</param>
        /// <param name="menu">Whether a menu bar is present.</param>
        /// <param name="exStyle">Extended style.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int AdjustWindowRectEx(RECT* rect, uint style, int menu, uint exStyle);

        /// <summary>Reads a window attribute (the style, for the fullscreen toggle).</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="index">GWL_* index.</param>
        /// <returns>The attribute value.</returns>
        [DllImport("user32.dll")]
        public static extern nint GetWindowLongPtrW(nint hwnd, int index);

        /// <summary>Writes a window attribute (the style, for the fullscreen toggle).</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="index">GWL_* index.</param>
        /// <param name="value">New value.</param>
        /// <returns>The previous value.</returns>
        [DllImport("user32.dll")]
        public static extern nint SetWindowLongPtrW(nint hwnd, int index, nint value);

        /// <summary>Finds the monitor a window is on.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="flags">MONITOR_DEFAULTTO* fallback.</param>
        /// <returns>Monitor handle.</returns>
        [DllImport("user32.dll")]
        public static extern nint MonitorFromWindow(nint hwnd, uint flags);

        /// <summary>Finds the monitor containing a point (the origin → the primary monitor before any window exists).</summary>
        /// <param name="pt">Screen point.</param>
        /// <param name="flags">MONITOR_DEFAULTTO* fallback.</param>
        /// <returns>Monitor handle.</returns>
        [DllImport("user32.dll")]
        public static extern nint MonitorFromPoint(POINT pt, uint flags);

        /// <summary>Reads a monitor's rect and work area.</summary>
        /// <param name="monitor">Monitor handle.</param>
        /// <param name="info">Receives the info; <c>cbSize</c> must be set.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int GetMonitorInfoW(nint monitor, MONITORINFO* info);

        /// <summary>Loads a stock cursor.</summary>
        /// <param name="instance">0 for system cursors.</param>
        /// <param name="cursorName">Resource id (e.g. <see cref="IDC_ARROW"/>).</param>
        /// <returns>Cursor handle.</returns>
        [DllImport("user32.dll")]
        public static extern nint LoadCursorW(nint instance, nint cursorName);

        /// <summary>Gets the window's private device context (stable for the window's life thanks to <see cref="CS_OWNDC"/>).</summary>
        /// <param name="hwnd">Window (0 = the screen).</param>
        /// <returns>DC handle.</returns>
        [DllImport("user32.dll")]
        public static extern nint GetDC(nint hwnd);

        /// <summary>Releases a DC from <see cref="GetDC"/>.</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="hdc">DC.</param>
        /// <returns>1 when released.</returns>
        [DllImport("user32.dll")]
        public static extern int ReleaseDC(nint hwnd, nint hdc);

        /// <summary>Destroys the window.</summary>
        /// <param name="hwnd">Window.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int DestroyWindow(nint hwnd);

        /// <summary>Sets the title bar text (the live FPS is mirrored there too).</summary>
        /// <param name="hwnd">Window.</param>
        /// <param name="text">New title.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern int SetWindowTextW(nint hwnd, string text);

        /// <summary>Captures the mouse so drags keep reporting even when the cursor leaves the client area.</summary>
        /// <param name="hwnd">Window.</param>
        /// <returns>The previous capture window.</returns>
        [DllImport("user32.dll")]
        public static extern nint SetCapture(nint hwnd);

        /// <summary>Ends a mouse capture.</summary>
        /// <returns>Nonzero on success.</returns>
        [DllImport("user32.dll")]
        public static extern int ReleaseCapture();

        /// <summary>Reads a key's current up/down state (high bit set = down); used for modifier keys.</summary>
        /// <param name="virtualKey">VK_* code.</param>
        /// <returns>State bits.</returns>
        [DllImport("user32.dll")]
        public static extern short GetKeyState(int virtualKey);

        /// <summary>Opts the process into per-monitor DPI awareness (must run before the first window is created).</summary>
        /// <param name="value">A DPI_AWARENESS_CONTEXT value.</param>
        /// <returns>Nonzero on success; 0 when unsupported or already set (harmless).</returns>
        [DllImport("user32.dll")]
        public static extern int SetProcessDpiAwarenessContext(nint value);

        // ---------------- kernel32 ----------------

        /// <summary>Handle of a loaded module (0 = the exe) — the <c>hInstance</c> for window classes.</summary>
        /// <param name="moduleName">Module name or null.</param>
        /// <returns>Module handle.</returns>
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern nint GetModuleHandleW(string moduleName);

        /// <summary>Loads a DLL (used for opengl32.dll, whose 1.1 entry points <c>wglGetProcAddress</c> refuses to return).</summary>
        /// <param name="fileName">DLL name.</param>
        /// <returns>Module handle, or 0.</returns>
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern nint LoadLibraryW(string fileName);

        /// <summary>Looks up an exported function.</summary>
        /// <param name="module">Module handle.</param>
        /// <param name="procName">ANSI export name.</param>
        /// <returns>Function address, or 0.</returns>
        [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern nint GetProcAddress(nint module, string procName);

        /// <summary>Pseudo-handle of this process.</summary>
        /// <returns>Always -1; never needs closing.</returns>
        [DllImport("kernel32.dll")]
        public static extern nint GetCurrentProcess();

        /// <summary>Sets a process information class — used once to opt OUT of EcoQoS so Windows does not park the simulation on efficiency cores.</summary>
        /// <param name="process">Process handle.</param>
        /// <param name="infoClass"><see cref="ProcessPowerThrottling"/>.</param>
        /// <param name="info">The state block.</param>
        /// <param name="infoSize">Size of the block.</param>
        /// <returns>Nonzero on success (0 on Windows versions without EcoQoS — harmless).</returns>
        [DllImport("kernel32.dll")]
        public static extern int SetProcessInformation(nint process, int infoClass, void* info, uint infoSize);

        // ---------------- gdi32 ----------------

        /// <summary>Picks the closest supported pixel format for a DC.</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="pfd">Requested format.</param>
        /// <returns>Format index, or 0.</returns>
        [DllImport("gdi32.dll", SetLastError = true)]
        public static extern int ChoosePixelFormat(nint hdc, PIXELFORMATDESCRIPTOR* pfd);

        /// <summary>Sets a DC's pixel format — allowed exactly ONCE per window, which is why the window is never recycled across context re-creation.</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="format">Format index.</param>
        /// <param name="pfd">The descriptor.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll", SetLastError = true)]
        public static extern int SetPixelFormat(nint hdc, int format, PIXELFORMATDESCRIPTOR* pfd);

        /// <summary>Presents the back buffer (honours the swap interval set by <c>wglSwapIntervalEXT</c>).</summary>
        /// <param name="hdc">DC.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int SwapBuffers(nint hdc);

        /// <summary>Creates an off-screen DC for rendering glyphs into a DIB.</summary>
        /// <param name="hdc">Reference DC (0 = screen-compatible).</param>
        /// <returns>DC handle.</returns>
        [DllImport("gdi32.dll")]
        public static extern nint CreateCompatibleDC(nint hdc);

        /// <summary>Deletes a DC from <see cref="CreateCompatibleDC"/>.</summary>
        /// <param name="hdc">DC.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int DeleteDC(nint hdc);

        /// <summary>Creates a DIB section whose pixels are directly addressable — the font atlas is drawn into it, then uploaded as a texture.</summary>
        /// <param name="hdc">Reference DC.</param>
        /// <param name="bmi">Header (32-bpp, negative height = top-down).</param>
        /// <param name="usage"><see cref="DIB_RGB_COLORS"/>.</param>
        /// <param name="bits">Receives the pixel pointer.</param>
        /// <param name="section">File mapping (0).</param>
        /// <param name="offset">Offset in the mapping (0).</param>
        /// <returns>Bitmap handle, or 0.</returns>
        [DllImport("gdi32.dll")]
        public static extern nint CreateDIBSection(nint hdc, BITMAPINFOHEADER* bmi, uint usage, void** bits, nint section, uint offset);

        /// <summary>Selects a GDI object into a DC.</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="obj">Object (bitmap/font).</param>
        /// <returns>The previously selected object.</returns>
        [DllImport("gdi32.dll")]
        public static extern nint SelectObject(nint hdc, nint obj);

        /// <summary>Deletes a GDI object.</summary>
        /// <param name="obj">Object.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int DeleteObject(nint obj);

        /// <summary>Creates a font.</summary>
        /// <param name="height">Negative = character height in pixels (em size).</param>
        /// <param name="width">0 = default aspect.</param>
        /// <param name="escapement">Rotation (0).</param>
        /// <param name="orientation">Rotation (0).</param>
        /// <param name="weight">FW_* weight.</param>
        /// <param name="italic">0/1.</param>
        /// <param name="underline">0/1.</param>
        /// <param name="strikeOut">0/1.</param>
        /// <param name="charSet">Charset.</param>
        /// <param name="outPrecision">Output precision.</param>
        /// <param name="clipPrecision">Clip precision.</param>
        /// <param name="quality">Anti-aliasing quality.</param>
        /// <param name="pitchAndFamily">Pitch and family.</param>
        /// <param name="faceName">Typeface name (falls back if missing).</param>
        /// <returns>Font handle.</returns>
        [DllImport("gdi32.dll", CharSet = CharSet.Unicode)]
        public static extern nint CreateFontW(int height, int width, int escapement, int orientation, int weight,
            uint italic, uint underline, uint strikeOut, uint charSet, uint outPrecision, uint clipPrecision,
            uint quality, uint pitchAndFamily, string faceName);

        /// <summary>Sets the text color for subsequent text output.</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="color">0x00BBGGRR.</param>
        /// <returns>The previous color.</returns>
        [DllImport("gdi32.dll")]
        public static extern uint SetTextColor(nint hdc, uint color);

        /// <summary>Sets the background mode (<see cref="TRANSPARENT"/> for the atlas).</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="mode">Mode.</param>
        /// <returns>The previous mode.</returns>
        [DllImport("gdi32.dll")]
        public static extern int SetBkMode(nint hdc, int mode);

        /// <summary>Draws a run of text.</summary>
        /// <param name="hdc">DC.</param>
        /// <param name="x">Left.</param>
        /// <param name="y">Top.</param>
        /// <param name="text">Characters.</param>
        /// <param name="count">Character count.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int TextOutW(nint hdc, int x, int y, char* text, int count);

        /// <summary>Measures a run of text (used to get each glyph's advance for the atlas).</summary>
        /// <param name="hdc">DC with the font selected.</param>
        /// <param name="text">Characters.</param>
        /// <param name="count">Character count.</param>
        /// <param name="size">Receives the extent.</param>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int GetTextExtentPoint32W(nint hdc, char* text, int count, SIZE* size);

        /// <summary>Flushes pending GDI drawing into the DIB before its bits are read.</summary>
        /// <returns>Nonzero on success.</returns>
        [DllImport("gdi32.dll")]
        public static extern int GdiFlush();

        // ---------------- helpers ----------------

        /// <summary>Signed client x from a mouse message's <c>lParam</c> (low word; sign matters when the mouse is captured and leaves the window).</summary>
        /// <param name="lParam">Message parameter.</param>
        /// <returns>X in client pixels.</returns>
        public static int LoWordSigned(nint lParam) => (short)((long)lParam & 0xFFFF);

        /// <summary>Signed client y from a mouse message's <c>lParam</c> (high word).</summary>
        /// <param name="lParam">Message parameter.</param>
        /// <returns>Y in client pixels.</returns>
        public static int HiWordSigned(nint lParam) => (short)(((long)lParam >> 16) & 0xFFFF);
    }
}
