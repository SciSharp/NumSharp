using System.Diagnostics;
using System.Globalization;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Views;

/// <summary>Accessible native menu controls surrounding our own arcade drawing surface.</summary>
public sealed class GameSurface : UserControl, IDisposable
{
    private readonly ArcadeSession _session = new();
    private readonly ArenaView _arena;
    private readonly PlayerProfile _profile;
    private readonly IGameAudio _audio;
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(8) };
    private readonly HashSet<Key> _held = [];
    private readonly bool _animate;
    private readonly TextBlock _score = Label("0", 30, Brushes.White);
    private readonly TextBlock _next = Label("NEXT CELL +1", 13, ArenaView.Mint);
    private readonly TextBlock _lives = Label("●  ●  ●", 18, ArenaView.Coral);
    private readonly TextBlock _sector = Label("SECTOR 01", 13, ArenaView.Secondary);
    private readonly TextBlock _best = Label("BEST 0", 12, ArenaView.Secondary);
    private readonly TextBlock _message = Label("MOVE: W / S · ↑ / ↓ · MOUSE", 12, ArenaView.Secondary);
    private readonly TextBlock _phaseStatus = Label("COLONY · 70%", 12, ArenaView.Mint);
    private readonly TextBlock _hitFeedback = Label("20 SURGE · 50 OVERDRIVE · 100 SUPERNOVA", 13, ArenaView.Coral);
    private readonly TextBlock _overlayTitle = Label("LIFE ARCADE", 23, Brushes.White);
    private readonly TextBlock _overlayDetail = Label("", 14, ArenaView.Secondary);
    private readonly TextBlock _overlayStats = Label("", 13, ArenaView.Mint);
    private readonly Border _overlay;
    private readonly Button _primary, _restart, _pause, _cancel, _title;
    private readonly CheckBox _sound, _motion, _contrast;
    private readonly ProgressBar _progress;
    private readonly Grid _playLayout;
    private long _lastTimestamp;
    private double _accumulator;
    private bool _disposed, _confirmRestart, _recorded;
    private IPointer? _pointer;
    private double _feedbackSeconds;
    private string? _feedback;
    private bool _milestoneFeedback;

    public GameSurface() : this(true, PlayerProfile.OpenLocal(), App.AudioFactory()) { }
    internal GameSurface(bool startAnimation) : this(startAnimation, new PlayerProfile(), new SilentGameAudio()) { }
    internal GameSurface(bool animate, PlayerProfile profile, IGameAudio audio)
    {
        _animate = animate; _profile = profile; _audio = audio; Focusable = true; Background = ArenaView.Ink;
        if (animate) _session.NewRun(Random.Shared.Next());
        _arena = new ArenaView(_session, profile);
        _primary = MakeButton("Start run", StartOrResume, true);
        _pause = MakeButton("Pause / Esc", SuspendPlay, false);
        _restart = MakeButton("Restart run", () => { _confirmRestart = true; Refresh(); }, false);
        _cancel = MakeButton("Keep this run", () => { _confirmRestart = false; Refresh(); _primary.Focus(); }, false);
        _title = MakeButton("Back to title", () => { BeginNewRun(); Refresh(); Focus(); }, false);
        _sound = MakeOption("Sound", profile.Sound, value => profile.Sound = value); _sound.IsEnabled = audio.Available;
        if (!audio.Available) ToolTip.SetTip(_sound, "Sound is available in the Windows desktop release.");
        _motion = MakeOption("Reduced motion", profile.ReducedMotion, value => profile.ReducedMotion = value);
        _contrast = MakeOption("High contrast", profile.HighContrast, value => profile.HighContrast = value);
        _progress = new ProgressBar { Minimum = 0, Maximum = 40, Height = 3, Foreground = ArenaView.Mint, Background = ArenaView.Raised };
        AutomationProperties.SetName(_progress, "Cells destroyed toward next sector");
        var heading = new StackPanel { Spacing = 3, VerticalAlignment = VerticalAlignment.Center };
        heading.Children.Add(Label("NUMSHARP", 11, ArenaView.Mint)); heading.Children.Add(Label("LIFE ARCADE", 22, Brushes.White));
        var scoreGroup = new StackPanel { Spacing = 1, VerticalAlignment = VerticalAlignment.Center };
        scoreGroup.Children.Add(_score); scoreGroup.Children.Add(_next);
        var lifeGroup = new StackPanel { Spacing = 4, VerticalAlignment = VerticalAlignment.Center };
        lifeGroup.Children.Add(_lives); lifeGroup.Children.Add(_best);
        var sectorGroup = new StackPanel { Spacing = 9, Width = 150, VerticalAlignment = VerticalAlignment.Center };
        sectorGroup.Children.Add(_sector); sectorGroup.Children.Add(_progress);
        var hud = new Grid { ColumnDefinitions = new ColumnDefinitions("*,*,*,Auto,Auto"), ColumnSpacing = 28, Margin = new Thickness(28, 18, 28, 14) };
        Add(hud, heading, 0); Add(hud, scoreGroup, 1); Add(hud, lifeGroup, 2); Add(hud, sectorGroup, 3); Add(hud, _pause, 4);
        var menu = new StackPanel { Spacing = 14 };
        menu.Children.Add(_overlayTitle); menu.Children.Add(_overlayDetail); menu.Children.Add(_overlayStats);
        var actions = new StackPanel { Spacing = 8 };
        actions.Children.Add(_primary); actions.Children.Add(_restart); actions.Children.Add(_cancel); actions.Children.Add(_title); menu.Children.Add(actions);
        var options = new StackPanel { Spacing = 8 };
        options.Children.Add(_sound); options.Children.Add(_motion); options.Children.Add(_contrast);
        menu.Children.Add(new Border { Background = ArenaView.Line, Height = 1, Margin = new Thickness(0, 2) }); menu.Children.Add(options);
        _overlay = new Border
        {
            Background = new SolidColorBrush(Color.Parse("#F509141D")),
            BorderBrush = ArenaView.Line,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(20),
            Padding = new Thickness(20),
            Width = 320,
            HorizontalAlignment = HorizontalAlignment.Stretch,
            VerticalAlignment = VerticalAlignment.Stretch,
            Child = new ScrollViewer { Content = menu, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Disabled }
        };
        _playLayout = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto"), Margin = new Thickness(22, 0, 22, 0) };
        _playLayout.Children.Add(_arena); Add(_playLayout, _overlay, 1);
        var status = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto"), ColumnSpacing = 16, Margin = new Thickness(28, 0, 28, 10) };
        status.Children.Add(_phaseStatus); Add(status, _hitFeedback, 1);
        var footer = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto"), Margin = new Thickness(28, 8, 28, 16) };
        footer.Children.Add(_message); Add(footer, Label("GROW → SHATTER → REPEAT", 11, ArenaView.Mint), 1);
        var layout = new Grid { RowDefinitions = new RowDefinitions("Auto,Auto,*,Auto") };
        layout.Children.Add(hud); Grid.SetRow(status, 1); layout.Children.Add(status); Grid.SetRow(_playLayout, 2); layout.Children.Add(_playLayout); Grid.SetRow(footer, 3); layout.Children.Add(footer);
        Content = layout; _timer.Tick += OnFrame; Refresh();
    }
    internal ArcadeSession Session => _session;
    internal ArenaView Arena => _arena;
    internal Button PrimaryButton => _primary;
    internal Button PauseButton => _pause;
    internal Button RestartButton => _restart;
    internal Border MenuPanel => _overlay;
    internal TextBlock HitFeedback => _hitFeedback;
    internal bool HasTransientInputForTesting => _held.Count > 0 || _pointer is not null;
    private static TextBlock Label(string text, double size, IBrush color) => new() { Text = text, FontSize = size, Foreground = color, FontWeight = FontWeight.Medium, TextWrapping = TextWrapping.Wrap };
    private static void Add(Grid grid, Control control, int column) { Grid.SetColumn(control, column); grid.Children.Add(control); }
    private static Button MakeButton(string label, Action action, bool primary)
    {
        var button = new Button { Content = label, Padding = new Thickness(18, 12), CornerRadius = new CornerRadius(9), Background = primary ? ArenaView.Mint : ArenaView.Raised, Foreground = primary ? ArenaView.Ink : Brushes.White, VerticalAlignment = VerticalAlignment.Center, FontWeight = FontWeight.SemiBold };
        button.Click += (_, _) => action(); return button;
    }
    private CheckBox MakeOption(string label, bool initial, Action<bool> change)
    {
        var box = new CheckBox { Content = label, IsChecked = initial, Foreground = Brushes.White, FontSize = 13 };
        box.IsCheckedChanged += (_, _) => { change(box.IsChecked == true); _profile.Save(); _arena.InvalidateVisual(); Refresh(); }; return box;
    }
    private void StartOrResume()
    {
        if (_confirmRestart || _session.State == RunState.GameOver)
            BeginNewRun();
        _session.LaunchOrResume(); Refresh(); Focus();
    }
    private void BeginNewRun()
    {
        _session.NewRun(Random.Shared.Next()); _recorded = false; _confirmRestart = false;
        _feedback = null; _feedbackSeconds = 0; _milestoneFeedback = false; _arena.ClearEffects();
    }
    internal void SuspendPlay() { ReleaseInput(); _session.Pause(); _accumulator = 0; Refresh(); }
    private void ReleaseInput() { _held.Clear(); _session.ReleaseInput(); var pointer = _pointer; _pointer = null; pointer?.Capture(null); }
    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    { base.OnAttachedToVisualTree(e); _lastTimestamp = Stopwatch.GetTimestamp(); if (_animate && !_disposed) _timer.Start(); }
    protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
    { _timer.Stop(); SuspendPlay(); base.OnDetachedFromVisualTree(e); }
    protected override void OnLostFocus(FocusChangedEventArgs e)
    {
        base.OnLostFocus(e);
        if (!ReferenceEquals(e.Source, this)) return;
        ReleaseInput();
        if (_session.State == RunState.Playing) { _session.Pause(); _accumulator = 0; Refresh(); }
    }
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e); if (e.Handled) return;
        var first = _held.Add(e.Key);
        if (e.Key is Key.W or Key.Up or Key.S or Key.Down) { UpdateIntent(); e.Handled = true; }
        else if (e.Key == Key.Space)
        {
            if (first)
            {
                if (_session.State == RunState.Playing) PauseFromKey(e.Key);
                else StartOrResume();
            }
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        { if (first) { _confirmRestart = false; PauseFromKey(e.Key); } e.Handled = true; }
        else if (e.Key == Key.Enter && _session.State == RunState.GameOver) { if (first) StartOrResume(); e.Handled = true; }
    }
    private void UpdateIntent() => _session.SetIntent((_held.Contains(Key.S) || _held.Contains(Key.Down) ? 1 : 0) - (_held.Contains(Key.W) || _held.Contains(Key.Up) ? 1 : 0));
    private void PauseFromKey(Key key)
    {
        ReleaseInput(); _held.Add(key); _session.Pause(); _accumulator = 0; Refresh();
    }
    protected override void OnKeyUp(KeyEventArgs e)
    { base.OnKeyUp(e); _held.Remove(e.Key); if (e.Key is Key.W or Key.Up or Key.S or Key.Down) UpdateIntent(); }
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e); if (e.Handled || _session.State is RunState.Paused or RunState.GameOver || (_pointer is not null && _pointer != e.Pointer)) return;
        Focus(); _pointer = e.Pointer; e.Pointer.Capture(this); UpdatePointer(e.GetPosition(_arena));
    }
    protected override void OnPointerMoved(PointerEventArgs e)
    { base.OnPointerMoved(e); if (_session.State is RunState.Ready or RunState.Playing && (_pointer is null || _pointer == e.Pointer)) UpdatePointer(e.GetPosition(_arena)); }
    private void UpdatePointer(Point point)
    {
        var world = _arena.ToWorld(point);
        if (world.X >= ArcadeSession.Midline && world.X <= ArcadeSession.Width && world.Y >= 0 && world.Y <= ArcadeSession.Height) _session.SetPointerTarget((float)world.Y);
    }
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    { base.OnPointerReleased(e); if (_pointer != e.Pointer) return; _pointer = null; e.Pointer.Capture(null); _session.ReleaseInput(); }
    protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e)
    { base.OnPointerCaptureLost(e); if (_pointer != e.Pointer) return; _pointer = null; _session.ReleaseInput(); }
    private void OnFrame(object? sender, EventArgs e)
    {
        var now = Stopwatch.GetTimestamp(); var dt = Math.Clamp(Stopwatch.GetElapsedTime(_lastTimestamp, now).TotalSeconds, 0, .05); _lastTimestamp = now; AdvanceSimulation(dt);
    }
    internal void AdvanceSimulation(double seconds)
    {
        if (!double.IsFinite(seconds) || seconds < 0) throw new ArgumentOutOfRangeException(nameof(seconds));
        const double step = 1d / 120; _accumulator += Math.Min(.1, seconds);
        while (_accumulator >= step)
        {
            _session.Advance(step); _accumulator -= step;
            while (_session.TryTakeEvent(out var item))
            {
                _arena.OnEvent(item);
                if (item.Kind == ArcadeEventKind.Cell && (!_milestoneFeedback || _feedbackSeconds <= 0)) { _feedback = $"+{item.Value:N0} · {item.ShotHits} CELLS THIS SHOT"; _feedbackSeconds = .75; _milestoneFeedback = false; }
                if (item.Kind == ArcadeEventKind.Milestone) { _feedback = $"{ArenaView.TierName(ArcadeSession.TierForHits(item.Value))} · {item.Value} CELLS"; _feedbackSeconds = 1.8; _milestoneFeedback = true; }
                if (item.Kind is ArcadeEventKind.Paddle or ArcadeEventKind.Miss) { _feedback = null; _feedbackSeconds = 0; _milestoneFeedback = false; }
                if (_profile.Sound) _audio.Play(item);
            }
        }
        if (_session.State == RunState.GameOver && !_recorded) { _profile.Record(_session); _recorded = true; }
        var visualTime = _session.State == RunState.Paused ? 0 : Math.Min(.1, seconds);
        _feedbackSeconds = Math.Max(0, _feedbackSeconds - visualTime);
        _arena.AdvanceEffects(visualTime); Refresh();
    }
    private void Refresh()
    {
        if (_disposed) return;
        var longScore = _session.Score >= 1_000_000_000_000;
        _score.Text = _session.Score.ToString(longScore ? "0" : "N0", CultureInfo.InvariantCulture); _score.FontSize = longScore ? 18 : 30;
        _score.TextWrapping = TextWrapping.NoWrap;
        _next.Text = $"NEXT CELL +{_session.NextAward}  ·  SHOT {_session.Chain}";
        _lives.Text = string.Join("  ", Enumerable.Range(0, 3).Select(i => i < _session.Lives ? "●" : "○"));
        AutomationProperties.SetName(_lives, $"{_session.Lives} lives remaining");
        _sector.Text = $"SECTOR {_session.Sector:00}  ·  {_session.Destroyed % 40}/40"; _progress.Value = _session.Destroyed % 40; _best.Text = $"BEST {_profile.Best:N0}";
        _overlay.IsVisible = _session.State != RunState.Playing; _pause.IsEnabled = _session.State == RunState.Playing;
        _playLayout.ColumnSpacing = _overlay.IsVisible ? 18 : 0;
        _restart.IsVisible = _session.State == RunState.Paused && !_confirmRestart; _cancel.IsVisible = _confirmRestart;
        _title.IsVisible = _session.State == RunState.GameOver;
        _message.Text = _profile.SaveError ?? "MOVE: W / S · ↑ / ↓ · MOUSE     SPACE: LAUNCH / PAUSE";
        var phase = _session.State == RunState.Paused ? "PAUSED" : _session.ReturnAssist ? "RETURN" : _session.Frozen ? "SHATTER" : _session.Growing ? "GROW" : "COLONY";
        _phaseStatus.Text = $"{phase} · {_session.Life.LiveCount} LIVING · LIFE 70% / PLAYER 30%" + (_session.Replenishing ? " · NEW LIFE" : "");
        _phaseStatus.Foreground = _session.Frozen ? ArenaView.Coral : ArenaView.Mint;
        _hitFeedback.Text = _feedbackSeconds > 0 && _feedback is not null ? _feedback : _session.EffectTier > 0 ? $"{ArenaView.TierName(_session.EffectTier)} · {_session.Chain} CELLS THIS SHOT" : "20 SURGE · 50 OVERDRIVE · 100 SUPERNOVA";
        _hitFeedback.Foreground = _profile.HighContrast ? Brushes.White : ArenaView.TierBrush(_session.EffectTier);
        _hitFeedback.RenderTransformOrigin = RelativePoint.Center;
        var pulse = _profile.ReducedMotion || _feedbackSeconds <= 0 ? 1 : 1 + .04 * Math.Min(1, _feedbackSeconds);
        _hitFeedback.RenderTransform = new ScaleTransform(pulse, pulse);
        if (_confirmRestart) { _overlayTitle.Text = "START OVER?"; _overlayDetail.Text = "This run will be discarded. Your saved best stays safe."; _overlayStats.Text = $"CURRENT SCORE  {_session.Score:N0}"; _primary.Content = "Restart"; }
        else if (_session.State == RunState.Ready)
        {
            _overlayTitle.Text = _session.Lives == 3 ? "LET IT LIVE.\nMAKE IT SHATTER." : "ONE MORE SHOT.";
            _overlayDetail.Text = _session.Lives == 3 ? "Your paddle is on the right. Defend while Life grows.\nSend the ball left to freeze and destroy the colony." : "Score kept. Chain reset. Take a breath, then launch.";
            _overlayStats.Text = $"{_session.Lives} LIVES\nCELL AWARDS: +1, +2, +4, +6…\nPADDLE HIT RESETS THE COUNTER"; _primary.Content = _session.Lives == 3 ? "Start run  /  Space" : "Launch  /  Space";
        }
        else if (_session.State == RunState.Paused)
        { _overlayTitle.Text = "TAKE A BREATH."; _overlayDetail.Text = "Everything is paused. Your colony and chain are safe."; _overlayStats.Text = $"SCORE {_session.Score:N0}   ·   NEXT CELL +{_session.NextAward}"; _primary.Content = "Resume  /  Space"; }
        else if (_session.State == RunState.GameOver)
        {
            _overlayTitle.Text = _session.Score >= _profile.Best && _session.Score > 0 ? "NEW PERSONAL BEST." : "ONE MORE RUN?";
            _overlayDetail.Text = $"{_session.Score:N0} POINTS"; _overlayStats.Text = $"BEST SHOT {_session.BestChain} CELLS\n{_session.Destroyed} TOTAL · SECTOR {_session.Sector}\nSEED {_session.Seed}"; _primary.Content = "Retry  /  Enter";
        }
        _arena.InvalidateVisual();
    }
    public void Dispose()
    { if (_disposed) return; ReleaseInput(); _timer.Stop(); _timer.Tick -= OnFrame; _session.Dispose(); _audio.Dispose(); _disposed = true; }
}
