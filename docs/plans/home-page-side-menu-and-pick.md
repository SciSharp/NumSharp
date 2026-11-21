# Docs home page: side menu, smaller editor, Pick ring, smaller integrations picture

The owner's five asks of 2026-11-21 for https://scisharp.github.io/NumSharp/, executed as phases on `master` (one commit
per phase, not pushed: the site redeploys only when `master` is pushed). Every decision below that the owner did not
state is an interpretation, listed again under "Open for rediscussion".

Files: `docs/website-src/index.md` (the page), `docs/website-src/templates/numsharp/public/main.css` (its styles),
`docs/website-src/templates/numsharp/public/home.js` (its behavior). Verification: `docfx build` into a scratch folder,
served over HTTP, screenshots and DOM measurements from an isolated headless Chromium (Python Playwright).

## Status

| Phase | Ask | State |
|-------|-----|-------|
| 0 | Plan, baseline build and screenshots | done: docs pages' menu 283 px at 1440, 360 px at 1920 |
| 1 | The docs side menu on the home page, narrower | redone: two static 252 px copies beside the first and last areas; band full width |
| 2 | The NumPy / C# editor 20% smaller | done: 912 px (was 1140), code at most 12.8 px (was 16) |
| 3 | The example title centered above the button | done: centers equal (x 846 at 1440), gap 10 px |
| 4 | "Pick": six topic buttons around the button, then "Crazier" inside the picked topic | done: mouse, keyboard, touch verified |
| 5 | The integrations picture 30% smaller, "Integrations" centered above the section | done: 70% width at every desktop size; title on the picture's axis |

## Phase 1: the side menu

- The home page already carries DocFX's table-of-contents markup (`.toc-offcanvas` / `#toc`) and its
  `docfx:tocrel` already points at `docs/toc.html` (docs/toc.yml lists the home page as "Home"), so `docfx.min.js`
  fills it on the home page too. The `landing` layout only hides it (`body > main > :not(.content)`). The phase is
  CSS: show it on the home page as a sticky left column, as the conceptual layout does on `docs/intro.html`.
- Narrower than on docs pages (there `flex: .35`, up to 360 px): a fixed width that fits the top-level entries, so
  the home page's own content keeps its room.
- Phones (< 768 px): the docs pages open the menu from a button in the action bar; the home page shows that bar's
  button too (the breadcrumb stays hidden).
- Done: `--ns-side-width: 252px`. The longest top-level entry ("The absolute basics for beginners", 207 px of
  Segoe UI at 14 px) needs 237 px of column on Windows and wrapped at 236; 252 leaves room for macOS / Linux fonts.
  Measured: sticky at `top: 60px` after scrolling, "Home" active, no horizontal scroll at 1920 / 1440 / 1024 /
  390 px, the drawer opens at 390. Side effect fixed: beside the menu a 1024 px window leaves the hero 772 px, where
  the install box (its command cannot wrap) overran the editor's right edge by 13 px; its grid column now has a
  `min-content` floor, and both edges meet at 1024 / 1280 / 1440.
- Redone the same day. The owner: "The sidemenu should ONLY exist on the first and last areas of 'Write NumPy in
  C#.' maybe even static two copies will do better." The sticky column ran beside every band, so the integrations
  band lost its full width. Now:
  - Two areas, `.ns-area` grids of [menu column | band column]: the hero with a copy beside it, the integrations
    band at the whole width, the documentation map with a second copy beside it. DocFX's real table stays hidden
    in `#toc` (and is still the phone drawer).
  - The copies are static: in the flow at their whole height, no scrolling of their own. With the docs pages'
    1.6rem top padding the default table (Interoperability open: 633 px of entries) fits beside the hero's own
    716 px (it grew 1 px); an opened section lengthens the area. Beside the map the filter box is level with the
    "Documentation" title (72 px).
  - home.js `installSideMenus` mirrors the real table into each copy (a MutationObserver re-clones the list after
    every DocFX render) and forwards what the reader does: an expand arrow or a section toggle clicks the real
    entry at the same path (marked `MIRRORED_TOC_CLICK`, so main.js's 48 h memory records it); typing in a copy's
    filter box sets the real box and fires its `input`. Both copies and the docs pages therefore share one state.
  - Measured (headless Chromium): at 1440 / 1920 / 1024 px the band spans the window; each copy lists the real
    table's 60 entries, entry for entry, "Home" active, with no hidden entry; opening "Fundamentals and usage" in
    the top copy opens it in both copies, and docs/intro.html opened next shows it open; typing "torch" in the
    bottom copy filters both copies and the real table to 3 entries while keeping focus; Shift+Tab from the hero
    reaches the top copy's last link; at 390 px the copies are gone and the drawer opens; docs/intro.html keeps its
    282.6 px menu; no console errors. With the band at full width again, phase 5's picture measures 927 / 927 /
    862 / 683 px at 1920 / 1440 / 1280 / 1024, the title still on its axis.

## Phase 2: the editor 20% smaller

- One knob, `--ns-ed-scale: 0.8` on `.ns-editor`: the frame's width, its fixed sizes (tabs, paddings, the button)
  and the font-fitting bounds in `home.js` (`EDITOR_FONT_PX`) all scale by it, so the editor looks as before, at 80%.
- Done. The editor is centered in the hero column at 80% of its width, never under `--ns-ed-min-width: 760px` (or
  the whole column when that is narrower): beside the side menu a 1024 px window leaves a 724 px column, whose 80%
  (579 px) made most rungs' lines scroll at the smallest size. Narrow screens (stacked panes) keep scale 1.
- The fitter now measures rendered lines against the narrower pane minus a pixel (the measuring part of journey5
  `ba338276`, adapted to master's inline gutter; not its layout changes). On the live code, rung 4's C# line
  overflows into a scrollbar at 1440 px (the owner's "one of the crazier side's lines got a scrollbar"); now no
  code line of any rung overflows at 1920 / 1440 / 1280 / 1200 / 1100 / 1024 px (fonts 9.2-12.8 px).

## Phase 3: the title centered above the button

- The caption (`.ns-ed-caption`: the hero's line, then "<topic> k/n <title>") is centered over the stage, so it
  sits directly above the button, with room for the button's breathing scale.
- Done: `text-align: center`, bottom margin `calc(34px * scale)` (the button rises about 19 px above the frame at
  0.8); on narrow screens, where the button has its own line, 14 px. Measured: the caption's text center equals the
  button's center (x = 846 at 1440 px, 195 at 390 px), the gap is 10 px on desktop (11 px on a longer title), 14 px
  on a phone.

## Phase 4: Pick

- The floating button starts as "Pick". Hovering, focusing or clicking it opens six topic buttons around it: four
  at the corners of a box around it and one on each side (left column top / middle / bottom, right column the same).
- Picking a topic shows the topic's first example; the center button becomes "Crazier" and steps through the
  topic's examples (its meter has one cell per example). Hovering "Crazier" (or focusing it) shows the six again,
  the current topic marked. At a topic's last example the center button says "Pick" again.
- Topics come from the page: each `<template data-ns-example>` carries `data-topic="<button label>"`; topics appear
  in document order (slots: top-left, top-right, left, right, bottom-left, bottom-right). Changing what each button
  holds is editing those attributes.
- Placeholder content until the owner picks the examples: the ten live examples plus the hero, grouped into Arrays,
  Text, Random, Linear algebra, Signals, Simulations.
- Keyboard: Tab reaches the ring while it is open, Escape closes it and returns focus to the center button. Touch:
  a tap on "Pick" opens the ring; "Crazier" advances.
- Done. Markup: `.ns-pick` (the button plus `.ns-pick-ring`) replaces the lone button; home.js creates one
  `.ns-pick-opt` per topic. The ring sits on its own soft panel (the hover bridge between options, and legibility
  over the code); options fly out of the button, staggered; the topic on show wears the button's face (lavender,
  cyan edge); the caption fades out while the ring is open (its top corners cover it). "Pick" opens on hover at
  once, "Crazier" after resting 350 ms (so repeated clicks never flash the ring), and a click on "Crazier" keeps it
  closed until the pointer leaves. Measured in headless Chromium: the six options in two columns beside the button
  (top / middle / bottom rows), Pick -> Random 1/2 (label Crazier, meter 10) -> Random 2/2 (label Pick, meter 11);
  resting on Crazier is closed at 150 ms and open at 750 ms with the topic marked; keyboard Tab / Enter / Escape
  path and touch tap / tap-outside path as designed; ring fits 1440, 1024, 800 (stacked) and 390 px; dark theme;
  no console errors.

## Phase 5: integrations

- `.ns-picture` at 70% (`zoom: 0.7`, which the owner tried); the link layer is in percentages of the picture, so it
  follows.
- "Integrations" as the band's heading (h2), centered on the picture's axis above "Where an NDArray can go" (which
  keeps its look and becomes h3).
- Done, on the width instead of zoom (`--ns-picture-scale: 0.7`). Measured: `zoom: 0.7` shrank only the max-width
  (1324 -> 927 px at a 1920 px window, but 1140 -> 927 at 1440, 19% smaller, and at 1024 it GREW 724 -> 738: its
  48 px margin was zoomed and the 100% was not) and set the octagons' "Open Library" / "Open Docs" text at 8.75 px.
  The width gives 927 / 798 / 507 px at 1920 / 1440 / 1024 (70% each) and keeps the buttons at 30 px; narrow
  screens keep the full width. The title's center equals the picture's (x 1086 / 846 / 638); DocFX's anchor.js "#"
  link after it had pushed it 23-29 px off that axis, so the link is out of flow (still shown on hover).
  `tools/gen_integrations_diagram.py --check` passes (the link layer is untouched).

## Final verification

A clean docfx 2.78.4 build of master after phase 5 (`0ed98b11`), served over HTTP, in an isolated headless
Chromium: walking every topic through every example at 1920 / 1440 / 1024 px leaves no code line overflowing (code
9.2-12.8 px); docs/intro.html and docs/interop/index.html keep their 282.6 px menu at 1440 and have no picker (no
other page uses the home page's classes); under `prefers-reduced-motion: reduce` the ring is complete 60 ms after
the hover; no console error on any page or step.

## Open for rediscussion

Collected while executing; the owner reviews them all after phase 5.

1. Side menu width 252 px (fits every top-level entry on one line on Windows); narrower makes "The absolute basics
   for beginners" wrap. The copies hug the window's left edge at any width, while docs pages center their
   container (at 1920 their menu starts at x = 88). The hero and the map are centered beside their copies, so their
   axis is half a menu (126 px) right of the full-width band's.
2. Between 768 and 991 px the copies stay columns; the hero then has 516-739 px and uses its stacked (narrow)
   layout. The two copies share one state (open sections, filter text) with each other and with the docs pages.
3. The 20% applies to the editor's width and everything inside it (code, tabs, output, button), not to the caption
   above it. The editor is centered in the hero column, under the left-aligned headline.
4. FizzBuzz's printed line (72 characters) still scrolls in its terminal at 1280 px and below, at the smallest
   output size (9.2 px); master's design scrolls a line that cannot fit (journey5 wraps it instead). Placeholder
   content, so it may go away with the owner's picks.
5. "The picked gets replaced by the Crazier" was read as the center button: "Pick" becomes "Crazier" after a pick,
   and the picked topic stays in the ring, marked. The other reading (the picked topic's own slot turns into
   "Crazier") is a small change in `sync()`.
6. At a topic's last example the button says "Pick" again (the old ladder said "Start over"); picking the same
   topic restarts it.
7. Hover timing: "Pick" opens the ring at once, "Crazier" after 350 ms of rest (HOVER_INTENT_MS), and the ring
   closes 220 ms after the pointer leaves (CLOSE_DELAY_MS).
8. Placeholder topics (the owner picks the examples next): Arrays (the hero again, broadcast table), Text (argsort,
   FizzBuzz), Random (seed 42, Monte Carlo pi), Linear algebra (einsum distances, Markov chain), Signals (FFT, the
   only one-example topic), Simulations (Game of Life, Mandelbrot). Topic names are the button labels.
9. On touch screens there is no hover: "Pick" opens the ring on tap, "Crazier" only steps, and the ring comes back
   at a topic's end ("Pick").
10. The picture shrinks by width, not zoom (measurements above). Trade-off: the octagons sit closer together, and
    a box's full-size "Open Library" / "Open Docs" pair now reaches over its neighbor box (ONNX Runtime's over
    ML.NET); zoom would shrink the buttons with the picture but to 8.75 px text, and would not shrink the picture
    at all below about 1500 px.
11. "Integrations" is a large centered title (51 px at 1920, with a soft neon glow) over the existing heading row,
    so the band now has two big headings; an alternative is a small centered label, or dropping "Where an NDArray
    can go" into the paragraph.
