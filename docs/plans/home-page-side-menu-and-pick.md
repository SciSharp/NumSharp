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
| 1 | The docs side menu on the home page, narrower | done: 252 px, sticky; phone drawer from the action bar |
| 2 | The NumPy / C# editor 20% smaller | done: 912 px (was 1140), code at most 12.8 px (was 16) |
| 3 | The example title centered above the button | pending |
| 4 | "Pick": six topic buttons around the button, then "Crazier" inside the picked topic | pending |
| 5 | The integrations picture 30% smaller, "Integrations" centered above the section | pending |

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

## Phase 5: integrations

- `.ns-picture` at 70% (`zoom: 0.7`, which the owner tried); the link layer is in percentages of the picture, so it
  follows.
- "Integrations" as the band's heading (h2), centered on the picture's axis above "Where an NDArray can go" (which
  keeps its look and becomes h3).

## Open for rediscussion

Collected while executing; the owner reviews them all after phase 5.

1. Side menu width 252 px (fits every top-level entry on one line on Windows); narrower makes "The absolute basics
   for beginners" wrap. The menu hugs the window's left edge at any width, while docs pages center their container
   (at 1920 their menu starts at x = 88).
2. Between 768 and 991 px the menu stays a column, as on docs pages; the hero then has 516-739 px and uses its
   stacked (narrow) layout.
3. The 20% applies to the editor's width and everything inside it (code, tabs, output, button), not to the caption
   above it. The editor is centered in the hero column, under the left-aligned headline.
4. FizzBuzz's printed line (72 characters) still scrolls in its terminal at 1280 px and below, at the smallest
   output size (9.2 px); master's design scrolls a line that cannot fit (journey5 wraps it instead). Placeholder
   content, so it may go away with the owner's picks.
