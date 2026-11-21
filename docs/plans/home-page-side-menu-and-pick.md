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
| 1 | The docs side menu on the home page, narrower | pending |
| 2 | The NumPy / C# editor 20% smaller | pending |
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

## Phase 2: the editor 20% smaller

- One knob, `--ns-ed-scale: 0.8` on `.ns-editor`: the frame's width, its fixed sizes (tabs, paddings, the button)
  and the font-fitting bounds in `home.js` (`EDITOR_FONT_PX`) all scale by it, so the editor looks as before, at 80%.

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
