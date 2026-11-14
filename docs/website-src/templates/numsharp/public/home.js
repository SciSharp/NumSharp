// NumSharp docs home page behavior (docs/website-src/index.md).
//
// main.js imports this module and calls installHomePage() from the modern
// template's start hook on every page; it returns at once on pages without the
// home page's editor, so other pages pay one querySelector.
//
// Three behaviors, all progressive: without JavaScript the page still shows the
// install command, the first Python/C# pair with its output, and every link.
//   1. The Copy button next to `dotnet add package NumSharp`.
//   2. The split editor: syntax colors for both languages, a font size that
//      fits the longest line of the shown pair, and the "Crazier" button that
//      steps through ten wilder pairs (each verified to print exactly what
//      NumPy prints) and back to the first.
//   3. Nothing for the integrations picture: its link layer is plain HTML and
//      CSS written by tools/gen_integrations_diagram.py.

/**
 * Python tokens the editor colors, as one global, multiline regex whose
 * capture groups name the token kind: 1 comment, 2 string, 3 number,
 * 4 keyword. Order matters: a comment or string swallows any keyword-looking
 * text inside it. Numbers accept `_` separators and the `j` suffix of complex
 * literals; `\b` keeps digits inside names (`int64`, `rfft2`) uncolored.
 * @type {RegExp}
 */
const PYTHON_TOKENS = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?j?)\b|\b(import|from|as|for|in|if|elif|else|while|def|return|lambda|and|or|not|is|None|True|False|with|class|pass|yield)\b/gm

/**
 * C# tokens the editor colors, with the same group numbering as
 * {@link PYTHON_TOKENS}. Number suffixes (`f`, `d`, `m`, `L`, `u`) belong to
 * the literal.
 * @type {RegExp}
 */
const CSHARP_TOKENS = /(\/\/.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[fFdDmMlLuU]?)\b|\b(using|var|new|for|foreach|in|if|else|while|return|int|long|double|float|bool|string|true|false|null|class|static|void|public|private)\b/gm

/**
 * The token class each capture group of the token regexes maps to; the
 * stylesheet colors `.ns-tok-c` (comment), `.ns-tok-s` (string), `.ns-tok-n`
 * (number) and `.ns-tok-k` (keyword).
 * @type {string[]}
 */
const TOKEN_CLASSES = ['', 'c', 's', 'n', 'k']

/**
 * The editor never sets code smaller or larger than these sizes, in px. Below
 * the minimum a long line scrolls inside its pane instead of shrinking into
 * illegibility; above the maximum the code would outgrow the hero's type.
 * @type {{min: number, max: number}}
 */
const EDITOR_FONT_PX = { min: 11.5, max: 16 }

/**
 * Line-number gutter width plus the gap after it, in character widths. It must
 * match the stylesheet's `.ns-ed-row code::before` (3ch wide, 1.2ch margin),
 * because the font fitting counts it as part of every line.
 * @type {number}
 */
const GUTTER_CH = 4.2

/**
 * How long the Copy button says "Copied" (or "Press Ctrl+C") before it says
 * "Copy" again, in milliseconds.
 * @type {number}
 */
const COPY_FEEDBACK_MS = 1600

/**
 * Escape text for insertion as HTML element content.
 * @param {string} text raw source text.
 * @returns {string} the text with `&`, `<` and `>` replaced by entities.
 */
function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/**
 * Color one line of source code.
 * @param {string} text the line, as plain text.
 * @param {'python'|'csharp'} language which token set to use.
 * @returns {string} HTML: the escaped line with each token wrapped in a
 *   `span.ns-tok-*`.
 */
function highlightLine(text, language) {
  const tokens = language === 'python' ? PYTHON_TOKENS : CSHARP_TOKENS
  // The regexes are shared and global, so their scan position must start over
  // for every line.
  tokens.lastIndex = 0
  let html = ''
  let last = 0
  let match
  while ((match = tokens.exec(text)) !== null) {
    const group = match.findIndex((value, index) => index > 0 && value !== undefined)
    html += escapeHtml(text.slice(last, match.index))
    html += `<span class="ns-tok-${TOKEN_CLASSES[group]}">${escapeHtml(match[0])}</span>`
    last = tokens.lastIndex
  }
  return html + escapeHtml(text.slice(last))
}

/**
 * Color every row of the editor body: the first cell of a row is Python, the
 * second C#. Safe to call twice on the same row, because it reads textContent.
 * @param {Element} body the `[data-ns-body]` element.
 * @returns {void}
 */
function highlightBody(body) {
  for (const row of body.querySelectorAll('.ns-ed-row')) {
    const [python, csharp] = row.querySelectorAll('code')
    if (python) python.innerHTML = highlightLine(python.textContent, 'python')
    if (csharp) csharp.innerHTML = highlightLine(csharp.textContent, 'csharp')
  }
}

/**
 * Wire the Copy button beside the install command.
 *
 * The clipboard API needs a secure context and permission; where it is missing
 * or refused, the command's own text is selected and the button says
 * "Press Ctrl+C", so the reader can still copy it in one keystroke.
 * @param {HTMLButtonElement} button the button, with the text to copy in
 *   `data-ns-copy`.
 * @returns {void}
 */
function installCopyButton(button) {
  const label = button.textContent
  let timer = 0

  /** Show a short confirmation on the button, then restore its label. @param {string} text @returns {void} */
  function say(text) {
    button.textContent = text
    clearTimeout(timer)
    timer = setTimeout(() => { button.textContent = label }, COPY_FEEDBACK_MS)
  }

  /** Select the command text so a keyboard copy takes it. @returns {void} */
  function selectCommand() {
    const code = button.parentElement && button.parentElement.querySelector('code')
    const selection = window.getSelection()
    if (!code || !selection) return
    const range = document.createRange()
    range.selectNodeContents(code)
    selection.removeAllRanges()
    selection.addRange(range)
  }

  button.addEventListener('click', () => {
    const text = button.dataset.nsCopy || ''
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      selectCommand()
      say('Press Ctrl+C')
      return
    }
    navigator.clipboard.writeText(text).then(
      () => say('Copied'),
      () => { selectCommand(); say('Press Ctrl+C') })
  })
}

/**
 * Size the editor's code so the longest line of the shown pair fits its pane,
 * and the printed result so its longest line fits its terminal.
 *
 * One code size serves both panes, so a line and its translation always share
 * a size. The result gets its own size (`--ns-ed-out-font`), never larger than
 * the code's, because a wide printed array (FizzBuzz's 73 characters) would
 * otherwise shrink programs whose lines are short. The character width is
 * measured in the font the browser actually picked from the monospace stack,
 * because the fallbacks differ (Consolas is narrower than JetBrains Mono).
 * @param {HTMLElement} frame the `.ns-ed-frame` element.
 * @returns {void}
 */
function fitEditorFont(frame) {
  const cell = frame.querySelector('.ns-ed-row code')
  const output = frame.querySelector('[data-ns-out]')
  if (!cell || !output) return

  // Measure one character at a known size; a fixed-pitch font makes ten
  // identical glyphs exactly ten advances wide.
  const probe = document.createElement('span')
  probe.textContent = '0000000000'
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;font-size:100px'
  cell.appendChild(probe)
  const charPerPx = probe.getBoundingClientRect().width / 10 / 100
  probe.remove()
  if (!(charPerPx > 0)) return

  let codeChars = 0
  for (const code of frame.querySelectorAll('.ns-ed-row code')) {
    codeChars = Math.max(codeChars, code.textContent.length)
  }
  let outputChars = 0
  for (const pre of frame.querySelectorAll('[data-ns-out]')) {
    for (const line of pre.textContent.split('\n')) outputChars = Math.max(outputChars, line.length)
  }

  // Available text width: the cell's content box (padding excluded) for code,
  // the output block's content box for the printed result.
  const style = getComputedStyle(cell)
  const codeWidth = cell.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
  const outStyle = getComputedStyle(output)
  const outWidth = output.clientWidth - parseFloat(outStyle.paddingLeft) - parseFloat(outStyle.paddingRight)
  /**
   * Keep a fitted size within {@link EDITOR_FONT_PX} and round it down to a
   * tenth of a pixel, so a fit never overshoots its pane by a rounding error.
   * @param {number} px the size that would exactly fit, in px.
   * @returns {number} the size to apply, in px.
   */
  const clamp = px => Math.floor(Math.max(EDITOR_FONT_PX.min, Math.min(EDITOR_FONT_PX.max, px)) * 10) / 10
  const codeSize = clamp(codeWidth / ((codeChars + GUTTER_CH) * charPerPx))
  const outSize = outputChars > 0 ? clamp(Math.min(codeSize, outWidth / (outputChars * charPerPx))) : codeSize
  frame.style.setProperty('--ns-ed-font', `${codeSize}px`)
  frame.style.setProperty('--ns-ed-out-font', `${outSize}px`)
}

/**
 * One pair the editor can show.
 * @typedef {object} EditorExample
 * @property {string} title what the pair does, for the caption (empty for the
 *   first pair, which has its own caption).
 * @property {Node[]} rows the `.ns-ed-row` elements, uncolored.
 * @property {string} output what both programs print.
 */

/**
 * Wire the split editor and its "Crazier" button.
 *
 * The first pair is whatever the page renders inside `[data-ns-body]`, so it
 * shows without JavaScript; the wilder pairs are inert `<template
 * data-ns-example>` elements. The button walks the ladder one rung per click
 * and, on the last rung, offers "Start over" back to the first pair. The button
 * stays hidden (its `hidden` attribute) when the page has no templates, so it
 * never offers an action that does nothing.
 * @param {HTMLElement} editor the `[data-ns-editor]` element.
 * @returns {void}
 */
function installEditor(editor) {
  const frame = editor.querySelector('.ns-ed-frame')
  const body = editor.querySelector('[data-ns-body]')
  const outputs = Array.from(editor.querySelectorAll('.ns-ed-run [data-ns-out]'))
  const caption = editor.querySelector('[data-ns-caption]')
  const button = editor.querySelector('[data-ns-crazier]')
  const label = editor.querySelector('[data-ns-crazier-label]')
  const meter = editor.querySelector('[data-ns-meter]')
  if (!frame || !body || outputs.length === 0) return

  /** @type {EditorExample[]} */
  const examples = [{
    title: '',
    rows: Array.from(body.querySelectorAll('.ns-ed-row')).map(row => row.cloneNode(true)),
    output: outputs[0].textContent
  }]
  for (const template of editor.querySelectorAll('template[data-ns-example]')) {
    const out = template.content.querySelector('[data-ns-out]')
    examples.push({
      title: template.dataset.title || '',
      rows: Array.from(template.content.querySelectorAll('.ns-ed-row')),
      output: out ? out.textContent : ''
    })
  }
  const firstCaption = caption ? caption.innerHTML : ''
  const top = examples.length - 1

  highlightBody(body)
  fitEditorFont(frame)
  // Refit when the panes change width (window resize, the panes stacking on a
  // narrow screen) and when a late web font changes the character width.
  new ResizeObserver(() => fitEditorFont(frame)).observe(frame)
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => fitEditorFont(frame))

  if (!button || !label || !meter || top < 1) return

  // One cell per rung, filled up to the current one.
  meter.replaceChildren(...examples.slice(1).map(() => document.createElement('i')))
  let level = 0

  /**
   * Show one rung: its rows, its output in both terminals, the caption, the
   * meter and the button's next action.
   * @param {number} next the index into `examples`.
   * @returns {void}
   */
  function show(next) {
    level = next
    const example = examples[level]
    body.replaceChildren(...example.rows.map(row => row.cloneNode(true)))
    highlightBody(body)
    for (const pre of outputs) pre.textContent = example.output
    if (caption) {
      caption.innerHTML = level === 0
        ? firstCaption
        : `<b>Crazy ${level}/${top}</b> ${escapeHtml(example.title)}`
    }
    meter.querySelectorAll('i').forEach((cell, index) => cell.classList.toggle('on', index < level))
    label.textContent = level === top ? 'Start over' : 'Crazier'
    fitEditorFont(frame)
    // Restart the swap animation: removing the class, forcing a style flush,
    // and adding it again replays it even on two quick clicks.
    frame.classList.remove('ns-ed-swap')
    void frame.offsetWidth
    frame.classList.add('ns-ed-swap')
  }

  button.hidden = false
  button.addEventListener('click', () => show(level === top ? 0 : level + 1))
}

/**
 * Install the home page's behaviors when this page is the home page.
 * @returns {void}
 */
export function installHomePage() {
  const editor = document.querySelector('[data-ns-editor]')
  if (!editor) return
  for (const button of document.querySelectorAll('[data-ns-copy]')) installCopyButton(button)
  installEditor(editor)
}
