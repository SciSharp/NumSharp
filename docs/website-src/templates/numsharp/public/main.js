// NumSharp DocFX Template Customization
// https://dotnet.github.io/docfx/docs/template.html
//
// The modern template auto-imports this file's default export via its
// `options()` hook (docfx.min.js -> import('./main.js')). This is the
// supported way to add site JavaScript without forking docfx.min.js.
//
// Beyond the navbar icon links, this adds ONE behavior: it remembers which
// #toc sections the reader expanded/collapsed and restores them on the next
// page, for a 48h window that slides forward on every visit. See
// installTocStatePersistence below.

/**
 * How long a remembered TOC layout survives with no visits, in milliseconds
 * (48 hours). Every read/write of the record re-stamps this from "now", so the
 * window is sliding: the state only expires after 48h of *inactivity*.
 * @type {number}
 */
const TOC_STATE_TTL_MS = 48 * 60 * 60 * 1000

/**
 * Storage-format version. Bump to invalidate every reader's saved TOC state at
 * once when the record shape below changes (a mismatched version is discarded
 * as if expired), so an old blob can never be misread as the new shape.
 * @type {number}
 */
const TOC_STATE_SCHEMA = 1

/**
 * Separator joining ancestor labels into a node key. U+203A (›) is chosen
 * because it does not occur in these TOC titles; a title that contained it
 * could collide two distinct nodes onto one key (a cosmetic, non-fatal glitch).
 * @type {string}
 */
const TOC_KEY_SEP = '›'

/**
 * Wire up 48h sliding-window persistence of the modern-template TOC's
 * expand/collapse state onto `localStorage`.
 *
 * Why this is not a one-liner: the modern template keeps each node's expanded
 * flag in a private JavaScript closure and re-renders the ENTIRE `#toc` subtree
 * on every toggle (lit-html). So we cannot simply set the `.expanded` CSS class
 * on restore — the next toggle of any node would regenerate the tree from
 * docfx's own model and wipe our change. Instead we drive docfx's model by
 * synthesizing clicks on the nodes whose remembered state differs from what
 * docfx rendered, which is the only state the template treats as authoritative.
 *
 * Safe to call before `#toc` is populated: population is asynchronous (after
 * toc.json is fetched), so a MutationObserver defers the restore until the
 * first expander node appears, and the function no-ops when there is no `#toc`
 * or no usable `localStorage` (private mode, file://, the PDF renderer).
 *
 * @returns {void}
 */
function installTocStatePersistence() {
  // localStorage access itself can throw (blocked cookies/site-data), so probe
  // it behind try/catch and bail rather than breaking the rest of page init.
  let store
  try {
    store = window.localStorage
  } catch {
    return
  }
  if (!store) {
    return
  }

  const toc = document.getElementById('toc')
  if (!toc) {
    return
  }

  // Scope the record to THIS page's toc.json. Two reasons: (1) folders with
  // their own toc keep independent state; (2) localStorage is per-ORIGIN, not
  // per-path, so NumSharp and OptunaSharp docs served from the same
  // *.github.io host would otherwise share (and clobber) one record.
  const tocRelMeta = document.querySelector('meta[name="docfx:tocrel"]')
  const tocRel = tocRelMeta ? tocRelMeta.content || '' : ''
  let scope
  try {
    scope = new URL(tocRel, window.location.href).href
  } catch {
    scope = tocRel
  }
  const storageKey = 'docfx:tocExpanded:' + scope

  /**
   * Read the remembered per-node states, discarding an expired, malformed, or
   * wrong-version record (any of which is treated as "nothing remembered").
   * @returns {{[nodeKey: string]: boolean}} map of node key -> expanded flag; empty when nothing valid is stored.
   */
  function loadNodes() {
    let raw
    try {
      raw = store.getItem(storageKey)
    } catch {
      return {}
    }
    if (!raw) {
      return {}
    }
    let data
    try {
      data = JSON.parse(raw)
    } catch {
      // A corrupt blob is unrecoverable; drop it so it cannot keep erroring.
      removeRecord()
      return {}
    }
    const expired = !data ||
      data.v !== TOC_STATE_SCHEMA ||
      typeof data.expiresAt !== 'number' ||
      data.expiresAt < Date.now()
    if (expired) {
      removeRecord()
      return {}
    }
    return data.nodes && typeof data.nodes === 'object' ? data.nodes : {}
  }

  /**
   * Persist the per-node states and (re-)stamp the 48h expiry from now, which
   * is what makes the window slide forward on every visit and every toggle.
   * @param {{[nodeKey: string]: boolean}} nodeStates map of node key -> expanded flag to persist.
   * @returns {void}
   */
  function saveNodes(nodeStates) {
    const record = { v: TOC_STATE_SCHEMA, expiresAt: Date.now() + TOC_STATE_TTL_MS, nodes: nodeStates }
    try {
      store.setItem(storageKey, JSON.stringify(record))
    } catch {
      // Quota/availability failures are non-fatal: the TOC still works, it just
      // won't remember state this time.
    }
  }

  /** Delete this scope's record; used when it is expired or corrupt. @returns {void} */
  function removeRecord() {
    try {
      store.removeItem(storageKey)
    } catch {
      // Ignore: nothing we can do, and it must not break page init.
    }
  }

  /** Remembered node states for this scope, loaded once and mutated on toggles. */
  const nodeStates = loadNodes()

  /**
   * The label text of an `<li>`'s own node (its direct `<a>`, or the
   * `name-only` span for group nodes that have no link). Deliberately excludes
   * the empty `.expand-stub` span and the nested child `<ul>` so the key is the
   * node's title alone. `<wbr>` word-break elements the template injects carry
   * no text, so `textContent` is the clean title.
   * @param {Element} li the TOC list item.
   * @returns {string} the node's own label, trimmed (empty string if none found).
   */
  function labelOf(li) {
    const anchor = li.querySelector(':scope > a')
    if (anchor) {
      return anchor.textContent.trim()
    }
    const nameSpan = li.querySelector(':scope > span.name-only')
    return nameSpan ? nameSpan.textContent.trim() : ''
  }

  /**
   * A stable identity for a TOC node: the chain of ancestor labels from the
   * root down to it. This survives docfx's full re-render on every toggle (the
   * `<li>` elements are recreated but their titles are not) and is identical
   * across pages that share one toc.json — which is exactly what lets a choice
   * made on one page be restored on the next.
   * @param {Element} li the TOC list item to identify.
   * @returns {string} the ancestor-label path, joined by {@link TOC_KEY_SEP}.
   */
  function keyOf(li) {
    const parts = []
    // Walk up to (but not including) the #toc container, collecting each <li>.
    for (let cur = li; cur && cur !== toc; cur = cur.parentElement) {
      if (cur.tagName === 'LI') {
        parts.unshift(labelOf(cur))
      }
    }
    return parts.join(TOC_KEY_SEP)
  }

  /**
   * True while we are synthesizing restore clicks, so the click listener can
   * tell our own programmatic toggles apart from real reader interaction and
   * not re-record them.
   * @type {boolean}
   */
  let applyingRestore = false

  /**
   * Find the first expander whose remembered state disagrees with what docfx
   * currently renders. Nodes with no remembered state are skipped, so docfx's
   * defaults (e.g. auto-expanding the active page's path) are preserved for
   * everything the reader never touched.
   * @returns {Element|null} the mismatched `<li>`, or null when the DOM already matches memory.
   */
  function findMismatchedExpander() {
    const expanders = toc.querySelectorAll('li.expander')
    for (const li of expanders) {
      const key = keyOf(li)
      if (!(key in nodeStates)) {
        continue
      }
      const isExpanded = li.classList.contains('expanded')
      if (isExpanded !== nodeStates[key]) {
        return li
      }
    }
    return null
  }

  /**
   * Reconcile the rendered TOC with the remembered states by clicking the
   * `.expand-stub` of each mismatched node. Each click triggers a synchronous
   * re-render that replaces the elements, so we re-query every pass rather than
   * hold stale references. Toggling one node never changes another node's
   * rendered state (docfx re-renders each from its own model), so every pass
   * fixes exactly one node and the loop makes monotonic progress; the budget is
   * a safety cap against a pathological tree, not the expected exit.
   * @returns {void}
   */
  function restore() {
    if (Object.keys(nodeStates).length === 0) {
      return
    }
    applyingRestore = true
    try {
      const budget = toc.querySelectorAll('li.expander').length * 2 + 8
      for (let pass = 0; pass < budget; pass++) {
        const li = findMismatchedExpander()
        if (!li) {
          break
        }
        const stub = li.querySelector(':scope > .expand-stub')
        if (!stub) {
          // No toggle handle to drive docfx's model; stop rather than spin.
          break
        }
        stub.click()
      }
    } finally {
      applyingRestore = false
    }
  }

  // Record real reader toggles. Capture phase on the persistent #toc container
  // runs before docfx's own bubble-phase @click handler, so we can identify the
  // node while its element is still attached, then read the ACTUAL resulting
  // state after docfx re-renders on the next frame (storing ground truth rather
  // than a prediction of the toggle).
  toc.addEventListener('click', event => {
    // Ignore our restore clicks (untrusted) and any stray untrusted events.
    if (applyingRestore || !event.isTrusted) {
      return
    }
    const target = event.target
    if (!(target instanceof Element)) {
      return
    }
    // A toggle is either the expand caret, or a group node's own href='#' link;
    // a real navigation link (href to a page) is not a toggle and is ignored.
    const onStub = target.closest('.expand-stub')
    const anchor = target.closest('a')
    const onGroupToggle = anchor && anchor.getAttribute('href') === '#'
    if (!onStub && !onGroupToggle) {
      return
    }
    const li = target.closest('li.expander')
    if (!li) {
      return
    }
    const key = keyOf(li)
    // docfx re-renders synchronously in its handler (which runs after this
    // capture handler); rAF fires after that, so the DOM reflects the new state.
    requestAnimationFrame(() => {
      let current = null
      const expanders = toc.querySelectorAll('li.expander')
      for (const el of expanders) {
        if (keyOf(el) === key) {
          current = el
          break
        }
      }
      if (!current) {
        return
      }
      nodeStates[key] = current.classList.contains('expanded')
      saveNodes(nodeStates)
    })
  }, true)

  /**
   * True once we have restored for this page load, so the MutationObserver's
   * repeated firings (and our own restore-driven mutations) do not re-run it.
   * @type {boolean}
   */
  let booted = false

  /**
   * Restore once the TOC has rendered its first expander, then slide the 48h
   * window forward for this visit. Idempotent and cheap to call repeatedly.
   * @returns {void}
   */
  function boot() {
    if (booted) {
      return
    }
    if (!toc.querySelector('li.expander')) {
      // TOC not populated yet; wait for the next mutation.
      return
    }
    booted = true
    // Stop observing BEFORE restoring so our own restore mutations don't re-fire
    // this callback.
    observer.disconnect()
    restore()
    // Re-stamp the expiry on every visit, even when nothing was toggled, so
    // simply browsing keeps a remembered layout alive.
    if (Object.keys(nodeStates).length > 0) {
      saveNodes(nodeStates)
    }
  }

  // #toc is filled asynchronously after toc.json loads, so observe it for the
  // first expander. Also call boot() immediately in case the TOC was already
  // rendered before this ran.
  const observer = new MutationObserver(boot)
  observer.observe(toc, { childList: true, subtree: true })
  boot()
}

export default {
  // Icon links displayed in the navbar (top-right)
  iconLinks: [
    {
      icon: 'github',
      href: 'https://github.com/SciSharp/NumSharp',
      title: 'GitHub'
    },
    {
      icon: 'box-seam',
      href: 'https://www.nuget.org/packages/NumSharp',
      title: 'NuGet'
    }
  ],

  // Default theme: 'light', 'dark', or 'auto' (system preference)
  // defaultTheme: 'auto',

  // Startup script (runs when page loads)
  start: () => {
    // Remember TOC expand/collapse state for 48h (sliding window). See
    // installTocStatePersistence above.
    installTocStatePersistence()
  },

  // Customize syntax highlighting (highlight.js)
  // configureHljs: (hljs) => {
  //   // Register additional languages or customize
  // }
}
