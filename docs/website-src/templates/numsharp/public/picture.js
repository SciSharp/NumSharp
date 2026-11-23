// NumSharp docs: the integrations picture's links, on every page that shows it.
//
// The picture with its link layer is one file, includes/integrations-picture.md,
// which two pages include: the home page's integrations band (index.md) and
// the Interoperability guide (docs/interop/index.md). Its hover rings and
// "Open Library" / "Open Docs" buttons are HTML and CSS alone (main.css, "The
// integrations picture"); this module only fixes what one shared copy cannot
// say per page. Its links are written once, from the site's root, so they lead
// to the same pages from both: the picture's own link and NumSharp's octagon
// lead to the guide. On the guide that is the page the reader is on, and a
// click would only reload it, so there:
//   - the picture's own link opens the picture at full size in a new tab, as
//     DocFX's template does for any large image that is not already a link
//     (which is what the guide's plain image did before it was shared);
//   - any other such link (NumSharp's octagon and its "Open Docs") opens the
//     docs home page, whose band shows the same picture.
// A link with a #fragment is a jump within the page, not a reload, and stays.
// Without JavaScript every link stays as written: on the guide, those reload
// the guide, which is harmless.
//
// main.js imports this module and calls installPictures() from the modern
// template's start hook on every page; a page without the picture pays one
// querySelectorAll.

/**
 * The text the picture link's visually hidden purpose span carries on the
 * guide, where the link opens the picture itself. It starts with a space
 * because a screen reader reads it right after the image's alt text, as the
 * shared file's own " Opens the interoperability guide." does.
 * @type {string}
 */
const FULL_SIZE_PURPOSE = ' Opens the picture at full size, in a new tab.'

/**
 * The identity of the page an address loads, for telling a reload from a
 * navigation: the origin and the path with `index.html`, a `.html` extension
 * and a trailing slash removed, in lower case. GitHub Pages serves
 * `docs/interop/`, `docs/interop/index.html` and `docs/interop/index` as one
 * page, and main.js's table-of-contents code treats them alike the same way.
 * The query and the fragment are not part of it; the caller decides what a
 * fragment means.
 * @param {URL} url an absolute address.
 * @returns {string} a key that two addresses of the same page share.
 */
function pageKey(url) {
  const path = url.pathname
    .replace(/\/index(\.html)?$/i, '/')
    .replace(/\.html$/i, '')
    .replace(/\/+$/, '')
  return (url.origin + path).toLowerCase()
}

/**
 * The docs home page's address, from the `docfx:rel` meta tag the modern
 * template writes on every page: the path from this page up to the site's
 * root (`../../` on the guide, empty on the home page itself).
 * @returns {URL} the absolute address of the site's index.html.
 */
function homeAddress() {
  const rel = document.querySelector('meta[name="docfx:rel"]')
  // A page without the tag (none is known) resolves against itself, which on
  // the site's own folder layout is still the nearest index.html.
  return new URL((rel ? rel.content || '' : '') + 'index.html', window.location.href)
}

/**
 * Point the links of one picture that would only reload the page they sit on
 * elsewhere: the picture's own link to the picture at full size in a new tab,
 * any other one to the docs home page. Links to other pages, and links with a
 * fragment (a jump within this page), are left alone.
 * @param {Element} picture a `.ns-picture` container: the picture's own link
 *   (`.ns-picture-link`, holding the `<img>` and its `[data-ns-picture-purpose]`
 *   span) and the generated link layer.
 * @param {string} here the current page's {@link pageKey}.
 * @param {URL} home the docs home page's address ({@link homeAddress}).
 * @returns {void}
 */
function retargetSelfLinks(picture, here, home) {
  for (const link of picture.querySelectorAll('a[href]')) {
    // link.href is the browser's absolute resolution of the attribute, so the
    // comparison sees exactly the address a click would load. A malformed href
    // leaves it unparsable as a URL: skip that link rather than stop the rest.
    let target
    try {
      target = new URL(link.href)
    } catch {
      continue
    }
    if (target.hash || pageKey(target) !== here) continue
    if (link.classList.contains('ns-picture-link')) {
      const img = link.querySelector('img')
      // No image means a hand edit broke the shared file; a link to nothing
      // useful is better left as written than pointed at an empty address.
      if (!img || !img.src) continue
      // img.src is already absolute: the address DocFX resolved for this page.
      link.href = img.src
      link.target = '_blank'
      link.rel = 'noopener'
      // Screen readers name the link by the image's alt text plus this span,
      // so the stated purpose must follow the new target.
      const purpose = link.querySelector('[data-ns-picture-purpose]')
      if (purpose) purpose.textContent = FULL_SIZE_PURPOSE
    } else {
      link.href = home.href
    }
  }
}

/**
 * Fix the integrations picture's links on this page, wherever it is shown.
 * Returns at once on a page without the picture.
 * @returns {void}
 */
export function installPictures() {
  const pictures = document.querySelectorAll('.ns-picture')
  if (pictures.length === 0) return
  const here = pageKey(new URL(window.location.href))
  const home = homeAddress()
  for (const picture of pictures) retargetSelfLinks(picture, here, home)
}
