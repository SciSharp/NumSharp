"""Resolve coverage links from an offline snapshot of NumPy's official Sphinx index.

The snapshot contains API identifiers and URLs, not documentation text. It keeps
coverage generation deterministic and offline while the documentation audit checks
the resulting links against the live latest-stable index. Refresh it explicitly:

    python coverage/audit_documentation.py --refresh-inventory

NumPy publishes some APIs on generated pages, others as anchors in topic pages,
and some runtime exports only on their parent documentation page. Never fabricate
a generated page for the latter: a real parent link is more useful than a 404.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DOCS_BASE_URL = "https://numpy.org/doc/stable/"
SNAPSHOT_PATH = Path(__file__).with_name("numpy_documentation_inventory.json")

# Public aliases whose canonical documentation uses another spelling/namespace.
API_ALIASES = {
    "numpy.abs": "numpy.absolute",
    "numpy.bitwise_not": "numpy.invert",
    "numpy.row_stack": "numpy.vstack",
    "numpy.test": "numpy.testing",
    "numpy.ndarray.interop": "numpy.ndarray",
    "numpy.polynomial.Polynomial": "numpy.polynomial.polynomial.Polynomial",
    "numpy.polynomial.Chebyshev": "numpy.polynomial.chebyshev.Chebyshev",
    "numpy.polynomial.Legendre": "numpy.polynomial.legendre.Legendre",
    "numpy.polynomial.Hermite": "numpy.polynomial.hermite.Hermite",
    "numpy.polynomial.HermiteE": "numpy.polynomial.hermite_e.HermiteE",
    "numpy.polynomial.Laguerre": "numpy.polynomial.laguerre.Laguerre",
}


@lru_cache(maxsize=1)
def documentation_index() -> dict[str, str]:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or not snapshot.get("api_urls"):
        raise ValueError(f"Invalid NumPy documentation snapshot: {SNAPSHOT_PATH}")
    links: dict[str, str] = snapshot["api_urls"]
    # A few Sphinx object names retain private implementation modules while their
    # URL anchors use the public name (notably testing.suppress_warnings).
    for url in tuple(links.values()):
        anchor = url.partition("#")[2]
        if anchor.startswith("numpy."):
            links.setdefault(anchor, url)
    return links


def documentation_url(surface: str, name: str, kind: str = "function") -> str:
    """Return a verified API URL or its nearest documented parent for any surface."""
    prefix = "numpy" if surface in {"np", "numpy"} else "numpy." + surface
    api_id = f"{prefix}.{name}" if name else prefix
    for alias, canonical in API_ALIASES.items():
        if api_id == alias or api_id.startswith(alias + "."):
            api_id = canonical + api_id[len(alias):]
            break

    links = documentation_index()
    candidate = api_id
    while candidate.startswith("numpy."):
        if candidate in links:
            return DOCS_BASE_URL + links[candidate]
        candidate = candidate.rpartition(".")[0]
    return DOCS_BASE_URL + "reference/index.html"
