# Original Karpathy numerical test fixtures

These six files retain the exact previously fetched source bytes. Filenames follow
`GitHubHandle.GistName.py`; [sources.json](sources.json) records original URLs,
acquisition paths, SHA256 pins and observed license declarations. They are source
fixtures, not executable entry points or new NumSharp implementations.

The interop test assembly embeds them so a test run does not depend on an ephemeral
`outputs/` folder or network retrieval. `KarpathyOriginalSource` checks every hash,
selects an explicit list of numerical class/function definitions with Python AST,
and never executes the original top-level downloads, imports or application loops.
Compatibility edits are confined to that loader and recorded explicitly. The raw
fixtures themselves are unchanged. Do not run these files as standalone programs.

Keep the provenance and license observations with these research/test fixtures;
public availability or attribution does not establish a redistribution license.
