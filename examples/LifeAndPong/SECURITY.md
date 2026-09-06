# Security policy

Life Arcade is maintained as an example within the
[SciSharp/NumSharp repository](https://github.com/SciSharp/NumSharp). It has no
independent release or security-support lifecycle documented at this time.

## Report a concern

For a build, packaging, local-data, or runtime issue that is safe to discuss
publicly, use the upstream
[Life Arcade bug form](https://github.com/SciSharp/NumSharp/issues/new?template=life-arcade.yml).
Include:

- the output of `git rev-parse HEAD`, or the ZIP digest from `SHA256SUMS`;
- Windows version, architecture, and .NET SDK version when building from source;
- concise reproduction steps and the expected and actual result; and
- sanitized logs or screenshots with credentials, personal information, and
  private filesystem paths removed.

Do not publish exploit details, secrets, or personal data in a public issue. If
a concern requires private disclosure, first check the upstream repository's
[Security page](https://github.com/SciSharp/NumSharp/security) for a currently
advertised private reporting route. This subproject does not claim that GitHub
private vulnerability reporting is enabled and does not designate the
code-of-conduct enforcement address as a security mailbox. If the upstream
repository shows no private route, open only a sanitized public issue asking
the maintainers how to provide the sensitive details.

No response time, embargo period, fix timeline, or supported-version promise is
made by this document. Coordinate disclosure with the upstream maintainers and
avoid publishing sensitive details until they provide direction.

## Package verification

The local Windows packaging script creates `SHA256SUMS` next to
`NumSharp-LifeAndPong-win-x64.zip`. Compare the recorded lowercase SHA-256 value
with a fresh `Get-FileHash -Algorithm SHA256` result before using or sharing the
archive. A matching digest verifies file integrity against that checksum file;
it is not a code-signing or publisher-identity guarantee.
