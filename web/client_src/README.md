# Blink Browser UI Source

This directory is the repo-owned browser client workspace for the packaged Blink
browser UI.

- Client assets live in [`src/`](./src)
- Authored Blink overlays live beside vendored browser runtime assets that
  `/client/` loads directly
- Generated package copies are written to `../../src/blink/web/client_dist`
- Rebuild the local package copy with `node web/client_src/build.mjs`

The current source baseline is vendored from the working Blink browser client so runtime behavior stays stable while branding and product-surface cleanup continue.
