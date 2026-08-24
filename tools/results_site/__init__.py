"""The inference results explorer (WP11).

The tier-1 and tier-2 suites answer one question: does the pipeline still agree with its oracle
and its goldens. They answer it in pass or fail, which is the right shape for a gate and the wrong
shape for looking at what the models actually did. This package answers the other question. It
runs every model over its corpus, keeps the whole wire-shaped result body for each image, draws
the regions the model reported onto a copy of the picture, and writes a static website that opens
straight off disk.

The site is deliberately inert: ``index.html``, ``site.css``, ``site.js``, and a ``data.js`` that
assigns ``window.RESULTS``. There is no fetch, no bundler, and no external resource of any kind,
so ``file:///...`` behaves exactly as a served copy does.

Modules:
    corpus: Which models run over which images, for both suites.
    overlays: Denormalization, region drawing, and thumbnails.
    model: The ``data.js`` document, its per-entry records, and the merge of two runs.
    render: The static files the site is made of.
    build: The run itself, from staged bundle to written site.
"""

from __future__ import annotations

#: The ``data.js`` document version. It moves when the site's data model changes shape, so a
#: ``--merge`` onto a site an older tool built is refused rather than half read.
SITE_SCHEMA_VERSION = 1
