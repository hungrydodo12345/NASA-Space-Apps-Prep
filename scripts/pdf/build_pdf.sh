#!/bin/sh
# Build docs/paper/PAPER.pdf from docs/paper/PAPER.md (pandoc -> HTML with embedded figures -> Chromium print via Playwright).
set -e; R=$(cd "$(dirname "$0")/../.." && pwd); T=$(mktemp -d)
cd "$R/docs/paper" && pandoc PAPER.md -f gfm -t html5 --standalone --metadata title="paper" -c "$R/scripts/pdf/paper.css" --embed-resources --resource-path=.:../.. -o "$T/paper.html"
sed -i 's#<header id="title-block-header">.*</header>##; /<h1 class="title">paper<\/h1>/d' "$T/paper.html"
node "$R/scripts/pdf/topdf.js" "$T/paper.html" "$R/docs/paper/PAPER.pdf"; echo "wrote docs/paper/PAPER.pdf"
