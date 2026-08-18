# wiki/ — vibetrading.wiki STATIC SITE

Hand-written static site on Cloudflare Pages. No SSG, no build step — raw HTML/JS/CSS per section; docs/ is a tiny JS SPA driven by content.js.

## STRUCTURE
- home/, docs/, tutorials/, research-lab/, alpha-library/ — site sections.
- functions/ — CF Pages Functions: _middleware.js writes bot/AI-agent analytics to D1.
- scripts/build_alpha_library.py — GENERATES alpha-library/ content (output gitignored).
- locales/, main.js, theme.js, theme-init.js, styles.css — shared chrome.
- wrangler.toml, _headers, _redirects — CF Pages config.

## CONVENTIONS
- Deploy: GitHub Actions (wiki-deploy.yml) runs `wrangler pages deploy --project-name=vibetrading-wiki` on pushes to main touching wiki/**. Needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID.
- wiki.yml validates wiki content on every PR.

## ANTI-PATTERNS
- alpha-library pages are GENERATED: edit scripts/build_alpha_library.py, never the output html.
- CI gate forbids per-stock codes in wiki content: no `NNNNNN.(SH|SZ|BJ)` or `TICKER.US` in any wiki json/csv, and none in alpha-library html (vendor-data ToS).
- Keep drafts/generated assets out of wiki/ (root .gitignore bans internal docs/ but whitelists wiki/docs/).
