source "https://rubygems.org"

# Matches the build GitHub Pages runs: .github/workflows/jekyll-gh-pages.yml uses
# actions/jekyll-build-pages, which pins Jekyll and its plugins via the github-pages gem.
gem "github-pages", group: :jekyll_plugins

gem "webrick", "~> 1.8"   # needed by `jekyll serve` on Ruby 3+
gem "faraday-retry"       # silences a jekyll-github-metadata warning during local builds
