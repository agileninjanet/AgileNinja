# CLAUDE.md - AI Assistant Guide for AgileNinja

## Project Overview

This is a **Hugo + React + Netlify CMS boilerplate** (Victor Hugo) for building content-managed static websites. It combines Hugo static site generation with Netlify CMS for content management, modern ES6/React for interactivity, and Tachyons-based utility CSS for styling.

**Current theme**: Kaldi (a coffee company website demo with blog, products, values, and contact sections)

## Tech Stack

| Category | Technology |
|----------|------------|
| Static Site Generator | Hugo (binaries in `/bin`) |
| Frontend | React 15.6.1, ES6+ |
| CSS | Tachyons (utility-first), PostCSS, cssnext, cssnano |
| Build Tools | Gulp 3.9.1, Webpack 3.0.0, Babel 6.x |
| CMS | Netlify CMS 1.0.2+ with git-gateway backend |
| Auth | Netlify Identity (gotrue-js) |
| Linting | ESLint 3.1.1 with babel-eslint parser |

## Directory Structure

```
/AgileNinja
├── /src                          # Frontend source files
│   ├── /js
│   │   ├── app.js               # Main app entry point
│   │   ├── cms.js               # Netlify CMS setup & preview templates
│   │   └── /cms-preview-templates  # React components for CMS previews
│   └── /css
│       ├── main.css             # Main CSS entry (imports Tachyons)
│       ├── cms.css              # CMS-specific styles
│       └── /imports             # CSS modules (~40 utility files)
│
├── /site                         # Hugo site source
│   ├── config.toml              # Hugo configuration
│   ├── /content                 # Markdown content (YAML frontmatter)
│   │   ├── _index.md            # Home page
│   │   ├── /post                # Blog posts
│   │   ├── /products            # Products page
│   │   ├── /values              # About/Values page
│   │   └── /contact             # Contact page
│   ├── /layouts
│   │   ├── /_default            # Base templates (baseof, single, list)
│   │   ├── /partials            # Reusable template components (~20 files)
│   │   └── /post                # Post-specific layouts
│   └── /static
│       ├── /img                 # Images, icons, favicons
│       └── /admin               # Netlify CMS (config.yml, index.html)
│
├── /bin                          # Hugo binaries (darwin, linux, exe)
├── /dist                         # Build output (generated)
├── gulpfile.babel.js            # Gulp build tasks
├── webpack.conf.js              # Webpack configuration
├── .babelrc                     # Babel config (ES2015 + React presets)
├── .eslintrc                    # ESLint config (YAML format)
├── netlify.toml                 # Netlify deployment config
└── package.json                 # Dependencies and scripts
```

## Development Commands

```bash
# Install dependencies
yarn install

# Start development server (with live reload)
yarn start

# Build for production
yarn build

# Build with drafts and future posts (for previews)
yarn build-preview

# Lint JavaScript
yarn lint

# Individual build tasks
yarn hugo      # Build Hugo only
yarn webpack   # Bundle JS only
```

## Build Pipeline

The build process (via Gulp) runs these tasks:

1. **css** - Compiles CSS with PostCSS (import, cssnext, cssnano)
2. **js** - Bundles JavaScript with Webpack (outputs `app.js` and `cms.js`)
3. **cms-assets** - Copies Netlify CMS asset files
4. **svg** - Optimizes SVG icons and generates sprite
5. **hugo** - Builds the Hugo static site

**Development server** (BrowserSync) watches:
- `src/js/**/*.js` → triggers JS bundling
- `src/css/**/*.css` → triggers CSS compilation
- `site/static/img/icons-*.svg` → triggers SVG sprite generation
- `site/**/*` → triggers Hugo rebuild

## Code Conventions

### JavaScript/React
- ES6 modules with import/export
- React class components for CMS previews
- Immutable.js patterns in CMS (`entry.getIn()`)
- 2-space indentation, double quotes, trailing semicolons
- JSX uses double quotes

### CSS
- Utility-first approach (Tachyons classes)
- CSS custom properties defined in `src/css/imports/_variables.css`
- Responsive prefixes: `-ns` (not small), `-l` (large), `-xl` (extra large)
- Primary color: `--primary: rgba(255, 68, 0, 1)` (orange)

### Hugo Templates
- Partial-based architecture with dict data passing
- Range loops for collections
- Paginator for blog (4 posts per page)
- Content blocks with `{{ define }}` and `{{ block }}`

### Content (Markdown)
- YAML frontmatter for structured data
- Image paths relative to `/static` directory
- Nested objects for complex content types

## Key Files to Understand

| File | Purpose |
|------|---------|
| `package.json` | Scripts, dependencies |
| `gulpfile.babel.js` | Build pipeline tasks |
| `webpack.conf.js` | JS bundling config (dual entry: app + cms) |
| `site/config.toml` | Hugo base configuration |
| `site/static/admin/config.yml` | Netlify CMS schema (collections, fields) |
| `src/js/cms.js` | CMS setup and preview component registration |
| `src/css/imports/_variables.css` | Global CSS variables (colors, spacing) |
| `site/layouts/_default/baseof.html` | Base Hugo template |

## CMS Content Structure

**Collections defined in `site/static/admin/config.yml`:**

- **Posts** (`/site/content/post/`) - Blog articles with title, date, description, image, body
- **Home Page** (`/site/content/_index.md`) - Hero, intro, products list, values
- **Contact Page** (`/site/content/contact/`) - Contact entries with heading + text
- **Products Page** (`/site/content/products/`) - Intro, main section, testimonials, pricing
- **Values Page** (`/site/content/values/`) - List of values with images

## Common Tasks

### Adding a new blog post
Create markdown file in `site/content/post/` with frontmatter:
```yaml
---
title: "Post Title"
date: 2024-01-01T00:00:00.000Z
description: "Brief description"
image: /img/optional-image.jpg
---
Post content here...
```

### Modifying styles
1. For global variables: edit `src/css/imports/_variables.css`
2. For component styles: add/modify files in `src/css/imports/`
3. Import new files in `src/css/main.css`

### Adding new Hugo partials
1. Create partial in `site/layouts/partials/`
2. Include with `{{ partial "filename.html" . }}` or with dict: `{{ partial "filename.html" (dict "key" "value") }}`

### Working with SVG icons
1. Add SVG to `site/static/img/icons/`
2. Build generates sprite in `site/layouts/partials/svg.html`
3. Use: `<svg><use xlink:href="#icon-id"></use></svg>`

## Important Notes

- **No test suite**: Project has no automated tests configured
- **Platform binaries**: Hugo binaries for darwin/linux/windows in `/bin`
- **Yarn preferred**: `netlify.toml` specifies Yarn 1.3.2 for builds
- **Output directory**: All builds output to `/dist`
- **CMS previews**: React components in `src/js/cms-preview-templates/` provide live editing previews
- **Pagination**: Blog uses 4 posts per page (configured in Hugo templates)

## Deployment

Configured for Netlify deployment via `netlify.toml`:
- Build command: `yarn build`
- Publish directory: `dist`
- Deploy previews use: `yarn build-preview`

## ESLint Rules Summary

Key rules from `.eslintrc`:
- 2-space indentation
- Double quotes for strings and JSX
- Semicolons required
- No trailing whitespace
- Errors for undefined variables and unused variables
- Import plugin for module resolution
