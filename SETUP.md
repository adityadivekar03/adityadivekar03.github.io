# Setup

These files were prepared without running Hugo (sandbox network couldn't reach
GitHub's release downloads), so `hugo.toml`, the workflow YAML, and post
front matter were validated as well-formed TOML/YAML/YAML by parsing them
directly — but the actual `hugo server`/`hugo build` has not been run.
Do that first, locally, before pushing.

## 1. Install Hugo (extended version, needed for SCSS/asset pipeline)

```bash
brew install hugo
hugo version   # confirm "extended" appears in the output
```

## 2. Create the repo and drop these files in

```bash
mkdir adityadivekar && cd adityadivekar
git init
hugo new site . --format=toml --force   # --force: lets it write into a dir with existing files
```

Copy everything from this `site-scaffold/` folder into the new repo root,
overwriting the generated `hugo.toml`.

## 3. Add the theme as a submodule

```bash
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod --depth=1
```

## 4. Preview locally

```bash
hugo server -D
```

Open `http://localhost:1313`. Confirm:
- Home page shows the intro text, no images
- `/posts/` lists the tax-aware portfolio post (drafts show because of `-D`)
- Math renders (KaTeX) once you replace the placeholder LaTeX in the post
- Code blocks are syntax-highlighted

## 5. Push and enable Pages

```bash
git add -A
git commit -m "Initial site scaffold"
git branch -M main
git remote add origin https://github.com/adityadivekar03/adityadivekar.git
git push -u origin main
```

Then in the repo on GitHub: **Settings → Pages → Build and deployment →
Source: GitHub Actions**. The included workflow
(`.github/workflows/hugo.yml`) builds and deploys automatically on every
push to `main`.

Site will be live at: `https://adityadivekar03.github.io/adityadivekar/`

## 6. Writing new posts

```bash
hugo new posts/my-next-post.md
```

Edit the file, set `draft: false` when ready, commit, push.
