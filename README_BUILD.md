Building the modular two-column LaTeX paper

Files:
- `notes/main.tex` — main entry (two-column) that includes files under `sections/`
- `notes/sections/0_abstract.tex`, `1_introduction.tex`, `2_background.tex`, `3_model.tex`
- `notes/sections/refr.bib` — bibliography
- `notes/.latexmkrc` — latexmk config (builds to `build/` directory)
- `notes/.vscode/settings.json` — VS Code LaTeX Workshop config (disables auto-compile, uses build/)
- `notes/Makefile` — build automation
- `notes/.gitignore` — keeps workspace clean

Quick build (keeps workspace clean):

```bash
cd analytical/notes
make
```

This builds all artifacts to `build/` and copies the final PDF to `main.pdf`.

VS Code LaTeX Workshop:
- Auto-build is disabled to keep workspace clean
- Manual build: Use `make` or Ctrl+Alt+B
- All artifacts go to `build/` directory

Manual build (if make not available):

```bash
cd analytical/notes
mkdir -p build
pdflatex -output-directory=build main.tex
cp sections/refr.bib build/
cd build && bibtex main && cd ..
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
cp build/main.pdf .
```

Clean build artifacts:

```bash
cd analytical/notes
make clean
```

To convert to a strict MLSys/NeurIPS/ICLR style, replace the documentclass and packages in `main.tex` with the target conference `.cls` file and adjust author/title metadata.
