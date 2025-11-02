.PHONY: all clean build view

all: build

build:
	@mkdir -p build
	# Build standalone TikZ figures to PDF with stable names
	@cd figures && pdflatex -interaction=nonstopmode dsm_architecture.tex >/dev/null 2>&1 || true
	@TEXINPUTS=".:_mlsys_template/mlsys2025style//:" BSTINPUTS=".:_mlsys_template/mlsys2025style//:" pdflatex -output-directory=build -interaction=nonstopmode main.tex
	@cp sections/refs.bib build/ 2>/dev/null || true
	@cd build && BSTINPUTS=".:../_mlsys_template/mlsys2025style//:" bibtex main || true
	@TEXINPUTS=".:_mlsys_template/mlsys2025style//:" BSTINPUTS=".:_mlsys_template/mlsys2025style//:" pdflatex -output-directory=build -interaction=nonstopmode main.tex
	@TEXINPUTS=".:_mlsys_template/mlsys2025style//:" BSTINPUTS=".:_mlsys_template/mlsys2025style//:" pdflatex -output-directory=build -interaction=nonstopmode main.tex
	@cp build/main.pdf main.pdf
	@echo "PDF generated: main.pdf"

clean:
	@rm -rf build/
	@rm -f main.pdf
	@rm -f *.aux *.fdb_latexmk *.fls *.log *.out *.synctex.gz
	@rm -f sections/*.aux sections/*.fdb_latexmk sections/*.fls sections/*.log sections/*.pdf sections/*.synctex.gz
	@echo "Cleaned build artifacts"

view: build
	@xdg-open main.pdf 2>/dev/null || open main.pdf 2>/dev/null || echo "Please open main.pdf manually"
