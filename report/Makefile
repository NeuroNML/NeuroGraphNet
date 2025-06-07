# LaTeX Makefile for IEEE conference paper

# Main document name (without extension)
MAIN = report

# Directory for figures
FIGURES_DIR = figures

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error

# Default target
all: $(MAIN).pdf

# Main PDF target
$(MAIN).pdf: $(MAIN).tex $(MAIN).bbl $(wildcard $(FIGURES_DIR)/*)
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN)

# Bibliography
$(MAIN).bbl: $(MAIN).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN)

# Clean temporary files
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Watch mode (requires latexmk)
watch:
	latexmk -pvc -pdf -interaction=nonstopmode $(MAIN).tex

.PHONY: all clean cleanall watch 