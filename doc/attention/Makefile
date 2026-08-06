TEX  = pdflatex
BIB  = bibtex
SRC  = mamba3_attention.tex
BASE = $(SRC:.tex=)
PDF  = $(BASE).pdf

.PHONY: all clean

all: $(PDF)

# Generated artifacts are prerequisites, not just the .tex. Without them a rebuild
# after regenerating the plot or table is a silent no-op and the PDF keeps the old
# image -- which is exactly what happened when the T=1025 grid completed: the figure
# on disk was current, the one in the PDF was two revisions stale.
FIGS = $(wildcard *.png) $(wildcard *.pdf_tex) ablation_table.tex

$(PDF): $(SRC) references.bib $(FIGS)
	$(TEX) $(SRC)
	$(BIB) $(BASE)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.bbl *.blg *.pdf
