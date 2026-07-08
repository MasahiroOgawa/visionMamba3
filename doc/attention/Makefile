TEX  = pdflatex
BIB  = bibtex
SRC  = mamba3_attention.tex
BASE = $(SRC:.tex=)
PDF  = $(BASE).pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC) references.bib
	$(TEX) $(SRC)
	$(BIB) $(BASE)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.bbl *.blg *.pdf
