TEX  = pdflatex
SRC  = SSDforDA3.tex
BASE = $(SRC:.tex=)
PDF  = $(BASE).pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.pdf
