TEX = pdflatex
SRC = mamba3_derivation.tex
PDF = $(SRC:.tex=.pdf)

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.pdf
