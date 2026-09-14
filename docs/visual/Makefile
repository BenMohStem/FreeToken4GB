PDF := FreeToken4GB-Visual-Guide.pdf
TEX := FreeToken4GB-Visual-Guide.tex
BUILDDIR := ../../.latex-build

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	@mkdir -p $(BUILDDIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -outdir=$(BUILDDIR) -jobname=$(basename $(PDF)) $(TEX)
	cp $(BUILDDIR)/$(PDF) $(PDF)

clean:
	rm -rf $(BUILDDIR) $(PDF)
