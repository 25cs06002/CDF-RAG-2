# Editable LaTeX working thesis

This is a **provisional** A4 report-class layout, not the official IIT Bhubaneswar template. The institute's Academic Resources page links to https://www.iitbbs.ac.in/curriculum_doc/ThesisTemplate.rar but that archive could not be retrieved during preparation. Confirm School-specific requirements and whether the document is the interim MTP report or final dissertation before submission. The institution's regulation page is https://www.iitbbs.ac.in/index.php/home/academics/regulations/ . Front-matter certificate, declaration and acknowledgements are unsigned placeholders; verify personal details in metadata.tex. Confirm and complete bio-data, abstract placement, bibliography, AI-assistance disclosure and approvals. The literature source audit records source-access limits.

## Build from repository root (Ubuntu)

    sudo apt install pandoc latexmk texlive-latex-extra texlive-fonts-recommended texlive-pictures lmodern
    python docs/thesis_latex/build_thesis.py --compile

The script refuses to overwrite an existing output directory, verifies every frozen release artifact, converts the six chapters with Pandoc, maps bibliography keys to natbib citations, copies six published experimental plots, generates a semantic table from the archived CSV, and adds a controller diagram and formal equations. No new experimental runs are required.

Generated files: docs/thesis_latex/build/main.tex, metadata.tex, frontmatter.tex, chapters/*.tex, references.bib, figures/*.png, architecture.tex, formalism.tex, biodata.tex, source_manifest.json and main.pdf (when compiled). The generated folder is a standalone editable LaTeX project. For Overleaf, upload the generated source project (not the Python script); choose main.tex and pdfLaTeX. For edits, work inside the generated project, then run latexmk -pdf main.tex there. For another conversion use --output-dir with a new path; never overwrite your revised chapter files.

The manuscript is a working draft for supervisor review. Experimental numbers are unchanged, except clarification of a denominator expression. Clinical validation, independent holdout and repeat timing are still limitations. Review the layout and bibliography manually before submission.

## Compilation failures

First-pass missing TOC/LOF/LOT and undefined citations are expected until BibTeX and reruns finish. Look for the first fatal `!` or `file.tex:line:` message in main.log. The build prints the first such diagnostic and preserves editable files. For an existing project, run from the repository root:

    python docs/thesis_latex/build_thesis.py --compile-existing docs/thesis_latex/build

If a failed compilation has stale auxiliary files, run `latexmk -C main.tex` inside the generated project first; this removes generated outputs, not the source. To regenerate with updated templates without overwriting existing edits, use `--output-dir docs/thesis_latex/build_fixed --compile`.
