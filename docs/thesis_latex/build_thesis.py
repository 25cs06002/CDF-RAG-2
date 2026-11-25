#!/usr/bin/env python3
"""Prepare editable LaTeX chapter files from the archived manuscript and figures.
No LLM inference. Never modify the frozen experimental release.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DRAFT=ROOT/"docs/thesis_manuscript/THESIS132_WORKING_DRAFT.md"
BIB=DRAFT.with_name("references.bib")
RELEASE=ROOT/"presentation_20261118/thesis132/final_thesis_results_132/release_20261125T212811325446Z"
FILES=("main.tex","metadata.tex","frontmatter.tex","biodata.tex","architecture.tex","formalism.tex")
FIGS=[
 ("01_baseline_latency","Mean end-to-end latency for all 132 queries per model, including abstentions."),
 ("02_baseline_strict_yield","Strict supported-and-grounded yield as a percentage of the 132 development questions; AI reference, not clinical accuracy."),
 ("03_baseline_ai_hallucination","AI-reference hallucination prevalence among accepted binary labels. Empty answers can receive No; zero observed Yes labels do not prove zero hallucinations."),
 ("04_variant_strict_complete","B9 variants: strict supported, grounded and complete yield over all 132 queries."),
 ("05_variant_tradeoff","Observed all-query mean latency versus strict complete yield for B9 variants. Single runs on a development cohort."),
 ("06_generator_coverage","Answer availability across the three compact-ID generator runs, using the same 132 inputs.")
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read_csv(p):
    with p.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def escape(s):
    return "".join({"&":r"\&","%":r"\%","_":r"\_","#":r"\#","$":r"\$"}.get(c,c) for c in s)
def format_proxy(v,n):
    return f"{float(v):.4f} ({int(n)})" if v and n else "N/A"
def table_data():
    rows=read_csv(RELEASE/"table_B1_B9_132.csv")
    require([r["model"] for r in rows]==[f"B{i}" for i in range(1,10)],"Wrong model cohort")
    lines=[r"\begin{table}[htbp]",r"\centering",r"\small",r"\caption{Semantic similarity proxies (observation count in parentheses). Unavailable values are N/A.}",
           r"\label{tab:semantic-proxies}",r"\begin{tabular}{lrrr}",r"\toprule",r"Model & SRS & Context relevance & Groundedness\\",r"\midrule"]
    for r in rows:
        values=[format_proxy(r[k],r[k+"_n"]) for k in ("srs_cosine_proxy","context_cosine_proxy","grounding_cosine_proxy")]
        lines.append(" & ".join([r["model"],*values])+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    return "\n".join(lines)
def convert(md,target,keys):
    md=re.sub(r"^#{2,3} \d+(?:\.\d+)+ ",lambda m:m.group(0).split(" ")[0]+" ",md,flags=re.M)
    md=re.sub(r"\[([A-Za-z][A-Za-z0-9_-]+)\](?!\()",lambda m:"[@"+m[1]+"]" if m[1] in keys else m[0],md)
    # Force uniform proportional widths for dense prose tables instead of
    # letting Pandoc's natural-width table overflow the page.
    md=re.sub(r"(?m)^\|(?:\s*:?-+:?\s*\|)+$",lambda m:"|"+ "|".join(" "+("-"*12)+" " for _ in range(m[0].count("|")-1))+"|",md)
    # Pandoc handles Unicode text with the LaTeX writer; explicit math
    # replacements for the mathematical signs in the manuscript.
    md=md.replace("⊆",r"$\subseteq$").replace("≥",r"$\geq$").replace("→",r"$\rightarrow$").replace("×",r"$\times$")
    subprocess.run(["pandoc","--from=markdown+raw_tex+tex_math_dollars","--to=latex","--top-level-division=chapter","--natbib","--wrap=none","--lua-filter",str(HERE/"table_widths.lua"),"-o",str(target)],input=md,text=True,check=True)
def prepare(dest):
    dest=Path(dest).resolve()
    require(not dest.exists(),"Destination already exists; choose a new build folder to preserve edits")
    manifest=json.loads((RELEASE/"manifest.json").read_text())
    require(manifest["schema"]=="THESIS132_FINAL_EXPORT_V1" and manifest["primary"]=="B9_v2","Wrong frozen release")
    for f,h in manifest["output_sha256"].items():
        p=(RELEASE/f).resolve()
        require(p.is_relative_to(RELEASE) and sha(p)==h,"Frozen artifact changed: "+f)
    text=DRAFT.read_text(encoding="utf-8")
    bib=BIB.read_text(encoding="utf-8")
    keys=set(re.findall(r"(?m)^@\w+\{([^,]+),",bib))
    parts=re.split(r"(?m)^# Chapter (\d+) — ([^\n]+)\n",text)
    require(len(parts)==19 and [parts[i] for i in range(1,len(parts),3)]==list("123456"),"Unexpected chapter structure")
    abstract=parts[0].split("## Abstract\n",1)[1].strip()
    dest.mkdir(parents=True)
    (dest/"chapters").mkdir();(dest/"figures").mkdir()
    for f in FILES:shutil.copyfile(HERE/f,dest/f)
    shutil.copyfile(BIB,dest/"references.bib")
    for name,_ in FIGS:shutil.copyfile(RELEASE/"figures"/(name+".png"),dest/"figures"/(name+".png"))
    convert(abstract,dest/"chapters/abstract.tex",keys)
    appendix=None
    for i in range(1,len(parts),3):
        number,title,body=parts[i:i+3]
        if number=="2":
            body=body.replace("The citation keys in brackets correspond to [references.bib](references.bib). ","").replace("[LITERATURE_SOURCE_AUDIT.md](LITERATURE_SOURCE_AUDIT.md)","the accompanying literature source audit")
        if number=="3":
            body=body.replace("### Algorithm (pseudocode)",r"\input{architecture}"+"\n\n### Algorithm (pseudocode)",1)
            body+="\n\n"+r"\input{formalism}"+"\n"
        if number=="4":
            body=body.replace("accepted Yes / accepted Yes + No, i.e. Yes/(Yes+No)","accepted Yes/(Yes+No)")
        if number=="5":
            body=body.replace("Insert the six generated figures from the release's figures/ directory with captions describing the metric denominators and AI-reference status.","The following figures report the archived experimental measurements.")
            body+="\n\n"+table_data()+"\n\n"
            for name,caption in FIGS:
                body+=("\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width=0.96\\textwidth,height=0.72\\textheight,keepaspectratio]{"+name+".png}\n\\caption{"+escape(caption)+"}\n\\label{fig:"+name.replace("_","-")+"}\n\\end{figure}\n")
            body+="\n\\clearpage\n"
        if number=="6":
            body,appendix=body.split("## Reproducibility appendix",1)
        convert("# "+title+"\n"+body,dest/f"chapters/{int(number):02}.tex",keys)
    require(appendix is not None,"Missing reproducibility appendix")
    appendix=appendix.replace("Complete institutional front matter, related work citations, acknowledgements, bibliography, and supervisor review before submission.","The archived release and integrity manifest are retained in the repository.")
    convert("# Reproducibility and implementation provenance\n"+appendix,dest/"chapters/reproducibility.tex",keys)
    shutil.copyfile(HERE/"README.md",dest/"BUILD_NOTES.md")
    (dest/"source_manifest.json").write_text(json.dumps(dict(schema="THESIS132_LATEX_DRAFT_V1",manuscript_sha256=sha(DRAFT),bibliography_sha256=sha(BIB),release_manifest_sha256=sha(RELEASE/"manifest.json"),files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}),indent=2)+"\n")
    print("Prepared editable source:",dest)
    return dest
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,default=HERE/"build");p.add_argument("--compile",action="store_true")
    a=p.parse_args()
    require(shutil.which("pandoc"),"Install pandoc")
    if a.compile:require(shutil.which("latexmk"),"Install latexmk and TeX Live")
    dest=prepare(a.output_dir)
    if a.compile:
        subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","-file-line-error","main.tex"],cwd=dest,check=True)
        log=(dest/"main.log").read_text(errors="replace")
        require("undefined citations" not in log.lower() and "undefined references" not in log.lower(),"Unresolved citations/references")
        print("Compiled working thesis:",dest/"main.pdf")
if __name__=="__main__":main()
