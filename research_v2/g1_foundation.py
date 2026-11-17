"""G1: existing source registries -> manual concept/source/typed-edge review.

No automatic medical evidence validation; build writes G1_CANDIDATE only.
Frozen G0 and pilot100 are never modified. No Ollama/Neo4j/PPO.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, date, timezone
from pathlib import Path
from urllib.parse import urlparse

from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines.baselines import G0Corpus
from .medical_benchmark import load_benchmark

ROOT = PROJECT_ROOT
PILOT = ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
EXPANSION = ROOT / "data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
G0 = ROOT / "research_artifacts/graph-G0"
WORKSPACES = ROOT / "research_artifacts/experiment-artifacts/g1_review_workspaces"
SOURCE_FILES = [
    ROOT / "data/benchmark/source_candidates15_v2.jsonl",
    ROOT / "data/benchmark/additional_source_candidates8_v1.jsonl",
]
# These two checked official pages supplement existing candidate metadata.
# Their presence in the registry does not approve any source claim.
EXTRA_SOURCES = [
    {"source_id":"G1SRC_CDC_STROKE","target_concept":"stroke",
     "title":"About Stroke","url":"https://www.cdc.gov/stroke/about/",
     "scope_note":"Distinguish ischemic obstruction from hemorrhagic rupture"},
    {"source_id":"G1SRC_NIDDK_TYPE2","target_concept":"type 2 diabetes",
     "title":"Type 2 Diabetes","url":"https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes",
     "scope_note":"Distinguish mechanisms from risk factors and individual diagnosis"},
]
RELATIONS = {"CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "TRIGGERS_EXACERBATION",
             "PROTECTS_AGAINST", "IS_A", "UNCERTAIN"}
# Only CAUSES appears in original B5/B6 causal-edge format. Other types stay typed.
SOURCE_COLS = ("source_id","target_concept","source_title","source_url","scope_note",
               "review_status","passage_locator","support_summary","reviewer","review_date")
CONCEPT_COLS = ("concept","g0_present","pilot_questions","expansion_questions","candidate_source_ids",
                "review_status","source_id","passage_locator","reviewer","review_date","review_notes")
EDGE_COLS = ("edge_id","subject","relation","object","source_id","passage_locator",
             "support_summary","review_status","reviewer","review_date")
ALIAS_COLS = ("alias","canonical_concept","source_id","justification","review_status",
              "reviewer","review_date")


def _norm(s):
    return " ".join(str(s).casefold().strip().split())


def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path,columns,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");w.writeheader();w.writerows(rows)


def _jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def validate_expansion():
    pilot=load_benchmark(PILOT)
    extra=load_benchmark(EXPANSION)
    if len(pilot)!=100 or len(extra)!=32:
        raise ValueError("Frozen 100 and separate 32-question expansion required")
    if {q.query_id for q in pilot} & {q.query_id for q in extra}:
        raise ValueError("Duplicate benchmark query IDs")
    if len({_norm(q.query) for q in pilot+extra})!=132:
        raise ValueError("Identical query texts in benchmark")
    if any(q.review_status!="draft" or q.split is not None or q.answerable is not None for q in extra):
        raise ValueError("New queries must remain unreviewed and unassigned to splits")
    if Counter(q.query_type for q in extra)!={k:4 for k in (
        "direct","paraphrased","indirect","multi_hop","comparative","risk_factor","ambiguous","unsupported")}:
        raise ValueError("32-query distribution mismatch")
    return pilot,extra


def candidate_registry():
    sources={}
    for path in SOURCE_FILES:
        for item in _jsonl(path):
            sid=item["source_id"]
            if sid in sources: raise ValueError("Duplicate candidate source ID "+sid)
            sources[sid]=item
    for item in EXTRA_SOURCES:
        if item["source_id"] in sources: raise ValueError("Duplicate new source ID")
        sources[item["source_id"]]=item
    return sources


def _reviewed(row,fields):
    if row["review_status"] not in {"unreviewed","proposed","approved","rejected"}:
        raise ValueError("Invalid review_status: "+row["review_status"])
    if row["review_status"]!="approved":return False
    for field in fields:
        if not row.get(field,"").strip():
            raise ValueError("Approved row missing "+field)
    try: d=date.fromisoformat(row["review_date"])
    except ValueError as e: raise ValueError("Invalid review date") from e
    if d>datetime.now(timezone.utc).date():raise ValueError("Review date is in future")
    return True


def prepare(output=None):
    pilot,extra=validate_expansion()
    src=candidate_registry()
    graph=G0Corpus()
    g0_names={_norm(n):n for n in graph.nodes}
    per_concept=defaultdict(lambda: {"pilot":0,"extension":0,"display":""})
    for group,records in (("pilot",pilot),("extension",extra)):
        for q in records:
            if q.target_concept:
                key=_norm(q.target_concept)
                per_concept[key][group]+=1
                per_concept[key]["display"]=q.target_concept
    source_rows=[]
    source_by_target=defaultdict(list)
    for sid,item in src.items():
        target=item.get("target_concept") or ""
        if target:source_by_target[_norm(target)].append(sid)
        source_rows.append({
            "source_id":sid,"target_concept":target,"source_title":item.get("title",""),
            "source_url":item.get("url",""),"scope_note":item.get("scope_note",""),
            "review_status":"unreviewed","passage_locator":"","support_summary":"",
            "reviewer":"","review_date":""
        })
    inventory=[{
        "concept":v["display"],"g0_present":str(k in g0_names).lower(),
        "pilot_questions":v["pilot"],"expansion_questions":v["extension"],
        "candidate_source_ids":"|".join(sorted(source_by_target[k])),
        "review_status":"unreviewed","source_id":"","passage_locator":"",
        "reviewer":"","review_date":"","review_notes":""
    } for k,v in sorted(per_concept.items())]
    if not inventory:raise ValueError("Empty inventory")
    base=(Path(output) if output else WORKSPACES/(
        "G1_REVIEW_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
        "_"+uuid.uuid4().hex[:6])).expanduser().resolve()
    if not base.is_relative_to(ROOT):raise ValueError("Output must stay in repo")
    base.mkdir(parents=True,exist_ok=False)
    _write(base/"concept_inventory.csv",CONCEPT_COLS,inventory)
    _write(base/"source_review.csv",SOURCE_COLS,source_rows)
    _write(base/"typed_relationship_review.csv",EDGE_COLS,[])
    _write(base/"alias_review.csv",ALIAS_COLS,[])
    manifest={
        "stage":"G1_REVIEW_REQUIRED","pilot_queries":len(pilot),
        "separate_expansion_queries":len(extra),"concepts_in_inventory":len(inventory),
        "missing_target_concepts":sum(c["g0_present"]=="false" for c in inventory),
        "candidate_source_count":len(source_rows),
        "approved_new_edges":0,"clinical_gold":False,
        "reference_sha256":{str(p.relative_to(ROOT)):file_sha256(p) for p in
            (PILOT,EXPANSION,G0/"nodes.csv",G0/"edges.csv",*SOURCE_FILES)},
        "warnings":[
            "Existing source-candidate review history is NOT independently validated as medical gold",
            "This is an editable audit workspace; CSV approvals are reviewer attestations, not automatic factual checks",
            "Do not import graph relationships until direction, relation type and source passage are reviewed",
            "G0 historical edges will remain explicitly marked unverified when inherited."
        ],
    }
    (base/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(base),"pilot_queries":100,"new_queries":32,
        "missing_concepts":manifest["missing_target_concepts"],
        "candidate_sources":manifest["candidate_source_count"]},indent=2))
    return base


def audit_and_build(folder,*, build=False):
    folder=Path(folder).expanduser().resolve()
    if not folder.is_relative_to(ROOT):raise ValueError("Review folder must be in repository")
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["stage"]!="G1_REVIEW_REQUIRED":raise ValueError("Not a G1 review workspace")
    for rel,sha in manifest["reference_sha256"].items():
        if file_sha256(ROOT/rel)!=sha:raise ValueError("Frozen input changed: "+rel)
    concept_rows=_rows(folder/"concept_inventory.csv")
    source_rows=_rows(folder/"source_review.csv")
    edge_rows=_rows(folder/"typed_relationship_review.csv")
    alias_rows=_rows(folder/"alias_review.csv")
    approved_sources={}
    for row in source_rows:
        if _reviewed(row,("source_id","source_url","source_title","passage_locator","support_summary","reviewer","review_date")):
            url=urlparse(row["source_url"])
            if url.scheme!="https" or not url.netloc:raise ValueError("Invalid source URL")
            approved_sources[row["source_id"]]=row
    g0=G0Corpus()
    known={_norm(n):n for n in g0.nodes}
    approved_nodes={}
    for row in concept_rows:
        concept_key = _norm(row.get("concept", ""))
        # Existing G0 nodes have a frozen identity. G1 does not need to
        # re-approve their existence; their ORIGINAL edges remain unverified.
        # Still check review_status spelling for consistency.
        if concept_key in known:
            if row.get("review_status") not in {"unreviewed", "proposed", "approved", "rejected"}:
                raise ValueError("Invalid concept review_status: " + str(row.get("review_status")))
            continue
        if not _reviewed(row,("concept","source_id","passage_locator","reviewer","review_date","review_notes")):
            continue
        if row["source_id"] not in approved_sources:
            raise ValueError(
                "Approved NEW concept requires approved source: "
                + row["concept"] + " (source_id=" + row["source_id"] + ")"
            )
        if concept_key in approved_nodes:
            raise ValueError("Duplicate approved new concept: " + row["concept"])
        approved_nodes[concept_key] = row["concept"].strip()
    known.update(approved_nodes)
    accepted=[]
    for row in edge_rows:
        if not _reviewed(row,("edge_id","subject","relation","object","source_id",
                              "passage_locator","support_summary","reviewer","review_date")):
            continue
        if row["relation"] not in RELATIONS:raise ValueError("Unknown relationship type")
        if row["source_id"] not in approved_sources:raise ValueError("Unreviewed source in relation")
        subj,obj=_norm(row["subject"]),_norm(row["object"])
        if subj not in known or obj not in known or subj==obj:
            raise ValueError("Approved edge references missing concepts or self-edge")
        if len(row["support_summary"].strip())<20:raise ValueError("Evidence note needs detail")
        accepted.append(row)
    aliases=[]
    for row in alias_rows:
        if not _reviewed(row,("alias","canonical_concept","source_id","justification","reviewer","review_date")):
            continue
        if row["source_id"] not in approved_sources:raise ValueError("Unreviewed alias source")
        if _norm(row["canonical_concept"]) not in known:raise ValueError("Unknown alias canonical")
        if _norm(row["alias"]) in known:raise ValueError("Alias conflicts with canonical concept")
        aliases.append(row)
    identities=[(_norm(e["subject"]),e["relation"],_norm(e["object"])) for e in accepted]
    if len(identities)!=len(set(identities)):raise ValueError("Duplicate approved relationship")
    if len({_norm(r["alias"]) for r in aliases})!=len(aliases):
        raise ValueError("Duplicate approved alias")
    result={"approved_source_count":len(approved_sources),
        "approved_new_node_count":len(approved_nodes),
        "approved_typed_relations":dict(Counter(e["relation"] for e in accepted)),
        "approved_aliases":len(aliases),"source_urls_checked_online":False,
        "medical_gold":False}
    if not build:
        print(json.dumps(result,indent=2));return result
    if not accepted:raise ValueError("No approved typed relationships; G1 not created")
    out=folder/"graphs"/("G1_CANDIDATE_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    nodes=_rows(G0/"nodes.csv")
    existing=_rows(G0/"edges.csv")
    oldpairs={(_norm(r["cause"]),_norm(r["effect"])) for r in existing}
    for k,name in approved_nodes.items():
        nid="G1N_"+hashlib.sha256(k.encode()).hexdigest()[:16]
        nodes.append({"node_id":nid,"name":name})
    causal_additions=[]
    for e in accepted:
        pair=(_norm(e["subject"]),_norm(e["object"]))
        if e["relation"]=="CAUSES" and pair not in oldpairs:
            eid="G1E_"+hashlib.sha256("\x1f".join(pair).encode()).hexdigest()[:16]
            causal_additions.append({"edge_id":eid,"cause":known[pair[0]],
                                     "effect":known[pair[1]]})
            oldpairs.add(pair)
    _write(out/"nodes.csv",("node_id","name"),nodes)
    _write(out/"edges.csv",("edge_id","cause","effect"),existing+causal_additions)
    _write(out/"typed_relationships.csv",EDGE_COLS,accepted)
    _write(out/"approved_aliases.csv",ALIAS_COLS,aliases)
    # Snapshot checksums must bind to the files actually loaded by the
    # G1 typed retriever, not the pre-build editable review worksheets.
    snapshot_typed_sha = file_sha256(out/"typed_relationships.csv")
    snapshot_alias_sha = file_sha256(out/"approved_aliases.csv")
    result.update({
        "stage":"G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION",
        "new_causal_edges":len(causal_additions),
        "inherited_G0_edges_unverified":len(existing),
        "total_nodes":len(nodes),"total_causes_edges":len(existing+causal_additions),
        "review_workspace":folder.relative_to(ROOT).as_posix(),
        "source_ledger_sha256":file_sha256(folder/"source_review.csv"),
        "edge_ledger_sha256":snapshot_typed_sha,
        "alias_ledger_sha256":snapshot_alias_sha,
        "editable_edge_ledger_sha256":file_sha256(folder/"typed_relationship_review.csv"),
        "editable_alias_ledger_sha256":file_sha256(folder/"alias_review.csv"),
        "clinical_accuracy":False,
    })
    (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(out),**result},indent=2))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","audit","build","validate-queries"))
    parser.add_argument("--folder",type=Path)
    parser.add_argument("--output",type=Path)
    a=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:parser.error("Run from repository root")
    if a.command=="validate-queries":
        pilot,extra=validate_expansion();print(json.dumps({"pilot":len(pilot),"extra":len(extra)}))
    elif a.command=="prepare":prepare(a.output)
    else:
        if not a.folder:parser.error("--folder required for audit/build")
        audit_and_build(a.folder,build=(a.command=="build"))


if __name__=="__main__":main()
