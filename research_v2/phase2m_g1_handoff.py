"""Convert a PASSED Phase2M local review into a G1 *proposed* import workspace.

No direct approval, medical validation, Neo4j mutation, or model execution.
Every original (query, link, source) review event gets a distinct source ID;
a crosswalk retains *all* evidence IDs even where the G1 row has one primary
source_id. Source/event duplication is intentional for provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .g1_foundation import (
    prepare, _rows, _write, SOURCE_COLS, CONCEPT_COLS, EDGE_COLS, _norm, G0, RELATIONS
)
from .phase2m_independent_audit import BASE, SOURCE, LINK, SUPPORTED, analyze, read_rows, SOURCE_FIELDS, LINK_FIELDS, ids


def _source_key(row):
    # Source task IDs, not URL, identify reviewed passage-specific records.
    return (row["query_id"].strip(), row["link_task_id"].strip(),
            row["source_id"].strip())


def _new_id(row):
    value = "_".join(_source_key(row))
    return "P2M_" + re.sub(r"[^A-Za-z0-9_]+", "_", value)


def convert(audit_dir: Path, handoff_dir: Path = BASE, output_dir: Path | None = None):
    audit_dir=Path(audit_dir).expanduser().resolve()
    handoff_dir=Path(handoff_dir).expanduser().resolve()
    for path in (audit_dir,handoff_dir):
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Input folder must be inside repository")
    report=json.loads((audit_dir/"review_audit.json").read_text(encoding="utf-8"))
    source_path=handoff_dir/SOURCE
    link_path=handoff_dir/LINK
    if not report.get("structural_audit_passed") or report.get("error_count")!=0:
        raise ValueError("Phase2M structural audit must pass first")
    if report.get("source_sha256")!=file_sha256(source_path) or report.get("link_sha256")!=file_sha256(link_path):
        raise ValueError("Phase2M reviewed CSVs changed since audit; rerun validator")
    audited=report.get("source_folder")
    if audited!=handoff_dir.relative_to(PROJECT_ROOT).as_posix():
        raise ValueError("Audit was performed on a different handoff directory")
    sources=read_rows(source_path,SOURCE_FIELDS)
    links=read_rows(link_path,LINK_FIELDS)
    verified, candidates=analyze(sources,links)
    if not verified["structural_audit_passed"]:
        raise ValueError("Source/link consistency changed after audit")
    if len(candidates)!=report.get("supported_link_proposals"):
        raise ValueError("Proposal count differs from frozen audit")
    candidate_key={(r["query_id"],r["link_task_id"]):r for r in candidates}
    existing_sources={}
    for row in sources:
        key=_source_key(row)
        if key in existing_sources:
            raise ValueError("Duplicate source review event "+str(key))
        existing_sources[key]=row
    dest=prepare(output_dir)
    source_rows=_rows(dest/"source_review.csv")
    concept_rows=_rows(dest/"concept_inventory.csv")
    relation_rows=_rows(dest/"typed_relationship_review.csv")
    g0={_norm(r["name"]) for r in _rows(G0/"nodes.csv")}
    known={_norm(r["concept"]):r for r in concept_rows}
    existing_source_ids={r["source_id"] for r in source_rows}
    crosswalk=[]
    reviewed_source_ids={}
    # Retain ALL 52 source task events, including rejected and partial ones.
    for row in sources:
        key=_source_key(row)
        sid=_new_id(row)
        if sid in existing_source_ids:
            raise ValueError("Generated source ID collision "+sid)
        existing_source_ids.add(sid)
        reviewed_source_ids[key]=sid
        source_rows.append({
            "source_id":sid,
            "target_concept":"",
            "source_title":"VERIFY ORIGINAL SOURCE TITLE: "+row["source_id"],
            "source_url":row["source_url"],
            "scope_note":"Phase2M "+row["independent_source_decision"]+
                         "; originally "+row["original_review_decision"]+
                         "; review URL title, passage locator and scope",
            "review_status":"proposed",
            "passage_locator":row["independent_verified_passage"],
            "support_summary":row["independent_review_note"],
            "reviewer":row["independent_reviewer_id"],
            "review_date":row["independent_review_date"],
        })
        crosswalk.append({
            "query_id":row["query_id"],"link_task_id":row["link_task_id"],
            "original_source_id":row["source_id"],"g1_source_id":sid,
            "source_url":row["source_url"],"independent_source_decision":row["independent_source_decision"],
            "independent_relation":row["independent_relation"],
            "original_review_task":row["source_task_id"],
            "g1_source_approval":"proposed",
        })
    edge_provenance=[]
    for link in links:
        key=(link["query_id"],link["link_task_id"])
        if key not in candidate_key:continue
        relation=link["independent_relation"].strip().upper()
        if relation not in RELATIONS:
            raise ValueError("Unsupported typed relation "+relation)
        supports=ids(link["independent_supporting_source_ids"])
        if not supports:raise ValueError("Supported edge without sources: "+str(key))
        mapped=[reviewed_source_ids[(key[0],key[1],source_id)] for source_id in supports]
        # The G1 format currently stores one primary source ID per edge;
        # the immutable crosswalk below retains the complete support set.
        primary=mapped[0]
        edge_id="P2M_"+re.sub(r"[^A-Za-z0-9_]+","_",link["link_task_id"])
        relation_rows.append({
            "edge_id":edge_id, "subject":link["cause_node"],
            "relation":relation,"object":link["effect_node"],"source_id":primary,
            "passage_locator":existing_sources[(key[0],key[1],supports[0])]["independent_verified_passage"],
            "support_summary":link["independent_evidence_notes"],
            "review_status":"proposed","reviewer":link["independent_reviewer_id"],
            "review_date":link["independent_review_date"],
        })
        edge_provenance.append({
            "query_id":key[0],"link_task_id":key[1],"g1_edge_id":edge_id,
            "relation":relation,
            "original_supporting_source_ids":"|".join(supports),
            "g1_supporting_source_ids":"|".join(mapped),
            "g1_primary_source_id":primary,"g1_edge_approval":"proposed"
        })
        for concept in (link["cause_node"].strip(),link["effect_node"].strip()):
            ckey=_norm(concept)
            if ckey in known:continue
            record={
                "concept":concept,"g0_present":str(ckey in g0).lower(),
                "pilot_questions":0,"expansion_questions":0,
                "candidate_source_ids":"|".join(mapped),
                "review_status":"unreviewed","source_id":"",
                "passage_locator":"","reviewer":"","review_date":"",
                "review_notes":"New mechanism/intermediate from "+link["link_task_id"]+
                               "; review concept type and source support",
            }
            known[ckey]=record
            concept_rows.append(record)
    _write(dest/"source_review.csv",SOURCE_COLS,source_rows)
    _write(dest/"concept_inventory.csv",CONCEPT_COLS,concept_rows)
    _write(dest/"typed_relationship_review.csv",EDGE_COLS,relation_rows)
    _write(dest/"phase2m_source_crosswalk.csv",tuple(crosswalk[0]),crosswalk)
    _write(dest/"phase2m_edge_provenance.csv",tuple(edge_provenance[0]),edge_provenance)
    baseline=json.loads((dest/"manifest.json").read_text(encoding="utf-8"))
    baseline["phase2m_review_import"]={
        "phase2m_audit_sha256":file_sha256(audit_dir/"review_audit.json"),
        "source_sha256":file_sha256(source_path),
        "link_sha256":file_sha256(link_path),
        "proposed_edges":len(relation_rows),
        "source_review_events":len(sources),
        "source_events_including_partial":True,
        "edge_approvals_from_conversion":0,
        "source_approvals_from_conversion":0,
        "clinical_gold":False,
        "all_source_ids_retained_in_crosswalk":True,
        "warning":"Reviewer must verify source titles and graph concept identity, then explicitly approve applicable source, concept and typed edge rows. No automatic medical fact certification.",
    }
    (dest/"manifest.json").write_text(json.dumps(baseline,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "workspace":str(dest),
        "source_rows_added":len(sources),
        "typed_edge_proposals":len(relation_rows),
        "edge_provenance_count":len(edge_provenance),
        "reviewed_approved_automatically":0,
        "concept_inventory_rows":len(concept_rows),
        "source_crosswalk":"phase2m_source_crosswalk.csv",
        "link_provenance":"phase2m_edge_provenance.csv",
    },indent=2))
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir",type=Path,required=True)
    p.add_argument("--handoff-dir",type=Path,default=BASE)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:p.error("Run from repository root")
    convert(a.audit_dir,a.handoff_dir,a.output_dir)


if __name__=="__main__":
    main()
