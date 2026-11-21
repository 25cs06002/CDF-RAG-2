"""Screenshot-style metric reports for the fixed 132-query B1–B8 experiment.

Reads *existing* versioned B1–B8 CSVs, validates hashes via
research_v2.thesis132_evaluation._load, and writes separate summary, detailed,
comparison, and optional clinician-reviewed hallucination classification CSVs.

Cosine metrics replicate the OLD 22-query proxy idea, NOT medical accuracy:
SRS: cosine(original query, recorded refined query).
Context relevance: cosine(query, rewritten knowledge; else retrieved evidence).
Groundedness proxy: cosine(rewritten knowledge; else answer, retrieved evidence).
B1–B4 evidence = actual candidate document text, not document IDs.
B5–B8 evidence = serialized causal edge/path wording from unverified G0.
Different proxy bases are explicitly marked; no universal gold relevance exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import thesis132_evaluation as core
from .experiment_runner import PROJECT_ROOT as ROOT
from .thesis132_evaluation import MODELS, _mean, _quantile, _hash, _load, _derive, _aggregate, _paired, _write_csv, _resolve_dir

DEFAULT_SOURCE = (
    ROOT / "presentation_20261118/thesis132/csv_reports" /
    "batch132_csv_20261121T061653054773Z_bfeb15"
)
CANDIDATES = (
    ROOT / "presentation_20261118/corpus/"
    "provisional_source_summaries_v1/documents.jsonl"
)
METRIC_NAMES = (
    "Queries Evaluated", "Causal Retrieval Coverage", "Causal Chain Depth",
    "Semantic Refinement Score", "Context Relevance", "Groundedness",
    "Average Answer Length", "Hallucination Rate", "Detector Flag Rate (Not Gold)",
    "Retrieval Success Rate", "Average Direct Causes", "Total Direct Causes",
    "Total Execution Time (s)", "Average Latency (s)", "Average Latency (ms)",
    "Minimum Latency (ms)", "Maximum Latency (ms)", "Median Latency (ms)",
    "P95 Latency (ms)", "Average Throughput (QPS)",
    "Minimum Throughput (QPS, latency proxy)",
    "Maximum Throughput (QPS, latency proxy)",
    "Refinement Success Rate", "Recorded Query Change Rate",
    "Answers Present", "Abstained Queries", "Execution Errors",
    "Mean Latency if Answered (ms)", "Mean Latency if Abstained (ms)",
    "Number of Causal Paths", "Validation Pass Fraction (Not Gold)",
)
DETAIL_FIELDS = (
    "model","query_id","query","query_type","medical_domain","semantic_group",
    "status","matched_concept","refined_query_text","final_response",
    "retrieval_success","causal_coverage","direct_causes_count",
    "multi_hop_path_count","causal_chain_depth_nodes","answer_length_words",
    "refinement_recorded","refinement_changed_query",
    "SRS_Cosine_Proxy","Context_Relevance_Cosine_Proxy",
    "Groundedness_Cosine_Proxy","context_basis","grounding_basis",
    "hallucination_detector_flag","is_hallucination_gold","validator_passed",
    "latency_seconds","latency_ms","reciprocal_latency_qps_proxy",
    "num_retrieved_documents","answer_present","evidence_present",
)
GOLD_COLUMNS = (
    "model","query_id","is_hallucination_gold","review_status",
    "reviewer_id","reviewer_qualification","independence_declaration",
    "verified_source_evidence","review_notes",
)
HAL_COLUMNS = (
    "model","queries","detector_observations","detector_flags",
    "detector_flag_rate_not_gold","gold_approved_count",
    "gold_hallucination_count","gold_review_coverage_fraction",
    "classification_support","classification_status",
    "accuracy","precision","recall","f1","tn","fp","fn","tp",
)
SUMMARY_EXTRA = ("status","notes")


def _valid_float(x, label):
    if x is None or x == "":
        return None
    value = float(x)
    if not math.isfinite(value):
        raise ValueError("Nonfinite " + label)
    return value


def _b(raw):
    return core._boolean(raw, "boolean")


def _read_gold(path, known_ids):
    if not path:
        return {}
    gold_path = _resolve_dir(path)
    with gold_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(GOLD_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("Gold CSV must include " + ", ".join(GOLD_COLUMNS))
        records = list(reader)
    result = {}
    for row in records:
        key = (row["model"],row["query_id"])
        if row["model"] not in MODELS or row["query_id"] not in known_ids:
            raise ValueError("Unknown gold model or query_id: " + str(key))
        if key in result:
            raise ValueError("Duplicate gold model/query_id: " + str(key))
        decision = row["review_status"].strip().lower()
        value = row["is_hallucination_gold"].strip()
        if decision == "approved":
            label = _parse_gold_label(value)
            for field in ("reviewer_id","reviewer_qualification",
                          "independence_declaration","verified_source_evidence"):
                if not row[field].strip():
                    raise ValueError("Approved gold record missing " + field + ": " + str(key))
            if row["independence_declaration"].strip().lower() not in ("yes","true"):
                raise ValueError("Approved gold lacks independent reviewer declaration: " + str(key))
            result[key] = label
        elif decision in ("", "pending", "uncertain", "excluded"):
            if value:
                raise ValueError("Nonapproved row contains a gold label: " + str(key))
        else:
            raise ValueError("Unknown review_status for " + str(key))
    return result


def _parse_gold_label(value):
    v = str(value).strip().lower()
    if v in ("yes","true","1","hallucination"):
        return True
    if v in ("no","false","0","no hallucination"):
        return False
    raise ValueError("Gold label must be Yes or No")


def _doc_texts():
    if not CANDIDATES.is_file():
        raise FileNotFoundError("Missing source candidate corpus " + str(CANDIDATES))
    docs = {}
    for line in CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc["document_id"] in docs:
            raise ValueError("Duplicate candidate document ID")
        docs[doc["document_id"]] = doc["text"]
    # B5/B6 return G0 graph-verbalization document IDs, not phase-2
    # candidates. Reconstruct the exact original G0Corpus wording from the
    # versioned 37-edge snapshot; do not treat IDs as evidence text.
    edges_path = ROOT / "research_artifacts/graph-G0/edges.csv"
    with edges_path.open(newline="", encoding="utf-8") as stream:
        for idx, edge in enumerate(csv.DictReader(stream), 1):
            key = f"G0_EDGE_{idx:04d}"
            if key in docs:
                raise ValueError("Duplicate G0 graph document ID: " + key)
            docs[key] = (
                f'{edge["cause"]} is linked as a cause of '
                f'{edge["effect"]} in the G0 graph snapshot.'
            )
    return docs


def _evidence_text(row, documents):
    """Never substitute document IDs for evidence content."""
    parts = []
    ids = json.loads(row["retrieved_document_ids_json"])
    for identifier in ids:
        if identifier not in documents:
            raise ValueError("Retrieved candidate document not found: " + identifier)
        parts.append(documents[identifier])
    for edge in json.loads(row["direct_causes"]):
        parts.append(f'{edge["cause"]} causes {edge["effect"]}.')
    for path in json.loads(row["multi_hop_paths"]):
        parts.append(" causes ".join(path["nodes"]) + ".")
    return " ".join(parts).strip()


def _cosine_backend(texts, *, batch_size=32):
    """Use only pinned, local SentenceTransformer weights; never download."""
    if not texts:
        return {}
    from .sentence_model_assets import resolve_cached_snapshot
    from sentence_transformers import SentenceTransformer
    snapshot = resolve_cached_snapshot()
    model = SentenceTransformer(str(snapshot))
    ordered = sorted(texts)
    vectors = model.encode(ordered, batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=True)
    return {item: vec for item, vec in zip(ordered, vectors)}


def _cosine(a,b,embedding_map):
    if not a or not b or not embedding_map:
        return None
    va=embedding_map[a]
    vb=embedding_map[b]
    value=float(va @ vb)
    return max(-1.0,min(1.0,value)) if math.isfinite(value) else None


def _f1(precision,recall):
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0


def _classification(model, details):
    selected = [r for r in details
                if r["is_hallucination_gold"] is not None
                and r["hallucination_detector_flag"] is not None]
    with_gold = [r for r in details if r["is_hallucination_gold"] is not None]
    with_detector = [r for r in details if r["hallucination_detector_flag"] is not None]
    base = {
        "model":model,"queries":len(details),"detector_observations":len(with_detector),
        "detector_flags":sum(r["hallucination_detector_flag"] for r in with_detector),
        "detector_flag_rate_not_gold":_mean(
            r["hallucination_detector_flag"] for r in with_detector),
        "gold_approved_count":len(with_gold),
        "gold_hallucination_count":sum(r["is_hallucination_gold"] for r in with_gold),
        "gold_review_coverage_fraction":len(with_gold)/len(details),
        "classification_support":len(selected),
        "classification_status":(
            "unavailable_no_approved_gold" if not with_gold
            else "unavailable_no_detector_predictions" if not selected
            else "complete_132_approved" if len(selected)==132
            else "partial_review_subset_not_full_132"
        ),
        "accuracy":None,"precision":None,"recall":None,"f1":None,
        "tn":None,"fp":None,"fn":None,"tp":None,
    }
    if selected:
        tn=sum(not r["is_hallucination_gold"] and not r["hallucination_detector_flag"] for r in selected)
        fp=sum(not r["is_hallucination_gold"] and r["hallucination_detector_flag"] for r in selected)
        fn=sum(r["is_hallucination_gold"] and not r["hallucination_detector_flag"] for r in selected)
        tp=sum(r["is_hallucination_gold"] and r["hallucination_detector_flag"] for r in selected)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        base.update({"tn":tn,"fp":fp,"fn":fn,"tp":tp,
                     "accuracy":(tp+tn)/len(selected),
                     "precision":precision,"recall":recall,
                     "f1":_f1(precision,recall)})
    return base


def _classification_report_rows(classifier):
    """Per-class precision/recall/F1 like sklearn classification_report.

    Positives = hallucination. Report unavailable with zero approved,
    output-aligned detector+gold observations.
    """
    if not classifier["classification_support"]:
        return []
    tn, fp, fn, tp = (classifier[k] for k in ("tn","fp","fn","tp"))
    total = tn + fp + fn + tp
    p0 = tn/(tn+fn) if tn+fn else 0.0
    r0 = tn/(tn+fp) if tn+fp else 0.0
    p1 = tp/(tp+fp) if tp+fp else 0.0
    r1 = tp/(tp+fn) if tp+fn else 0.0
    rows = [
        {"label":"No Hallucination","precision":p0,"recall":r0,
         "f1_score":_f1(p0,r0),"support":tn+fp},
        {"label":"Hallucination","precision":p1,"recall":r1,
         "f1_score":_f1(p1,r1),"support":fn+tp},
    ]
    rows.append({
        "label":"macro avg",
        "precision":(p0+p1)/2,"recall":(r0+r1)/2,
        "f1_score":(_f1(p0,r0)+_f1(p1,r1))/2,"support":total,
    })
    rows.append({
        "label":"weighted avg",
        "precision":(p0*(tn+fp)+p1*(fn+tp))/total,
        "recall":(r0*(tn+fp)+r1*(fn+tp))/total,
        "f1_score":(_f1(p0,r0)*(tn+fp)+_f1(p1,r1)*(fn+tp))/total,
        "support":total,
    })
    return rows


def _metric_row(name, value, unit="", status="measured", notes=""):
    return {"Metric":name,"Value":value,"Unit":unit,"Status":status,"Notes":notes}


def _summarize(model, base, rows, classification):
    """Use measured run wall time, not sum of reciprocal query latencies."""
    n=len(rows)
    evidence=[r["evidence_present"] for r in rows]
    causal=[r["causal_coverage"] for r in rows]
    depths=[r["causal_chain_depth_nodes"] for r in rows]
    qps=[r["reciprocal_latency_qps_proxy"] for r in rows]
    lat=[r["latency_ms"] for r in rows]
    ans=[r["answer_length_words"] for r in rows if r["answer_present"]]
    changed=[r["refinement_changed_query"] for r in rows if r["refinement_changed_query"] is not None]
    val=[r["validator_passed"] for r in rows if r["validator_passed"] is not None]
    metric = {}
    def put(key,value,unit="",status="measured",notes=""):
        metric[key]=_metric_row(key,value,unit,status,notes)
    put("Queries Evaluated",n,"queries")
    if model in ("B1","B2","B3","B4"):
        put("Causal Retrieval Coverage",None,"fraction","not_applicable",
            "Document retrieval baseline; no causal graph edges reported")
    else:
        put("Causal Retrieval Coverage",_mean(causal),"fraction","operational_proxy",
            "Reported edges/paths; unverified G0, not medical recall")
    put("Causal Chain Depth",_mean(depths),"nodes","structural_proxy",
        "Average node count of returned paths, nonempty path queries only")
    put("Semantic Refinement Score",_mean(r["SRS_Cosine_Proxy"] for r in rows),
        "cosine","embedding_proxy","Original versus recorded refined query; similarity is NOT refinement success")
    put("Context Relevance",_mean(r["Context_Relevance_Cosine_Proxy"] for r in rows),
        "cosine","embedding_proxy","Query vs rewriting or retrieved text; basis differs by pipeline")
    put("Groundedness",_mean(r["Groundedness_Cosine_Proxy"] for r in rows),
        "cosine","embedding_proxy","Cosine similarity only, NOT factual/clinical groundedness")
    put("Average Answer Length",_mean(ans),"words","measured",
        "Answer-present questions only; excludes blank abstentions")
    gold_total=classification["gold_approved_count"]
    put("Hallucination Rate",
        classification["gold_hallucination_count"]/132 if gold_total==132 else None,
        "fraction","gold_only" if gold_total==132 else "not_available",
        "Only with 132 independently approved output-specific labels per model")
    put("Detector Flag Rate (Not Gold)",classification["detector_flag_rate_not_gold"],
        "fraction","detector_proxy" if classification["detector_observations"] else "not_available",
        "B7/B8 detector flags, NOT medically verified hallucinations")
    put("Retrieval Success Rate",_mean(evidence),"fraction","operational_proxy",
        "Any candidate document or graph evidence returned, NOT accuracy")
    put("Average Direct Causes",_mean(r["direct_causes_count"] for r in rows),
        "edges/query","structural_proxy","Reported CAUSES edges across all 132 questions")
    put("Total Direct Causes",sum(r["direct_causes_count"] for r in rows),
        "edges","structural_proxy","G0 edge occurrences, not validated medical truth")
    put("Total Execution Time (s)",base["wall_time_seconds"],"seconds")
    put("Average Latency (s)",_mean(lat)/1000 if lat else None,"seconds")
    put("Average Latency (ms)",_mean(lat),"ms")
    put("Minimum Latency (ms)",min(lat) if lat else None,"ms")
    put("Maximum Latency (ms)",max(lat) if lat else None,"ms")
    put("Median Latency (ms)",statistics.median(lat) if lat else None,"ms")
    put("P95 Latency (ms)",_quantile(lat,.95),"ms")
    put("Average Throughput (QPS)",base["run_throughput_qps"],"queries/sec","measured",
        "132 divided by measured model wall seconds")
    put("Minimum Throughput (QPS, latency proxy)",min(qps) if qps else None,
        "1/sec","reciprocal_latency_proxy","Min of 1 / individual query latency, NOT batch throughput")
    put("Maximum Throughput (QPS, latency proxy)",max(qps) if qps else None,
        "1/sec","reciprocal_latency_proxy","Max of 1 / individual query latency, NOT batch throughput")
    put("Refinement Success Rate",None,"fraction","not_available",
        "Refinement success needs reviewed task-quality references")
    put("Recorded Query Change Rate",_mean(changed),"fraction",
        "observed_text_change" if changed else "not_available",
        "Fraction of available recorded refinements where text differs from original")
    put("Answers Present",sum(r["answer_present"] for r in rows),"queries")
    put("Abstained Queries",sum(r["status"]=="abstained" for r in rows),"queries")
    put("Execution Errors",sum(r["status"]=="error" for r in rows),"queries")
    put("Mean Latency if Answered (ms)",_mean(
        r["latency_ms"] for r in rows if r["status"]=="ok"),"ms")
    put("Mean Latency if Abstained (ms)",_mean(
        r["latency_ms"] for r in rows if r["status"]=="abstained"),"ms")
    put("Number of Causal Paths",sum(r["multi_hop_path_count"] for r in rows),"paths",
        "structural_proxy")
    put("Validation Pass Fraction (Not Gold)",_mean(val),"fraction",
        "validator_proxy" if val else "not_available",
        "Validator PASS is not ground-truth factual correctness")
    assert tuple(metric)==METRIC_NAMES, (list(metric),list(METRIC_NAMES))
    return [metric[k] for k in METRIC_NAMES]


def _compare_numeric(a,b):
    return b-a if isinstance(a,(float,int)) and isinstance(b,(float,int)) else None


def evaluate(source=DEFAULT_SOURCE, output_dir=None, *, semantic_proxies=False,
             gold_csv=None, batch_size=32):
    base,manifest,existing,raw,_=_load(source)
    config = manifest["session_configuration"]
    for path, key in (
        (CANDIDATES, "candidate_corpus_sha256"),
        (ROOT / "research_artifacts/graph-G0/edges.csv", "G0_edges_sha256"),
    ):
        if _hash(path) != config[key]:
            raise ValueError("Source evidence file drift: " + str(path))
    parsed={model:_derive(model,raw[model]) for model in MODELS}
    base_summary={model:_aggregate(model,parsed[model],existing[model]) for model in MODELS}
    identifiers={row["query_id"] for row in raw["B1"]}
    reviewed=_read_gold(gold_csv,identifiers)
    docs=_doc_texts()
    proto={}
    all_text=set()
    for model in MODELS:
        proto[model]=[]
        for r in parsed[model]:
            q=r["query"]
            refined=r["refined_query_text"].strip()
            evidence=_evidence_text(r,docs)
            knowledge=r["rewritten_knowledge"].strip()
            answer=r["final_response"].strip()
            context=knowledge or evidence
            grounding=knowledge or answer
            texts=(q,refined,context,grounding,evidence)
            if semantic_proxies:
                all_text.update(t for t in texts if t)
            proto[model].append((r,texts,
                "rewritten_knowledge" if knowledge else "retrieved_evidence" if evidence else "missing",
                "rewritten_knowledge" if knowledge else "final_response" if answer else "missing"))
    vector_map=_cosine_backend(all_text,batch_size=batch_size) if semantic_proxies else {}
    per={}
    summaries={}
    classifications={}
    detail_by_model={}
    for model in MODELS:
        per[model]=[]
        for r,texts,context_basis,grounding_basis in proto[model]:
            q,refined,context,grounding,evidence=texts
            edge_count=r["_counts"]["num_direct_causes"]
            path_count=r["_counts"]["num_multi_hop_paths"]
            path_depth=r["_depth"]
            status=r["status"]
            latency=r["_timings"]["latency_ms"]
            gold=reviewed.get((model,r["query_id"]))
            changed=(refined.casefold()!=q.strip().casefold()) if refined else None
            details={
                "model":model, "query_id":r["query_id"],"query":q,
                "query_type":r["query_type"],"medical_domain":r["medical_domain"],
                "semantic_group":r["semantic_group"],
                "status":status,"matched_concept":r["matched_concept"],
                "refined_query_text":refined,"final_response":r["final_response"],
                "retrieval_success":r["candidate_evidence_available"],
                "causal_coverage":r["causal_evidence_available"],
                "direct_causes_count":edge_count,
                "multi_hop_path_count":path_count,
                "causal_chain_depth_nodes":path_depth+1 if path_depth is not None else None,
                "answer_length_words":r["answer_length_words"],
                "refinement_recorded":bool(refined or r["refinement_type"]),
                "refinement_changed_query":changed,
                "SRS_Cosine_Proxy":_cosine(q,refined,vector_map),
                "Context_Relevance_Cosine_Proxy":_cosine(q,context,vector_map),
                "Groundedness_Cosine_Proxy":_cosine(grounding,evidence,vector_map),
                "context_basis":context_basis,
                "grounding_basis":grounding_basis,
                "hallucination_detector_flag":r["_detector"],
                "is_hallucination_gold":gold,"validator_passed":r["_validator"],
                "latency_seconds":latency/1000,
                "latency_ms":latency,
                "reciprocal_latency_qps_proxy":1000/latency if latency>0 else None,
                "num_retrieved_documents":r["_counts"]["num_retrieved_documents"],
                "answer_present":r["answer_present"] in ("True","true"),
                "evidence_present":r["candidate_evidence_available"],
            }
            per[model].append(details)
        classifications[model]=_classification(model,per[model])
        summaries[model]=_summarize(model,base_summary[model],
                                    per[model],classifications[model])
    if output_dir:
        out=_resolve_dir(output_dir)
        if out.exists():
            raise ValueError("Refusing to overwrite an existing report directory")
    else:
        out=base/("full_metrics_132_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                  "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def emit(filename,headers,rows):
        dest=out/filename
        _write_csv(dest,headers,rows)
        hashes[filename]=_hash(dest)
    for model in MODELS:
        emit(f"evaluation_metrics_{model}_132_full.csv",DETAIL_FIELDS,per[model])
        emit(f"evaluation_summary_{model}_132_full.csv",
             ("Metric","Value","Unit","Status","Notes"),summaries[model])
        flags=[{
            "model":model,"query_id":r["query_id"],
            "detector_flag":r["hallucination_detector_flag"],
            "approved_gold_label":r["is_hallucination_gold"],
            "reviewed":r["is_hallucination_gold"] is not None,
        } for r in per[model]]
        emit(f"hallucination_predictions_{model}_132.csv",
             ("model","query_id","detector_flag","approved_gold_label","reviewed"),flags)
    compare=[]
    for i,key in enumerate(METRIC_NAMES):
        exemplar=summaries["B1"][i]
        row={"Metric":key,"Unit":exemplar["Unit"],
             "Status":exemplar["Status"],"Notes":exemplar["Notes"]}
        for model in MODELS:
            row[model]=summaries[model][i]["Value"]
        row["B8_minus_B7"]=_compare_numeric(row["B7"],row["B8"])
        compare.append(row)
    emit("comparison_all_models_132_full.csv",
         ["Metric","Unit",*MODELS,"B8_minus_B7","Status","Notes"],compare)
    hal_rows=[classifications[m] for m in MODELS]
    emit("hallucination_classification_summary_132.csv",HAL_COLUMNS,hal_rows)
    for model in MODELS:
        c=classifications[model]
        if c["classification_support"]:
            cm=[
                {"gold":"No Hallucination","predicted_no_hallucination":c["tn"],"predicted_hallucination":c["fp"]},
                {"gold":"Hallucination","predicted_no_hallucination":c["fn"],"predicted_hallucination":c["tp"]},
            ]
            emit(f"confusion_matrix_{model}_132.csv",
                 list(cm[0]),cm)
            classification_rows = _classification_report_rows(c)
            emit(f"classification_report_{model}_132.csv",
                 ("label","precision","recall","f1_score","support"),
                 classification_rows)
    # Paired comparison fully aligned by fixed query ID.
    paired=[]
    for ra,rb in zip(per["B7"],per["B8"]):
        assert ra["query_id"]==rb["query_id"]
        ea,eb=ra["evidence_present"],rb["evidence_present"]
        paired.append({
            "query_id":ra["query_id"],"query":ra["query"],
            "B7_target":ra["matched_concept"],"B8_target":rb["matched_concept"],
            "B7_evidence":ea,"B8_evidence":eb,
            "evidence_category":("both" if ea and eb else "B7_only" if ea
                                 else "B8_only" if eb else "neither"),
            "B7_latency_ms":ra["latency_ms"],"B8_latency_ms":rb["latency_ms"],
            "B8_minus_B7_latency_ms":rb["latency_ms"]-ra["latency_ms"],
            "B7_detector":ra["hallucination_detector_flag"],
            "B8_detector":rb["hallucination_detector_flag"],
        })
    emit("comparison_B7_vs_B8_paired_132_full.csv",list(paired[0]),paired)
    metadata={
        "schema":"THESIS132_FULL_METRICS_V1",
        "source_manifest_sha256":_hash(base/"export_manifest.json"),
        "source_export_dir":str(base.relative_to(ROOT)),
        "input_dataset_sha256":manifest["dataset_sha256"],
        "questions":132,"models":list(MODELS),
        "semantic_proxies_enabled":semantic_proxies,
        "semantic_proxy_encoder_revision":(
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            if semantic_proxies else None),
        "gold_labels_supplied":bool(gold_csv),
        "gold_labels_file_sha256":_hash(_resolve_dir(gold_csv)) if gold_csv else None,
        "labels_reviewed_by_model":{m:classifications[m]["gold_approved_count"]
                                    for m in MODELS},
        "verified_medical_quality":False,
        "caveats":[
            "SRS/context relevance/groundedness here are embedding cosine similarity proxies only.",
            "No unreviewed labels or pipeline detector flags are treated as medical ground truth.",
            "Accuracy/Precision/Recall/F1 require independently reviewed output-specific labels.",
            "B1-B6 have no hallucination detector flags; classification unavailable even with gold.",
            "B1-B4 source candidate text is not the same evidence base as B5-B8 G0 edges/paths.",
            "Query change is not query-refinement success. Legacy RL iteration counts may be missing.",
            "Min/max per-query reciprocal latency are NOT measured run throughput.",
            "These 132 queries were development-exposed; not an external clinical holdout.",
        ],
        "file_sha256":hashes,
    }
    (out/"full_metrics_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",
                                                   encoding="utf-8")
    return out,metadata


def _read_report(out):
    out=_resolve_dir(out)
    report=json.loads((out/"full_metrics_manifest.json").read_text(encoding="utf-8"))
    if report.get("schema")!="THESIS132_FULL_METRICS_V1":
        raise ValueError("Unexpected evaluation manifest")
    for filename,sha in report["file_sha256"].items():
        if _hash(out/filename)!=sha:
            raise ValueError("Report file changed: "+filename)
    return out,report


def _metric_display(value,name):
    if value is None or value=="":
        return "N/A"
    try:
        n=float(value)
    except (ValueError,TypeError):
        return str(value)
    if name in ("Causal Retrieval Coverage","Retrieval Success Rate","Hallucination Rate",
                "Detector Flag Rate (Not Gold)","Recorded Query Change Rate",
                "Refinement Success Rate","Validation Pass Fraction (Not Gold)"):
        return f"{100*n:.2f}%"
    if name in ("Queries Evaluated","Answers Present","Abstained Queries",
                "Execution Errors","Total Direct Causes","Number of Causal Paths"):
        return str(int(n))
    return f"{n:.4f}"


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def display_model(report_dir,model):
    folder,_=_read_report(report_dir)
    if model not in MODELS:
        raise ValueError("Use B1-B8")
    rows=_read_csv(folder/f"evaluation_summary_{model}_132_full.csv")
    d={r["Metric"]:r for r in rows}
    def section(title,fields):
        print("\n"+"="*70)
        print(title.center(70))
        print("="*70)
        for name in fields:
            r=d[name]
            print(f"{name:<42}: {_metric_display(r['Value'],name):>16}")
    print("\nMODEL:",model,"| fixed 132 development questions")
    section("QUALITY / SIMILARITY PROXIES",METRIC_NAMES[1:9])
    section("RETRIEVAL METRICS",("Retrieval Success Rate",
            "Average Direct Causes","Total Direct Causes","Number of Causal Paths"))
    section("PERFORMANCE METRICS",METRIC_NAMES[12:22])
    section("QUERY REFINEMENT METRICS",METRIC_NAMES[22:24])
    section("EXECUTION STATUS",METRIC_NAMES[24:])
    display_hallucination(report_dir,model)
    print("\nPer-query metrics:",folder/f"evaluation_metrics_{model}_132_full.csv")
    print("Aggregate summary:",folder/f"evaluation_summary_{model}_132_full.csv")


def display_comparison(report_dir):
    folder,_=_read_report(report_dir)
    rows=_read_csv(folder/"comparison_all_models_132_full.csv")
    print("\n"+"="*122)
    print("B1–B8 / 132-QUESTION EVALUATION COMPARISON".center(122))
    print("="*122)
    print(f"{'Metric':<37}"+ "".join(f"{m:>10}" for m in MODELS))
    for row in rows:
        if row["Metric"] in ("Queries Evaluated","Refinement Success Rate"):
            continue
        print(f"{row['Metric'][:37]:<37}"+
              "".join(f"{_metric_display(row[m],row['Metric']):>10}" for m in MODELS))
    print("\nB7 / B8 direct comparison")
    print(f"{'Metric':<44} {'B7 No-RL':>16} {'B8 PPO RL':>16} {'B8-B7':>16}")
    for row in rows:
        print(f"{row['Metric'][:44]:<44} "+
              f"{_metric_display(row['B7'],row['Metric']):>16} "+
              f"{_metric_display(row['B8'],row['Metric']):>16} "+
              f"{_metric_display(row['B8_minus_B7'],row['Metric']):>16}")
    print("\nN/A = unmeasured/unavailable, NOT zero.")
    print("Context/Groundedness/SRS are embedding similarity proxies, not clinical accuracy.")
    print("Comparison CSV:",folder/"comparison_all_models_132_full.csv")


def display_hallucination(report_dir,model):
    folder,_=_read_report(report_dir)
    c=next(r for r in _read_csv(folder/"hallucination_classification_summary_132.csv")
           if r["model"]==model)
    print("\n"+"="*70)
    print(f"HALLUCINATION DETECTION EVALUATION — {model}".center(70))
    print("="*70)
    print("Total queries           :", c["queries"])
    print("Detector observations   :", c["detector_observations"])
    print("Detector flagged (not gold):", c["detector_flags"])
    print("Approved gold labels    :", c["gold_approved_count"])
    print("Classification status   :", c["classification_status"])
    if not c["classification_support"] or c["classification_support"]=="0":
        print("Accuracy, Precision, Recall, F1: N/A — requires approved labels and detector predictions")
    else:
        print("Classification support  :", c["classification_support"],
              "(subset only)" if c["classification_status"].startswith("partial") else "(complete)")
        for field in ("accuracy","precision","recall","f1"):
            print(f"{field.title():<23}: {float(c[field]):.4f}")
        print("Confusion Matrix [No Hallucination, Hallucination]:")
        print(f"[[{c['tn']} {c['fp']}]")
        print(f" [{c['fn']} {c['tp']}]]")
        print("TP:",c["tp"],"TN:",c["tn"],"FP:",c["fp"],"FN:",c["fn"])
        print("\nClassification Report:")
        print(f"{'Class':<22}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>10}")
        numeric = {key: float(c[key]) if key in ("accuracy","precision","recall","f1") else int(c[key])
                   for key in ("accuracy","precision","recall","f1","tn","fp","fn","tp")}
        for row in _classification_report_rows(numeric):
            print(f"{row['label']:<22}{row['precision']:>12.3f}{row['recall']:>12.3f}"
                  f"{row['f1_score']:>12.3f}{row['support']:>10}")
    print("This does not evaluate false or correct medical claims without external review.")
