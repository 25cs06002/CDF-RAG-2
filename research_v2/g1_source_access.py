"""Check candidate health-agency source URL access; not medical fact verification.

Runs on the user's Ubuntu machine. Metadata-only: never stores full source
HTML or republishes copyrighted text. Use the resulting CSV as a source
review reading log, not as support for G1 relationships.
"""
import argparse
import csv
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from .experiment_runner import PROJECT_ROOT, file_sha256

COLUMNS = ("source_id","source_url","http_status","final_url","content_type",
           "retrieved_utc","access_status","content_sha256","bytes_read",
           "error_type","source_title","review_status")


def check_one(row, timeout=12, max_bytes=2_000_000):
    url=row["source_url"]
    if not url.startswith("https://"):
        return {**row,"access_status":"invalid_https","http_status":"",
                "content_sha256":"","bytes_read":0,"error_type":"invalid_scheme",
                "final_url":"","content_type":"",
                "retrieved_utc":datetime.now(timezone.utc).isoformat()}
    info={**row,"http_status":"","final_url":"","content_type":"",
          "retrieved_utc":datetime.now(timezone.utc).isoformat(),
          "access_status":"unchecked","content_sha256":"","bytes_read":0,
          "error_type":""}
    try:
        with requests.get(
            url,timeout=timeout,stream=True,
            headers={"User-Agent":"CDF-RAG-2-source-provenance-audit/1.0"},
            allow_redirects=True
        ) as res:
            info["http_status"]=res.status_code
            info["final_url"]=res.url
            info["content_type"]=res.headers.get("content-type","")
            if not res.ok:
                info["access_status"]="http_error"
                return info
            digest=hashlib.sha256()
            total=0
            for chunk in res.iter_content(chunk_size=32_768):
                if not chunk:continue
                if total+len(chunk)>max_bytes:
                    chunk=chunk[:max_bytes-total]
                digest.update(chunk);total+=len(chunk)
                if total>=max_bytes:break
            info["bytes_read"]=total
            info["content_sha256"]=digest.hexdigest() if total else ""
            info["access_status"]="partial_bytes_hashed" if total>=max_bytes else "reachable"
    except requests.exceptions.RequestException as exc:
        info["access_status"]="request_failed"
        info["error_type"]=type(exc).__name__
    return info


def run(workspace: Path, limit: int = 25, timeout: int = 12):
    folder=Path(workspace).expanduser().resolve()
    if not folder.is_relative_to(PROJECT_ROOT):
        raise ValueError("Review workspace must be within repository")
    path=folder/"source_review.csv"
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError("No candidate sources")
    if limit<1 or timeout<1:raise ValueError("Positive limit and timeout required")
    out=folder/("source_access_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    checked=[check_one(row,timeout) for row in rows[:limit]]
    with (out/"source_access.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction="ignore")
        writer.writeheader();writer.writerows(checked)
    summary={
        "tested":len(checked),
        "input_source_review_sha256":file_sha256(path),
        "source_content_saved":False,
        "source_claims_verified":False,
        "reachable_count":sum(r["access_status"] in {"reachable","partial_bytes_hashed"} for r in checked),
        "status_counts":{status:sum(r["access_status"]==status for r in checked)
                        for status in sorted({r["access_status"] for r in checked})},
        "warning":"URL reachability and fetched-byte hashes do not verify clinical statements, evidence type, direction or licensing."
    }
    (out/"access_manifest.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**summary},indent=2))
    return out,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--limit",type=int,default=25)
    p.add_argument("--timeout",type=int,default=12)
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:
        p.error("Run from repository root")
    run(a.folder,limit=a.limit,timeout=a.timeout)


if __name__=="__main__":main()
