#!/usr/bin/env python3
"""Make a candidate question_review.csv from a user's genuine review worksheet.

No decisions or identities are filled; copies only draft answer, source IDs, and
clearly marked AI review notes to six questions. Does not modify originals.
"""
import argparse
import csv
from pathlib import Path

TARGETS={'PILOT_001','PILOT_016','PILOT_004','PILOT_019','PILOT_007','PILOT_022'}

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--review-dir',required=True,type=Path)
 p.add_argument('--drafts',type=Path,default=Path(__file__).resolve().parent/'six_question_answer_drafts.csv')
 args=p.parse_args()
 target=args.review_dir/'question_review.csv'
 with args.drafts.open(encoding='utf-8',newline='') as fh:
  drafts={r['query_id']:r for r in csv.DictReader(fh)}
 assert set(drafts)==TARGETS, 'Expected exactly six drafts'
 with target.open(encoding='utf-8',newline='') as fh:
  reader=csv.DictReader(fh);fields=reader.fieldnames;rows=list(reader)
 assert fields and all(k in fields for k in ['query_id','query','candidate_source_id','answerability_decision','draft_reference_answer','supporting_source_ids','reviewer_id','reviewed_date','review_note']), 'Unexpected question_review.csv schema'
 assert len(rows)==30, f'Expected 30 questions, got {len(rows)}'
 ids=[r['query_id'] for r in rows]
 assert len(set(ids))==len(ids) and TARGETS.issubset(ids), 'Missing or duplicate question IDs'
 for r in rows:
  q=r['query_id']
  if q not in TARGETS:continue
  draft=drafts[q]
  assert r['query']==draft['query'] and r['candidate_source_id']==draft['candidate_source_id'], f'Immutable question/source mismatch: {q}'
  assert not r['answerability_decision'].strip() and not r['reviewer_id'].strip() and not r['reviewed_date'].strip(), f'{q} already has reviewer decision; refusing overwrite'
  # Keep unreviewed status. Do not modify source-linked immutable columns.
  r['draft_reference_answer']=draft['draft_reference_answer']
  r['supporting_source_ids']=draft['supporting_source_ids']
  r['review_note']=draft['ai_review_note']
 output=args.review_dir/'question_review_ai_prepared.csv'
 if output.exists():raise SystemExit(f'Output exists: {output}; move it away first')
 with output.open('w',newline='',encoding='utf-8') as fh:
  writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 print(f'Prepared {output}: 30 original questions, 6 AI reference-answer drafts, 30 decisions still unchanged')
 print('Audit-compatible question_review.csv replacement can be made only after checking the source notes.')

if __name__=='__main__': main()
