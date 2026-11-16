#!/usr/bin/env python3
"""Repair remaining70 question-review *editable* fields against current reviewed sources.

Run from the CDF-RAG-2 repository root:
    python3 reconcile_remaining70_audit.py --audit-dir "$REVIEW70_DIR"

Does not modify the source/link review files or any protected candidate fields.
Backs up question_review.csv, and retains original header/row order.
Does not invent medical source support or alter source decisions.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile


def load_csv(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def split_ids(text):
    return [x.strip() for x in text.split(';') if x.strip()]


def validate_protected(expected, actual, key, editable):
    if len(expected) != len(actual):
        raise ValueError(f'{key}: row counts differ: {len(expected)} vs {len(actual)}')
    if list(expected[0]) != list(actual[0]):
        raise ValueError(f'{key}: original header/order differs')
    refs = {r[key]: r for r in expected}
    if len(refs) != len(expected):
        raise ValueError(f'{key}: duplicate candidate keys')
    for r in actual:
        k = r[key]
        if k not in refs:
            raise ValueError(f'{key}: unexpected row {k}')
        for field, value in refs[k].items():
            if field not in editable and r[field] != value:
                raise ValueError(f'Protected field mismatch {k}.{field}: expected {value!r}, actual {r[field]!r}. Restore original value before continuing.')


def check_answerable(row, source_by_qid, link_by_qid):
    qid = row['query_id']
    reasons = []
    ids = split_ids(row['supporting_source_ids'])
    sources = source_by_qid.get(qid, {})
    if not row['draft_reference_answer'].strip() or not ids:
        reasons.append('missing a reference answer or supporting source IDs')
    for sid in ids:
        status = sources.get(sid, {}).get('review_decision', 'unregistered/unreviewed')
        if status != 'supports':
            reasons.append(f'{sid} is {status!r}, not supports')
    if row['query_type'] == 'unsupported':
        reasons.append('unsupported individual prediction cannot be labeled answerable')
    if row['needs_additional_sources'] == 'true' and not row['scope_resolution_note'].strip():
        reasons.append('source-discovery gap has no scope_resolution_note')
    if row['query_type'] == 'risk_factor' and not row['risk_vs_cause_note'].strip():
        reasons.append('risk-versus-causation note not reviewed')
    if row['query_type'] == 'comparative':
        for field in ('comparison_topic_a', 'comparison_topic_b', 'comparison_evidence_note'):
            if not row[field].strip():
                reasons.append(f'missing {field}')
        a,b=split_ids(row['comparison_source_ids_a']),split_ids(row['comparison_source_ids_b'])
        if not a or not b or not set(a+b).issubset(set(ids)):
            reasons.append('comparison lacks two registered source evidence sides')
    if row['query_type'] == 'multi_hop':
        status=row['candidate_path_status']
        if status not in ('two_links_source_supported','partial_links_supported','no_supported_chain'):
            reasons.append('missing/invalid candidate_path_status')
        lrows=link_by_qid.get(qid,[])
        if len(lrows)!=2 or any(not x['review_decision'].strip() for x in lrows):
            reasons.append('two link-review slots have not both been reviewed')
        else:
            supported=[x for x in lrows if x['review_decision']=='supports']
            if status=='two_links_source_supported':
                if len(supported)!=2:
                    reasons.append('two-link status does not have two supported link slots')
                elif supported[0]['effect_node'].casefold().strip()!=supported[1]['cause_node'].casefold().strip():
                    reasons.append('two supported links do not connect')
            if status=='partial_links_supported' and len(supported)!=1:
                reasons.append('partial-link status does not have one supported slot')
    return reasons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',required=True,type=Path)
    p.add_argument('--dry-run',action='store_true',help='Print changes without saving')
    args=p.parse_args()
    from research_v2.remaining_review_workspace import (
        build_remaining_review_tables, QUESTION_EDITABLE, SOURCE_EDITABLE,
        LINK_EDITABLE, audit_remaining_review_workspace,
    )
    root=args.audit_dir
    exp_q,exp_s,exp_l=build_remaining_review_tables()
    q_path=root/'question_review.csv'
    s_path=root/'source_review.csv'
    l_path=root/'link_review.csv'
    qhead, qs=load_csv(q_path)
    shead, ss=load_csv(s_path)
    lhead, ls=load_csv(l_path)
    for name, header, current, baseline, key, editable in (
        ('question',qhead,qs,exp_q,'query_id',QUESTION_EDITABLE),
        ('source',shead,ss,exp_s,'source_task_id',SOURCE_EDITABLE),
        ('link',lhead,ls,exp_l,'link_task_id',LINK_EDITABLE),
    ):
        if header != list(baseline[0]):
            raise ValueError(f'{name}: original headers/order changed')
        validate_protected(baseline,current,key,editable)
    source_by_qid={}
    for s in ss:
        source_by_qid.setdefault(s['query_id'],{})[s['candidate_source_id']]=s
    link_by_qid={}
    for l in ls:
        link_by_qid.setdefault(l['query_id'],[]).append(l)
    for slots in link_by_qid.values():
        slots.sort(key=lambda x: int(x['link_slot']))

    changes=[]
    for q in qs:
        qid=q['query_id']
        if q['answerability_decision'] == 'answerable':
            reasons=check_answerable(q,source_by_qid,link_by_qid)
            if reasons:
                if q['query_type']=='unsupported':
                    q['answerability_decision']='unanswerable'
                    q['abstention_reason'] = q['abstention_reason'] or 'The question asks for an exact individual outcome without clinical evidence.'
                    q['draft_reference_answer'] = q['draft_reference_answer'] or 'Cannot determine an exact individual outcome from the information provided.'
                else:
                    q['answerability_decision']='needs_more_evidence'
                    q['draft_reference_answer']=''
                    q['supporting_source_ids']=''
                rationale='Audit consistency correction: ' + '; '.join(reasons) + '. Retain original candidate fields; no unverified source upgraded.'
                q['review_note'] = (q['review_note'].rstrip()+' ' if q['review_note'].strip() else '') + rationale
                changes.append((qid, q['answerability_decision'],reasons))
        if q['answerability_decision']=='unanswerable' and not q['abstention_reason'].strip():
            q['abstention_reason']=q['scope_note'] or 'Insufficient individual clinical data to establish the requested diagnosis or prediction.'
            changes.append((qid,'added required abstention_reason',[]))

    print('Questions:',len(qs),' Sources:',len(ss),' Link slots:',len(ls))
    print('Decisions:',dict(Counter(q['answerability_decision'] for q in qs)))
    print('Rows requiring correction:',len(set(c[0] for c in changes)))
    for qid, action, reasons in changes:
        print(f'  {qid}: {action} — {"; ".join(reasons)}')
    if args.dry_run:
        print('DRY RUN: no files modified.')
        return
    if changes:
        # Explicitly confirm no protected fields changed by this program.
        validate_protected(exp_q,qs,'query_id',QUESTION_EDITABLE)
        stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
        backup=q_path.with_name('question_review.csv.bak-'+stamp)
        shutil.copy2(q_path,backup)
        fd,tempfile_path=tempfile.mkstemp(prefix='.question_review.',suffix='.csv',dir=root)
        try:
            with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=qhead)
                writer.writeheader()
                writer.writerows(qs)
            os.replace(tempfile_path,q_path)
        finally:
            if os.path.exists(tempfile_path):
                os.unlink(tempfile_path)
        print('Backup:',backup)
        print('Updated:',q_path)
    else:
        print('No question corrections needed.')
    # This runs the same audit as the user command. No --strict: pending review slots are permitted.
    profile=audit_remaining_review_workspace(root)
    print('AUDIT PASSED:',profile['question_decision_counts'])

if __name__=='__main__':
    main()
