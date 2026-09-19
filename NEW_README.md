python3 -m venv venv
source venv/bin/activate

#python evaluate_batch_queries.py
#python evaluate_batch_queries_no_rl.py

python evaluate_metrics_no_rl.py
python evaluate_metrics_rl.py

python compare_hallucination_no_rl.py
python compare_hallucination_rl.py

python compare_metrics_rl_vs_no_rl.py
python compare_summary_rl_vs_no_rl.py