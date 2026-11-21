"""Export one or more existing fixed-132 B1-B8 sessions to separate CSVs.

Examples:
    python export_batch_metrics_132.py --session-dir presentation_20261118/thesis132/runs/batch_XXXX
    python export_batch_metrics_132.py --session-dir SESSION_1 --session-dir SESSION_2 --require-all

No models are executed. Each source session and its 132 records are verified.
"""
from research_v2.thesis132_csv_export import main

if __name__ == "__main__":
    main()
