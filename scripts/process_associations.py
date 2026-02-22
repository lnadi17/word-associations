from __future__ import annotations

import argparse

from lwow.config import load_config
from lwow.io import read_csv, write_csv
from lwow.processing import ProcessingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw association outputs.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_csv(config.paths.raw_output_path)

    processor = ProcessingRunner(
        repetitions_per_cue=config.processing.repetitions_per_cue,
        seed=config.processing.seed,
    )
    processed = processor.process_rows(rows)
    write_csv(
        config.paths.processed_output_path,
        ["cue", "trial", "responses", "raw_text"],
        [
            {
                "cue": row["cue"],
                "trial": row["trial"],
                "responses": "|".join(row["responses"]),
                "raw_text": row["raw_text"],
            }
            for row in processed
        ],
    )

    edges = processor.build_edge_counts(processed)
    write_csv(
        config.paths.edge_list_output_path,
        ["src", "tgt", "wt"],
        [{"src": src, "tgt": tgt, "wt": wt} for src, tgt, wt in edges],
    )


if __name__ == "__main__":
    main()
