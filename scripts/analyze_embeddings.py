from __future__ import annotations

import argparse

from lwow.config import load_config
from lwow.io import read_csv, write_csv
from lwow.analysis import EmbeddingAnalyzer
from lwow.clients.embeddings import EmbeddingClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze embedding distances.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    edge_rows = read_csv(config.paths.edge_list_output_path)
    edges = [(row["src"], row["tgt"], int(row["wt"])) for row in edge_rows]

    client = EmbeddingClient(
        provider=config.embeddings.provider,
        model=config.embeddings.model,
        input_type=config.embeddings.input_type,
        model_path=config.embeddings.model_path,
        model_name=config.embeddings.model_name,
        oov_strategy=config.embeddings.oov_strategy,
        batch_size=config.embeddings.batch_size,
    )
    analyzer = EmbeddingAnalyzer(client=client)
    distances = analyzer.compute_distances(edges)
    stats = analyzer.correlate(distances)
    stats["missing_words"] = len(set(analyzer.missing_words))
    stats["missing_edges"] = analyzer.missing_edges
    write_csv(
        config.paths.embedding_analysis_path,
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in stats.items()],
    )

    outliers = analyzer.find_outliers(
        distances,
        count_percentile=config.analysis.count_percentile,
        distance_percentile=config.analysis.distance_percentile,
    )
    write_csv(
        config.paths.outliers_path,
        ["src", "tgt", "wt", "distance"],
        [
            {"src": src, "tgt": tgt, "wt": wt, "distance": dist}
            for src, tgt, wt, dist in outliers
        ],
    )


if __name__ == "__main__":
    main()
