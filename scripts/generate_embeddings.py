#!/usr/bin/env python3
"""Interactive script to generate OpenAI embeddings from edgelist CSV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

from lwow.clients.embeddings import EmbeddingClient
from lwow.embedding_store import (
    EmbeddingStore,
    OPENAI_EMBEDDING_PRICES,
    _model_to_suffix,
)

# cl100k_base is used by text-embedding-3-small/large
_TIKTOKEN_ENCODING = "cl100k_base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embeddings for unique words from an edgelist CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to edgelist CSV (columns: src, tgt, wt)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help="OpenAI embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Words per API request (default: 100)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompts; embed all pending words",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of embeddings to generate (default: all pending)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    store_path = EmbeddingStore.store_path_for_csv(csv_path)
    suffix = _model_to_suffix(args.model)
    embeddings_file = "embeddings.csv" if suffix == "large" else "embeddings_small.csv"
    config_file = "config.json" if suffix == "large" else "config_small.json"
    store = EmbeddingStore(
        store_path,
        embeddings_filename=embeddings_file,
        config_filename=config_file,
    )

    total = store.ensure_from_edgelist(csv_path, model=args.model)
    embedded_count = store.count_embedded()
    pending_count = total - embedded_count

    # Migrate 2-col to 3-col (add tokens) before any logic
    enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    if store.migrate_add_tokens_column(token_counter=lambda w: len(enc.encode(w))):
        print("Migrated to add tokens column.")
        embedded_count = store.count_embedded()
        pending_count = total - embedded_count

    print(f"Loaded {total:,} unique words from {csv_path}")
    print(f"Store: {store_path}")
    print(f"Pending: {pending_count:,} | Already embedded: {embedded_count:,}")

    if pending_count == 0:
        print("No pending embeddings. Done.")
        return

    if args.non_interactive:
        to_embed = min(pending_count, args.limit or pending_count)
    else:
        default = str(pending_count)
        prompt = f"How many embeddings to generate? [{default}]: "
        try:
            raw = input(prompt).strip() or default
        except EOFError:
            raw = default
        try:
            to_embed = int(raw)
        except ValueError:
            print("Invalid number. Exiting.")
            sys.exit(1)
        if to_embed <= 0:
            print("Nothing to embed.")
            return
        to_embed = min(to_embed, pending_count)
        if args.limit is not None:
            to_embed = min(to_embed, args.limit)

    words = store.get_pending_words(limit=to_embed)
    if not words:
        print("No pending words to embed.")
        return

    # Ensure API key
    if not os.getenv("OPENAI_API_KEY"):
        if Path(".env").exists():
            try:
                with open(".env") as f:
                    for line in f:
                        if "=" in line and not line.strip().startswith("#"):
                            k, _, v = line.strip().partition("=")
                            if k.strip().upper() == "OPENAI_API_KEY" and v.strip():
                                os.environ["OPENAI_API_KEY"] = v.strip().strip("'\"")
                                break
            except OSError:
                pass
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set.")
            sys.exit(1)

    client = EmbeddingClient(
        provider="openai",
        model=args.model,
        input_type="document",  # unused for OpenAI
    )
    price_per_mtok = OPENAI_EMBEDDING_PRICES.get(args.model, 0.02) / 1_000_000

    enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    batch_size = args.batch_size
    total_tokens = 0
    done = 0

    print(f"\nEmbedding {len(words):,} words (batch size {batch_size})...")

    for start in range(0, len(words), batch_size):
        batch = words[start : start + batch_size]
        embeddings, usage = client.embed_texts_with_usage(batch)
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens += prompt_tokens
            spend_delta = prompt_tokens * price_per_mtok
            store.update_config(prompt_tokens_delta=prompt_tokens, total_spend_delta=spend_delta)

        # Per-word token counts for average pooling (using tiktoken, same as API)
        token_counts = [len(enc.encode(w)) for w in batch]
        items = [(w, emb, tc) for w, emb, tc in zip(batch, embeddings, token_counts)]
        store.append_embeddings(items)

        done += len(batch)
        est_spend = total_tokens * price_per_mtok
        pct = 100 * done / len(words) if words else 0
        bar_len = 30
        filled = int(bar_len * done / len(words)) if words else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r[{bar}] {done:,}/{len(words):,} ({pct:.0f}%) | "
            f"Tokens: {total_tokens:,} | Est. ${est_spend:.6f}",
            end="",
            flush=True,
        )

    print()
    cfg = store.get_config()
    print(f"Done. Total embedded: {store.count_embedded():,}. Spend: ${cfg.get('total_spend_usd', 0):.6f}")


if __name__ == "__main__":
    main()
