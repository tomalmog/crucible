#!/usr/bin/env python3
"""Phase 1: Comprehensive dataset operations test suite for Crucible.

Tests all ingest operations (JSONL, Parquet, CSV, plain text, dedup, unicode),
remote push signature verification, and dataset curation (score, stats).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

# Ensure src/ is on the path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Set a temporary data root so we don't pollute the real .crucible/
_TMP_ROOT = tempfile.mkdtemp(prefix="crucible_phase1_")
os.environ["CRUCIBLE_DATA_ROOT"] = _TMP_ROOT

from core.config import CrucibleConfig
from core.types import IngestOptions
from store.dataset_sdk import CrucibleClient


@dataclass
class TestResult:
    test_id: str
    status: str  # PASS or FAIL
    details: str


RESULTS: list[TestResult] = []


def record_result(test_id: str, status: str, details: str) -> None:
    RESULTS.append(TestResult(test_id=test_id, status=status, details=details))
    marker = "PASS" if status == "PASS" else "FAIL"
    print(f"  [{marker}] {test_id}: {details}")


def make_client() -> CrucibleClient:
    config = CrucibleConfig(
        data_root=Path(_TMP_ROOT),
        s3_region=None,
        s3_profile=None,
        random_seed=42,
    )
    return CrucibleClient(config)


# ─────────────────────────────────────────────────────────────
# 1.1.1  Ingest JSONL with text field
# ─────────────────────────────────────────────────────────────
def test_1_1_1() -> None:
    test_id = "1.1.1"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t111_")
        jsonl_path = Path(tmpdir) / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Hello world document {i}"}) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-1", source_uri=str(jsonl_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 10:
            record_result(test_id, "FAIL", f"Expected 10 records, got {manifest.record_count}")
            return
        if len(records) != 10:
            record_result(test_id, "FAIL", f"Expected 10 loaded records, got {len(records)}")
            return
        # Verify text content
        texts = sorted([r.text for r in records])
        expected = sorted([f"Hello world document {i}" for i in range(10)])
        if texts != expected:
            record_result(test_id, "FAIL", f"Text content mismatch: got {texts[:2]}...")
            return
        record_result(test_id, "PASS", "10 records ingested correctly from JSONL with text field")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.2  Ingest JSONL with prompt/response
# ─────────────────────────────────────────────────────────────
def test_1_1_2() -> None:
    test_id = "1.1.2"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t112_")
        jsonl_path = Path(tmpdir) / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-2", source_uri=str(jsonl_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 10:
            record_result(test_id, "FAIL", f"Expected 10 records, got {manifest.record_count}")
            return
        # Verify prompt/response joined with newline
        found_joined = False
        for r in records:
            if "\n" in r.text:
                parts = r.text.split("\n")
                if parts[0].startswith("Q") and parts[1].startswith("A"):
                    found_joined = True
                    break
        if not found_joined:
            record_result(test_id, "FAIL", f"Records not joined with newline. Sample: {records[0].text!r}")
            return
        record_result(test_id, "PASS", "10 prompt/response records ingested, joined with newline")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.3  Ingest JSONL with custom --text-field
# ─────────────────────────────────────────────────────────────
def test_1_1_3() -> None:
    test_id = "1.1.3"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t113_")
        jsonl_path = Path(tmpdir) / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"body": f"Custom field text {i}", "category": "test"}) + "\n")
        client = make_client()
        opts = IngestOptions(
            dataset_name="test-1-1-3", source_uri=str(jsonl_path), text_field="body",
        )
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 5:
            record_result(test_id, "FAIL", f"Expected 5 records, got {manifest.record_count}")
            return
        # Check text extracted from body
        sample = records[0]
        if "Custom field text" not in sample.text:
            record_result(test_id, "FAIL", f"Text not from body field. Got: {sample.text!r}")
            return
        # Check category preserved in extras
        extras = dict(sample.metadata.extra_fields)
        if extras.get("category") != "test":
            record_result(test_id, "FAIL", f"Category not preserved in extras. Extras: {extras}")
            return
        record_result(test_id, "PASS", "5 records ingested with custom text_field='body', category preserved")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.4  Ingest Parquet with text column
# ─────────────────────────────────────────────────────────────
def test_1_1_4() -> None:
    test_id = "1.1.4"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        record_result(test_id, "FAIL", "pyarrow not installed, cannot test parquet ingest")
        return
    try:
        tmpdir = tempfile.mkdtemp(prefix="t114_")
        parquet_path = Path(tmpdir) / "data.parquet"
        texts = [f"Parquet text record {i}" for i in range(8)]
        table = pa.table({"text": texts})
        pq.write_table(table, parquet_path)
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-4", source_uri=str(parquet_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 8:
            record_result(test_id, "FAIL", f"Expected 8 records, got {manifest.record_count}")
            return
        loaded_texts = sorted([r.text for r in records])
        expected = sorted(texts)
        if loaded_texts != expected:
            record_result(test_id, "FAIL", f"Text mismatch. Got: {loaded_texts[:2]}...")
            return
        record_result(test_id, "PASS", "8 parquet records ingested correctly")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.5  Ingest Parquet with all-null text column
# ─────────────────────────────────────────────────────────────
def test_1_1_5() -> None:
    test_id = "1.1.5"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        record_result(test_id, "FAIL", "pyarrow not installed")
        return
    try:
        from core.errors import CrucibleIngestError
        tmpdir = tempfile.mkdtemp(prefix="t115_")
        parquet_path = Path(tmpdir) / "null_data.parquet"
        table = pa.table({"text": pa.array([None, None, None], type=pa.string())})
        pq.write_table(table, parquet_path)
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-5", source_uri=str(parquet_path))
        try:
            client.ingest(opts)
            record_result(test_id, "FAIL", "Expected CrucibleIngestError but ingest succeeded")
        except CrucibleIngestError as e:
            record_result(test_id, "PASS", f"CrucibleIngestError raised correctly: {str(e)[:80]}")
        except Exception as e:
            record_result(test_id, "FAIL", f"Wrong exception type: {type(e).__name__}: {e}")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.6  Ingest Parquet with custom --text-field, extras preserved
# ─────────────────────────────────────────────────────────────
def test_1_1_6() -> None:
    test_id = "1.1.6"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        record_result(test_id, "FAIL", "pyarrow not installed")
        return
    try:
        tmpdir = tempfile.mkdtemp(prefix="t116_")
        parquet_path = Path(tmpdir) / "body_data.parquet"
        table = pa.table({
            "body": ["Body text one", "Body text two", "Body text three"],
            "label": ["cat", "dog", "bird"],
        })
        pq.write_table(table, parquet_path)
        client = make_client()
        opts = IngestOptions(
            dataset_name="test-1-1-6", source_uri=str(parquet_path), text_field="body",
        )
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 3:
            record_result(test_id, "FAIL", f"Expected 3 records, got {manifest.record_count}")
            return
        # Check extras preserved
        sample = records[0]
        extras = dict(sample.metadata.extra_fields)
        if "label" not in extras:
            record_result(test_id, "FAIL", f"'label' not in extras: {extras}")
            return
        if "Body text" not in sample.text:
            record_result(test_id, "FAIL", f"Text not from body field: {sample.text!r}")
            return
        record_result(test_id, "PASS", "3 parquet records with custom text_field='body', label preserved")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.7  Ingest CSV
# ─────────────────────────────────────────────────────────────
def test_1_1_7() -> None:
    test_id = "1.1.7"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t117_")
        csv_path = Path(tmpdir) / "data.csv"
        with open(csv_path, "w") as f:
            f.write("text,name,category\n")
            for i in range(6):
                f.write(f"CSV text record {i},item{i},cat{i}\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-7", source_uri=str(csv_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 6:
            record_result(test_id, "FAIL", f"Expected 6 records, got {manifest.record_count}")
            return
        # Check text content
        texts = sorted([r.text for r in records])
        expected = sorted([f"CSV text record {i}" for i in range(6)])
        if texts != expected:
            record_result(test_id, "FAIL", f"Text mismatch. Got: {texts[:2]}")
            return
        # Check extras
        sample = records[0]
        extras = dict(sample.metadata.extra_fields)
        if "name" not in extras or "category" not in extras:
            record_result(test_id, "FAIL", f"Extras missing name/category: {extras}")
            return
        record_result(test_id, "PASS", "6 CSV records ingested with name/category extras")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.8  Ingest plain text files
# ─────────────────────────────────────────────────────────────
def test_1_1_8() -> None:
    test_id = "1.1.8"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t118_")
        for i in range(3):
            txt_path = Path(tmpdir) / f"doc{i}.txt"
            txt_path.write_text(f"Plain text document number {i}.", encoding="utf-8")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-8", source_uri=tmpdir)
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 3:
            record_result(test_id, "FAIL", f"Expected 3 records, got {manifest.record_count}")
            return
        texts = sorted([r.text for r in records])
        expected = sorted([f"Plain text document number {i}." for i in range(3)])
        if texts != expected:
            record_result(test_id, "FAIL", f"Text mismatch. Got: {texts}")
            return
        record_result(test_id, "PASS", "3 plain text files ingested correctly")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.9  Ingest with dedup
# ─────────────────────────────────────────────────────────────
def test_1_1_9() -> None:
    test_id = "1.1.9"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t119_")
        jsonl_path = Path(tmpdir) / "duped.jsonl"
        with open(jsonl_path, "w") as f:
            # Write 5 unique + 5 duplicates = 10 lines
            for i in range(5):
                f.write(json.dumps({"text": f"Unique text {i}"}) + "\n")
            for i in range(5):
                f.write(json.dumps({"text": f"Unique text {i}"}) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-9", source_uri=str(jsonl_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count > 5:
            record_result(test_id, "FAIL", f"Dedup failed: expected <=5, got {manifest.record_count}")
            return
        if manifest.record_count < 5:
            record_result(test_id, "FAIL", f"Too few records: expected 5, got {manifest.record_count}")
            return
        record_result(test_id, "PASS", f"Dedup reduced 10 lines to {manifest.record_count} unique records")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.1.10  Ingest unicode (CJK, emoji, accented)
# ─────────────────────────────────────────────────────────────
def test_1_1_10() -> None:
    test_id = "1.1.10"
    try:
        tmpdir = tempfile.mkdtemp(prefix="t1110_")
        jsonl_path = Path(tmpdir) / "unicode.jsonl"
        unicode_texts = [
            "\u4f60\u597d\u4e16\u754c",                    # Chinese: Hello World
            "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",  # Japanese
            "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4",  # Korean
            "\U0001f600\U0001f680\U0001f4ab Emoji text \U0001f30d",  # Emoji
            "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fc\u00f6\u00e4",  # Accented Latin
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for text in unicode_texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-1-1-10", source_uri=str(jsonl_path))
        name = client.ingest(opts)
        manifest, records = client.dataset(name).load_records()
        if manifest.record_count != 5:
            record_result(test_id, "FAIL", f"Expected 5 records, got {manifest.record_count}")
            return
        loaded_texts = sorted([r.text for r in records])
        expected = sorted(unicode_texts)
        if loaded_texts != expected:
            record_result(test_id, "FAIL",
                          f"Unicode round-trip failed.\nExpected: {expected}\nGot: {loaded_texts}")
            return
        record_result(test_id, "PASS", "5 unicode records (CJK, emoji, accented) round-tripped correctly")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.2  Dataset Push (signature/import check only)
# ─────────────────────────────────────────────────────────────
def test_1_2() -> None:
    test_id = "1.2"
    try:
        from serve.remote_dataset_ops import push_dataset
        import inspect
        sig = inspect.signature(push_dataset)
        params = list(sig.parameters.keys())
        if not params:
            record_result(test_id, "FAIL", "push_dataset has no parameters")
            return
        record_result(test_id, "PASS",
                      f"push_dataset importable, params: {params}")
    except ImportError as e:
        record_result(test_id, "FAIL", f"Cannot import push_dataset: {e}")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.3.1  Curation: Score a dataset
# ─────────────────────────────────────────────────────────────
def test_1_3_1() -> None:
    test_id = "1.3.1"
    try:
        # First ingest a small dataset to score
        tmpdir = tempfile.mkdtemp(prefix="t131_")
        jsonl_path = Path(tmpdir) / "sft.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(20):
                f.write(json.dumps({
                    "text": f"This is a sufficiently long training example number {i} "
                            f"with enough content to be meaningful and produce a decent quality score. "
                            f"Adding more text to make it substantial."
                }) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-sft-mini", source_uri=str(jsonl_path))
        name = client.ingest(opts)

        from serve.dataset_curator import score_examples
        _, records = client.dataset(name).load_records()
        record_dicts = [{"id": r.record_id, "text": r.text} for r in records]
        scores = score_examples(record_dicts)
        if len(scores) != 20:
            record_result(test_id, "FAIL", f"Expected 20 scores, got {len(scores)}")
            return
        # Check scores are valid floats in [0, 1]
        for s in scores:
            if not (0.0 <= s.score <= 1.0):
                record_result(test_id, "FAIL", f"Score out of range: {s.score}")
                return
            if not s.record_id:
                record_result(test_id, "FAIL", f"Empty record_id in score")
                return
        avg_score = sum(s.score for s in scores) / len(scores)
        record_result(test_id, "PASS",
                      f"20 records scored, avg={avg_score:.2f}, all in [0,1]")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# 1.3.2  Curation: Stats on a dataset
# ─────────────────────────────────────────────────────────────
def test_1_3_2() -> None:
    test_id = "1.3.2"
    try:
        # Reuse the dataset from 1.3.1 or create a new one
        tmpdir = tempfile.mkdtemp(prefix="t132_")
        jsonl_path = Path(tmpdir) / "stats.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(15):
                f.write(json.dumps({
                    "text": f"Stats example {i}: " + " ".join(["word"] * (10 + i * 5))
                }) + "\n")
        client = make_client()
        opts = IngestOptions(dataset_name="test-stats", source_uri=str(jsonl_path))
        name = client.ingest(opts)

        from serve.dataset_curator import compute_distributions
        _, records = client.dataset(name).load_records()
        record_dicts = [{"text": r.text} for r in records]
        dist = compute_distributions(record_dicts)
        if dist.total_records != 15:
            record_result(test_id, "FAIL", f"Expected 15 total_records, got {dist.total_records}")
            return
        if dist.avg_token_length <= 0:
            record_result(test_id, "FAIL", f"avg_token_length should be positive: {dist.avg_token_length}")
            return
        if dist.min_token_length <= 0:
            record_result(test_id, "FAIL", f"min_token_length should be positive: {dist.min_token_length}")
            return
        if dist.max_token_length < dist.min_token_length:
            record_result(test_id, "FAIL",
                          f"max < min: {dist.max_token_length} < {dist.min_token_length}")
            return
        if not dist.token_length_histogram:
            record_result(test_id, "FAIL", "token_length_histogram is empty")
            return
        if not dist.quality_distribution:
            record_result(test_id, "FAIL", "quality_distribution is empty")
            return
        record_result(test_id, "PASS",
                      f"Stats: total={dist.total_records}, avg_tokens={dist.avg_token_length}, "
                      f"min={dist.min_token_length}, max={dist.max_token_length}, "
                      f"histogram_buckets={len(dist.token_length_histogram)}")
    except Exception as e:
        record_result(test_id, "FAIL", f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────
# Run all tests and print summary
# ─────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("CRUCIBLE PHASE 1: Dataset Operations Test Suite")
    print(f"Temp data root: {_TMP_ROOT}")
    print("=" * 70)
    print()

    tests = [
        ("1.1.1 Ingest JSONL text field", test_1_1_1),
        ("1.1.2 Ingest JSONL prompt/response", test_1_1_2),
        ("1.1.3 Ingest JSONL custom text_field", test_1_1_3),
        ("1.1.4 Ingest Parquet text column", test_1_1_4),
        ("1.1.5 Ingest Parquet all-null text", test_1_1_5),
        ("1.1.6 Ingest Parquet custom text_field", test_1_1_6),
        ("1.1.7 Ingest CSV", test_1_1_7),
        ("1.1.8 Ingest plain text files", test_1_1_8),
        ("1.1.9 Ingest with dedup", test_1_1_9),
        ("1.1.10 Ingest unicode", test_1_1_10),
        ("1.2 Push dataset (import check)", test_1_2),
        ("1.3.1 Curation: score", test_1_3_1),
        ("1.3.2 Curation: stats", test_1_3_2),
    ]

    for label, fn in tests:
        print(f"\n--- {label} ---")
        fn()

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'TEST_ID':<10} | {'STATUS':<6} | DETAILS")
    print("-" * 70)
    passed = 0
    failed = 0
    for r in RESULTS:
        # Truncate details for table display
        detail_short = r.details[:80] if len(r.details) <= 80 else r.details[:77] + "..."
        print(f"{r.test_id:<10} | {r.status:<6} | {detail_short}")
        if r.status == "PASS":
            passed += 1
        else:
            failed += 1
    print("-" * 70)
    print(f"TOTAL: {passed + failed} tests | {passed} PASSED | {failed} FAILED")
    print("=" * 70)

    # Write results to file
    results_path = Path(__file__).resolve().parent / "test_results_phase1.txt"
    with open(results_path, "w") as f:
        f.write("CRUCIBLE PHASE 1: Dataset Operations Test Results\n")
        f.write(f"Temp data root: {_TMP_ROOT}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'TEST_ID':<10} | {'STATUS':<6} | DETAILS\n")
        f.write("-" * 70 + "\n")
        for r in RESULTS:
            f.write(f"{r.test_id:<10} | {r.status:<6} | {r.details}\n")
        f.write("-" * 70 + "\n")
        f.write(f"TOTAL: {passed + failed} tests | {passed} PASSED | {failed} FAILED\n")
        f.write("=" * 70 + "\n")

        # Write detailed failure info
        failures = [r for r in RESULTS if r.status == "FAIL"]
        if failures:
            f.write("\n\nDETAILED FAILURES:\n")
            f.write("=" * 70 + "\n")
            for r in failures:
                f.write(f"\n{r.test_id}:\n{r.details}\n")
    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
