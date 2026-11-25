"""Task 5: ablation study on public_test.

Seven configurations, each adding exactly one component to the row above:

  B0  naive RAG baseline (get_text, RecursiveCharacterTextSplitter, required metadata,
      whitespace-token BM25 top-5, no query parsing, never abstains)
  V1  + table_to_text and section/table-aware chunking
  V2  + parse_query metadata filtering (company and document type)
  V3  + fiscal-year-aware document selection
  V4  + per-company quota for comparison questions
  V5  + richer tokeniser, synonyms and section/metric ranking boosts
  V6  + abstention on unknown companies, uncovered years and undisclosed metrics (= hw4.py)

Usage (from the repo root, with DEEPSEEK_API_KEY set or in .env):

    python experiments/ablation.py
    python experiments/ablation.py --variants B0 V1 V6 --output-dir experiments

Every variant writes experiments/results_<variant>.csv and the summary is written to
experiments/ablation_summary.csv.  Retrieval is deterministic; only the answer column needs
the fixed deepseek-v4-flash generator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hw4  # noqa: E402  (path set above)


def load_data(folder: Path):
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")
    docs_meta = hw4.load_json(folder / "docs.json")
    return corpus, questions, ground_truth, docs_meta


# --------------------------------------------------------------------------- chunking

def b0_chunks(corpus: dict[str, str]) -> list[Document]:
    """Naive baseline: raw page text and a recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks: list[Document] = []
    for file_name, html in corpus.items():
        soup = BeautifulSoup(html, "lxml")
        company, fiscal_year, doc_type = hw4._document_metadata(soup)
        text = soup.get_text(" ", strip=True)
        for piece in splitter.split_text(text):
            chunks.append(Document(page_content=piece, metadata={
                "source": file_name, "company": company,
                "fiscal_year": fiscal_year, "doc_type": doc_type}))
    return chunks


def solution_chunks(corpus: dict[str, str]) -> list[Document]:
    return [chunk for file_name in sorted(corpus)
            for chunk in hw4.chunk_document(file_name, corpus[file_name])]


# --------------------------------------------------------------------------- indexes

def whitespace_index(chunks: list[Document]) -> dict:
    """BM25 over lower-cased whitespace tokens, as in the B0 definition."""
    tokens = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokens, k1=1.5, b=0.75)
    by_company: dict[str, list[int]] = {}
    by_type: dict[str, list[int]] = {}
    companies: dict[str, str] = {}
    doc_freq: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = hw4.company_key(meta.get("company", ""))
        by_company.setdefault(key, []).append(i)
        companies[key] = meta.get("company", "")
        by_type.setdefault(str(meta.get("doc_type", "")), []).append(i)
        for token in set(tokens[i]):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {"chunks": chunks, "bm25": bm25, "by_company": by_company, "by_type": by_type,
            "companies": companies, "doc_freq": doc_freq}


def build_variant_index(variant: str, chunks: list[Document]) -> dict:
    if variant in ("V5", "V6"):
        return hw4.build_index(chunks)
    return whitespace_index(chunks)


# --------------------------------------------------------------------------- retrieval

def _targets(index: dict, parsed: dict) -> list[str]:
    out: list[str] = []
    for name in parsed.get("companies", []):
        key = hw4.company_key(name)
        if key in index["companies"] and key not in out:
            out.append(key)
    return out


def _candidates(index: dict, question: str, use_metadata: bool, use_year: bool):
    parsed = hw4.parse_query(question)
    if not use_metadata:
        return list(range(len(index["chunks"]))), parsed, []
    targets = _targets(index, parsed)
    if not targets:
        return list(range(len(index["chunks"]))), parsed, []
    candidates: list[int] = []
    for key in targets:
        for i in index["by_company"].get(key, []):
            if index["chunks"][i].metadata.get("doc_type") != parsed["doc_type"]:
                continue
            if use_year and parsed["fiscal_year"] is not None:
                years = hw4._year_window(index, targets, parsed["fiscal_year"], parsed["doc_type"])
                if years and index["chunks"][i].metadata.get("fiscal_year") not in years:
                    continue
            candidates.append(i)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:                       # no abstention before V6: fall back, never []
        return list(range(len(index["chunks"]))), parsed, targets
    return candidates, parsed, targets


def _simple_rank(index: dict, question: str, candidates: list[int]) -> list[int]:
    scores = index["bm25"].get_scores(question.lower().split())
    return sorted(candidates, key=lambda i: scores[i], reverse=True)


def _quota(index: dict, ordered: list[int], targets: list[str]) -> list[int]:
    if len(targets) <= 1:
        return ordered[: hw4.MAX_RETRIEVED]
    chosen: list[int] = []
    for key in targets:
        own = [i for i in ordered
               if hw4.company_key(index["chunks"][i].metadata.get("company", "")) == key]
        chosen.extend(own[:2])
    for i in ordered:
        if len(chosen) >= hw4.MAX_RETRIEVED:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[: hw4.MAX_RETRIEVED]


def retrieve_variant(variant: str, index: dict, question: str) -> list[Document]:
    if variant == "V6":
        return hw4.retrieve(index, question)
    if variant == "B0":
        ordered = _simple_rank(index, question, list(range(len(index["chunks"]))))
        return [index["chunks"][i] for i in ordered[: hw4.MAX_RETRIEVED]]
    candidates, parsed, targets = _candidates(
        index, question,
        use_metadata=variant in ("V2", "V3", "V4", "V5"),
        use_year=variant in ("V3", "V4", "V5"))
    if variant == "V5":
        scores = hw4._score_chunks(index, question, parsed, candidates, targets,
                                   parsed["fiscal_year"], parsed["doc_type"])
        ordered = sorted(candidates, key=lambda i: scores[i], reverse=True)
    else:
        ordered = _simple_rank(index, question, candidates)
    if variant in ("V4", "V5"):
        ordered = _quota(index, ordered, targets)
    return [index["chunks"][i] for i in ordered[: hw4.MAX_RETRIEVED]]


# --------------------------------------------------------------------------- scoring

def score_variant(variant: str, index: dict, questions: dict, ground_truth: dict,
                  generate: bool) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    retrieved: dict[str, list[Document]] = {}
    for qid, question in questions.items():
        try:
            retrieved[qid] = list(retrieve_variant(variant, index, question) or [])[: hw4.MAX_RETRIEVED]
        except Exception as exc:  # keep the table complete even if one question fails
            print(f"  {variant} retrieve failed on {qid}: {exc!r}")
            retrieved[qid] = []
    answers = {qid: "" for qid in questions}
    if generate:
        def answer(qid: str) -> str:
            try:
                return hw4.generate(questions[qid], retrieved[qid])
            except Exception as exc:
                print(f"  {variant} generator failed on {qid}: {exc!r}")
                return ""
        with ThreadPoolExecutor(max_workers=8) as pool:
            for qid, text in zip(questions, pool.map(answer, questions)):
                answers[qid] = text
    for qid, question in questions.items():
        gold = ground_truth[qid]
        hit = hw4.retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
        rows.append({"id": qid, "category": gold["category"], "question": question,
                     "retrieved": ";".join(sorted({c.metadata.get("source", "") for c in retrieved[qid]})),
                     "retrieval_hit": hit,
                     "answer": answers[qid],
                     "correct": hw4.answer_correct(answers[qid], gold) if generate else ""})
    answerable = [q for q in questions if ground_truth[q]["evidence"]]
    unanswerable = [q for q in questions if not ground_truth[q]["evidence"]]
    hits = sum(hw4.retrieval_hit(retrieved[q], ground_truth[q]) for q in answerable)
    summary = {
        "variant": variant,
        "recall_at_5": round(hits / len(answerable), 4),
        "kept": round(sum(bool(retrieved[q]) for q in answerable) / len(answerable), 4),
        "rejected": round(sum(not retrieved[q] for q in unanswerable) / len(unanswerable), 4),
        "accuracy": round(sum(rows[i]["correct"] is True for i in range(len(rows))) / len(rows), 4)
        if generate else "",
    }
    return rows, summary


VARIANTS = {
    "B0": "naive baseline: get_text + RecursiveCharacterTextSplitter + whitespace BM25 top-5, no parsing, never []",
    "V1": "table_to_text and section/table-aware chunking (chunk_document)",
    "V2": "metadata filtering from parse_query (company + doc type)",
    "V3": "fiscal-year-aware document selection",
    "V4": "per-company quota for comparison questions",
    "V5": "richer tokeniser, synonyms and section/metric boosts",
    "V6": "abstention on unknown companies, uncovered years, undisclosed metrics (submitted hw4.py)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework 4 ablation study")
    parser.add_argument("--data-folder", default=str(ROOT / "public_test"))
    parser.add_argument("--output-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--no-generator", action="store_true",
                        help="skip deepseek calls; the accuracy column is left empty")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, questions, ground_truth, _ = load_data(folder)
    if not args.no_generator and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY (or put it in .env) or pass --no-generator.")

    naive = b0_chunks(corpus)
    improved = solution_chunks(corpus)
    print(f"B0 chunks: {len(naive)} | solution chunks: {len(improved)}")

    summaries = []
    for variant in args.variants:
        start = time.perf_counter()
        chunks = naive if variant == "B0" else improved
        index = build_variant_index(variant, chunks)
        rows, summary = score_variant(variant, index, questions, ground_truth,
                                      generate=not args.no_generator)
        summary["seconds"] = round(time.perf_counter() - start, 1)
        summaries.append(summary)
        with open(out_dir / f"results_{variant}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        accuracy = "-" if summary["accuracy"] == "" else f"{summary['accuracy']:.1%}"
        print(f"{variant}: recall {summary['recall_at_5']:.1%} | kept {summary['kept']:.1%} | "
              f"rejected {summary['rejected']:.1%} | accuracy {accuracy} | {summary['seconds']}s")

    with open(out_dir / "ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "recall_at_5", "kept", "rejected",
                                               "accuracy", "seconds"])
        writer.writeheader()
        writer.writerows(summaries)
    with open(out_dir / "ablation_changes.json", "w", encoding="utf-8") as f:
        json.dump({variant: VARIANTS[variant] for variant in args.variants}, f, indent=2)
    print(f"\nWrote {out_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
