# FTEC5660 Homework 4 — Retrieval for a RAG assistant over company reports

This repository contains my solution to Homework 4. The five graded functions are in
[`hw4.py`](hw4.py); everything below the `Provided code below` line is untouched. The public-test
run produces [`results.csv`](results.csv), and the ablation study is in
[`experiments/ablation.py`](experiments/ablation.py) with its per-variant CSVs.

## Homework 4 solution

### Public-test result

| Task | Score | Measured on `public_test` |
| --- | ---: | --- |
| 1.1 Tables to text | 8 / 8 | 12/12 tables at 1.000 cell score |
| 1.2 Chunking with metadata | 7 / 7 | 40/40 documents pass |
| 2.1 Valid retrieval | 5 / 5 | 100% valid, non-empty on answerable questions |
| 2.2 Query parsing | 10 / 10 | 100% of the three fields correct |
| 2.3 Retrieval recall@5 | 15 / 15 | 46/46 answerable questions hit |
| 3 Abstention | 10 / 10 | kept 100%, rejected 100% |
| 4 End-to-end accuracy | 20 / 20 | 60/60 answers correct |
| **Tasks 1–4** | **75 / 75** | 875 chunks; full run about 6 s (index under 1 s) |

The run used the provided `deepseek-v4-flash` generator (46 calls, about 55.8k input tokens,
USD 0.009). Table parsing, chunking, query parsing, indexing and retrieval make **no** LLM calls;
the only stochastic component is the fixed generator.

### Pipeline

```mermaid
flowchart TD
    H[HTML document] --> T[table_to_text<br/>caption + one line per table row]
    H --> W[body walk<br/>headings, paragraphs, lists, tables]
    T --> W
    W --> C[chunk_document<br/>section-aware chunks <= 1500 chars + metadata]
    C --> I[build_index<br/>BM25 + company / doc-type maps + document frequencies]
    Q[question] --> P[parse_query<br/>companies, fiscal_year, doc_type]
    P --> F[filter<br/>company + doc type + fiscal-year coverage]
    I --> F
    F --> S[score<br/>BM25 + year, metric-row and section boosts]
    S --> A[select <= 5 chunks<br/>2 per company for comparisons]
    A --> X{answerable?}
    X -->|yes| R[return chunks]
    X -->|no| N[return empty list]
    R --> G[fixed deepseek-v4-flash generator]
    G --> O[answer or NOT_FOUND]
```

**Parsing.** `table_to_text` expands `colspan`/`rowspan` into a rectangular grid, takes the
`<caption>` (which carries the unit: “USD millions”, “number of people”), and emits one line per
data row in the form `label | header: value | header: value`. Row labels come from the first cell
(a `<th scope="row">` in the balance sheet), column headers from the `<thead>` row (“2025”,
“31 December 2024”, “FY2026E”, “Role”), and section rows that span the whole table are skipped so
they do not pollute the column headers. Lines are capped at 380 characters; a very wide row is
split into continuation lines that repeat the row label, so every value is still on a line with
its own label and header. Negative values keep their parentheses and commas are untouched
(`(1,762.4)`), which matters for both the cell scorer and the generator.

**Metadata and chunking.** `chunk_document` reads the company from `<h1>` (dropping the ticker
parenthesis and a legal suffix), the fiscal year from the title/`Annual Report 2025` /
`Full Year 2025` / `FY2025 results review`, and the type from the same first lines
(`annual_report`, `earnings_call`, `research_report`). It then walks the body in document order and
keeps headings, paragraphs, list items and tables as units. Narrative units are packed to about
1,150 characters per chunk; each table is kept as its own chunk, or split into several chunks with
the caption repeated so a continuation still knows its unit. Every chunk starts with
`Company | Annual Report 2025 | 6. Financial statements > Consolidated balance sheet`, i.e. a
short contextual header made only from that document's metadata and section title. The longest
chunk is 1,497 characters; all 40 public documents keep 100% of their probe values.

**Index.** `build_index` builds a `BM25Okapi` index (`k1=1.5`, `b=0.75`) over lower-cased
alphanumeric tokens, with question scaffolding stop words removed and numbers kept as numbers.
It also stores the chunks grouped by normalised company name and document type, plus document
frequencies for the abstention check. Company names are normalised by `company_key`, so
`Northgate Robotics`, `Northgate Robotics Inc.` and `NORTHGATE ROBOTICS` are the same key.

**Retrieval.** `parse_query` first finds known corpus companies by full-name matching (longest
match wins), masks brokers and auditors so they are never returned as companies, and then extracts
remaining capitalised company-like runs as out-of-corpus companies (for example
`Halcyon Air Cargo`); segment and region names learned from the corpus are rejected as companies.
The year is the `FY...` year, otherwise the single four-digit year in the question (so
“31 December 2024” gives 2024), otherwise `None`. The type is `earnings_call` for call/guidance
phrasing, `research_report` for rating/target-price/forecast/results-review phrasing, and
`annual_report` otherwise.

`retrieve` filters candidates by company and document type, then by fiscal-year coverage. For an
annual-report question it prefers the report of the named year (`FY2025` → the FY2025 report);
if that report does not exist it uses the next year's report, which holds the prior-year column
(FY2023 → FY2024 report). Research reports cover FY2024A–FY2027E in a FY2025 report, and earnings
calls cover FY2025 results and FY2026 guidance. Candidates are scored by BM25 plus a year match,
the presence of the question's metric as a row label (`Revenue | ...` is worth more than a passing
mention), a section hint (`cash` → balance sheet, `employees` → employees table, `auditor` →
independent auditor, `headquarters` → cover, …), and role cues for CFO/CEO/rating/target-price
questions. For comparisons, each company gets at least two slots before the remaining slots are
filled from the global ranking; five chunks of one company are a miss even if they are all
relevant.

**Abstention.** `retrieve` returns `[]` when no parsed company exists in the corpus, when the
named year is outside the corpus coverage (FY2021/FY2022 annual facts, FY2026 annual facts), or
when a content word the question needs is absent from the corpus and from the retrieved chunks
(`lobbying`, `carbon`, `churn`, `single`). That last check uses normalised word variants and
synonyms, so legitimate paraphrases such as “staff”/“headcount” or “borrowing”/“borrowings” do not
trigger a rejection. The generator itself is never asked an answerability question: `[]` is the
only signal it needs, and it is cheap, deterministic and easy to audit.

**Why this design.** The corpus is small and the questions are about named companies, named years
and a small set of financial metrics, so lexical retrieval plus strong metadata is enough. It also
avoids the two failure modes the handout highlights: similar company names (Northgate Robotics vs
Northgate Payments) and two-year columns (FY2025 reports contain FY2024). I deliberately did not
use an LLM in `parse_query`, `build_index` or `retrieve`: rules are inspectable, reproducible, and
fast, and the 30-minute budget is never at risk.

### Ablation study

`experiments/ablation.py` starts from the naive baseline required by the handout and adds one
component per row. `B0` uses `BeautifulSoup.get_text()`, `RecursiveCharacterTextSplitter`
(1,000 characters, 150 overlap), the four required metadata fields, lower-cased whitespace-token
BM25 over the whole corpus, top-5, no query parsing and never `[]`. Every later row changes only
the component named in the table; `V6` is exactly the submitted `hw4.py`. All rows use the same
public corpus, the same 60 questions, the same fixed `deepseek-v4-flash` generator (temperature 0)
and the same scoring functions.

| Variant | One change relative to the row above | Recall@5 | kept | rejected | Answer accuracy | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | naive RAG baseline (`get_text`, recursive splitter, whitespace BM25, no parsing, never `[]`) | 28.3% | 100% | 0% | 43.3% | 7.2 s |
| V1 | `table_to_text` + section/table-aware chunking | 78.3% | 100% | 0% | 76.7% | 6.6 s |
| V2 | metadata filtering from `parse_query` (company + doc type) | 89.1% | 100% | 0% | 86.7% | 6.2 s |
| V3 | fiscal-year-aware document selection | 93.5% | 100% | 0% | 91.7% | 6.1 s |
| V4 | per-company quota for comparison questions | 93.5% | 100% | 0% | 91.7% | 6.2 s |
| V5 | richer tokeniser, synonyms, section/metric ranking boosts | 100% | 100% | 0% | 96.7% | 7.0 s |
| V6 | abstention on unknown company, uncovered year, undisclosed metric | 100% | 100% | 100% | 100% | 5.3 s |

`kept` is the share of answerable questions with a non-empty result and `rejected` the share of
unanswerable questions with `[]`; B0–V5 never abstain, so their `rejected` is 0. Accuracy is over
all 60 questions and therefore includes the 14 unanswerable ones; this is why accuracy is higher
than recall in B0 (the generator often says `NOT_FOUND` even when it is handed irrelevant text).
Each row has its own CSV in `experiments/results_<variant>.csv`; the summary is
`experiments/ablation_summary.csv`.

**What each change did.**

* **V1 (table text + chunking): +48.4 recall, +33.4 accuracy.** This is the largest single jump.
  `get_text()` turns a balance sheet into a stream of numbers with no row or column, and the
  recursive splitter cuts through tables and mixes sections. Keeping one line per row with its
  label and year header, and adding a company/year/section header to every chunk, lets BM25 find
  a number that is actually attached to the metric the question asks for.
* **V2 (company + type filter): +10.8 recall, +10.0 accuracy.** Without a company filter, a
  question about one company retrieves another company's nearly identical governance or income
  statement section. In B0, `q010` (“Who was Brightwater Software's CEO?”) returned Corvane,
  Halcyon, Kestrel, Brightwater Energy and Northgate — every company except the right one.
* **V3 (year coverage): +4.3 recall, +5.0 accuracy.** Every annual report contains two year
  columns, so the same metric and company appears in two reports. Preferring the report of the
  named year (and using FY2024 for FY2023 facts) removes the column ambiguity; the effect is
  smaller than the company filter because both copies of a value are identical when they exist.
* **V4 (per-company quota): no measurable change on the public set.** By V3, both companies in
  every comparison question already appeared in the top five, because the company filter leaves
  only two candidate pools and balance-sheet chunks are distinctive. I kept the quota because it
  costs nothing and protects the private set when one company's BM25 score dominates; it changes
  which chunks fill the five slots, not which candidates exist, so it cannot add recall by itself.
* **V5 (ranking): +6.5 recall, +5.0 accuracy.** After filtering, the remaining errors are
  “right document, wrong chunk”. `q040` asks for Kestrel's inventories; V3 returns five chunks of
  the correct FY2025 report (business overview, executive officers, operating income, cover,
  liquidity) but never the balance-sheet row with `162.6`, because whitespace BM25 dilutes the one
  discriminative token in a long chunk and does not connect `inventories` to the balance sheet.
  The richer tokeniser, synonyms and row-label/section boosts put that row back in the top five.
  The same change fixes `q027` (“In which city is Halcyon Freight headquartered?”), where the
  question words `city` and `headquartered` do not occur in the corpus but `Headquarters: Memphis`
  does.
* **V6 (abstention): rejected 0 → 100%, accuracy +3.3.** V5 answers `q044` (“Orchid Bay Resorts'
  operating income in FY2024”) with `231.3`, the FY2024 operating income of the similarly named
  but different Orchid Bay Hotels. V6 parses `Orchid Bay Resorts` as a company, finds no such key
  in the index, and returns `[]`. The same rule fixes `q001` (`Northgate Robotic Systems`), which
  V5 answered with Northgate Robotics' cash.

### Error analysis

1. **B0, `q010` — company confusion.** Question: “Who was Brightwater Software's Chief Executive
   Officer at the end of FY2025?” Retrieved: `doc_031` (Corvane), `doc_016` (Halcyon), `doc_026`
   (Kestrel), `doc_032` (Brightwater Energy), `doc_038` (Northgate). The first passage is a board
   table listing Corvane's CEO; the word “Chief Executive Officer” is generic and BM25 cannot tell
   which company the question is about. The generator answered `NOT_FOUND`. This is exactly the
   failure the metadata filter (V2) removes: candidates are restricted to `brightwater software`
   and `annual_report`, and the FY2025 report's executive-officers table gives `Yara Castillo`.
2. **V3, `q040` — right document, wrong chunk.** Question: “What was the value of Kestrel
   Semiconductor's inventories at 31 December 2025?” Retrieved five chunks, all from the correct
   `doc_026.html`: business overview, executive officers (the CFO change), MD&A operating income,
   the cover, and liquidity. None contains the balance-sheet row
   `Inventories | 31 December 2025: 162.6 | 31 December 2024: ...`. The row exists in the document;
   it simply ranks sixth because the query's other words (`value`, `2025`, `Kestrel`) are common
   and the long balance-sheet chunk dilutes `inventories`. V5's metric-row boost and
   `inventories → balance sheet` section hint move the row into the top five, and the generator
   returns `162.6`.
3. **V5, `q044` — plausible answer to an unanswerable question.** Question: “What was Orchid Bay
   Resorts' operating income in FY2024?” `Orchid Bay Resorts` is a decoy for `Orchid Bay Hotels`.
   V5 retrieves Orchid Bay Hotels' FY2024 and FY2025 income statements plus two other companies
   and the generator answers `231.3`, which is Orchid Bay Hotels' FY2024 operating income — a
   believable but wrong answer. V6 rejects it before generation: the parsed company key is not in
   the corpus index, so `retrieve` returns `[]` and the runner reports `NOT_FOUND`. This is the
   reason abstention is worth its own component: without it, high recall can make the system
   confidently wrong.

### Reproducing the results

From the repository root on Windows (Python 3.10+):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "your_key_here"    # or put it in .env; .env is git-ignored
.\.venv\Scripts\python.exe hw4.py --data-folder public_test --output results.csv
.\.venv\Scripts\python.exe experiments\ablation.py
```

The first run writes `results.csv` (60 rows, one per question) and prints the public-test
estimates. The ablation script writes `experiments/results_B0.csv` … `results_V6.csv` and
`experiments/ablation_summary.csv`; it needs the API key only for the answer column. Use
`--no-generator` to compute recall/kept/rejected without spending tokens.

Repository layout:

```
hw4.py                         five graded functions + helpers (provided code untouched below the line)
requirements.txt               starter dependencies (bs4, lxml, rank-bm25, LangChain, dotenv)
results.csv                    official public-test run of the final configuration
experiments/ablation.py        B0, V1–V6 ablation driver
experiments/results_*.csv      one result file per variant
experiments/ablation_summary.csv   the table above
public_test/                   40 documents, 60 questions, 12 tables, metadata (course data)
```

## Task 6: Reflection

In a world where powerful personal AI agents handle much of the writing, programming, research and decision making, I hope to become a person who can judge what deserves to be done and who accepts responsibility for the consequences of that judgment. Such judgment comes from a long acquaintance with an industry, its institutions and the people inside them, and from the habit of checking the source of a claim before accepting it. While I completed this assignment, the demanding part was confirming which table, which year and which company produced a particular number, and composing a fluent answer felt comparatively easy. I want my future self to keep that sensitivity to evidence and to pause for the basis of a conclusion when others are eager to accept a plausible one.

My current ambitions focus on understanding and contribution, and both remain important in an age where capable AI agents handle much routine work. I want to understand how markets set prices, how companies generate cash flow and how institutions influence human behavior, because that understanding determines the questions I ask and the answers I am willing to accept. I still want to build things that help real people, to reach a level of craft that others can rely on, and to pass useful experience to those who follow. As execution becomes cheaper, the choice of what to pursue, what to decline and how to own the outcome will matter more, and these ambitions will deserve even more commitment.

Beyond professional achievement, I hope people remember me as fair, dependable and willing to speak plainly even when the plain answer is inconvenient. When information is uncertain, I will state what I know and what remains unclear, and I will spend the time needed to resolve the difference. When a team disagrees, I hope to hear every reason before offering a judgment, and to correct that judgment promptly when the evidence changes. These qualities come from repeated choices in specific situations and from long relationships among people, and the conduct itself remains a human responsibility that an AI assistant can support through analysis.

Even when an AI assistant can deliver a result quickly, I will choose to spend time struggling for understanding, craft and relationships. Deriving a formula myself, reading the notes to an annual report and repairing a model that produces errors will change the way I observe problems and help me notice anomalies sooner the next time. Relationships require the same effort, through listening to a person until the end, appearing when someone is in difficulty and addressing a conflict with honesty, and these experiences develop patience and judgment. An AI assistant can save time, while these experiences still require my direct participation, so I will reserve part of each week for work that produces value over a longer period.

The skills that will remain valuable include formulating a good question, judging the quality of evidence, making trade offs in complex situations, expressing a judgment clearly and maintaining trust with other people. Domain knowledge will still matter, because a person who understands the business knows which number is unusual and which assumption lacks support. Taste will matter as well, because when generation becomes cheap, the choice of what to keep and what to advance becomes a scarce capability. Ethical judgment will matter for the same reason, since an AI assistant pursues the objectives it is given, and the person who sets those objectives needs clarity about what is acceptable.

If output becomes abundant and society rewards production less, I will define myself through the attention I give, the commitments I keep and the character I show in ordinary situations. I will pay attention to the specific people around me, remember what they have said and respond to what they need; I will keep my promises and complete the work I have undertaken even when I am alone; and I will stay curious, looking for a fuller explanation after the first one that comes easily. These qualities are difficult to measure with a metric, and they determine how I treat other people, how I respond when a task becomes inconvenient and whether I deserve their trust over time.

Starting tomorrow, I will begin each morning by writing out the most important decision of the day, the evidence that supports it and the information that could change my mind, and I will do this before opening any AI tool. For the next forty five minutes I will work through the hardest part alone, and I will ask an AI assistant to search, check and extend the work after that. This routine will let me practice examining evidence and owning outcomes every day, and it will let me form my own judgment before I rely on a tool.
