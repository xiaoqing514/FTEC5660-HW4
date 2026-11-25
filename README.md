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

十年之后，强大的个人 AI 助手会承担大量写作、编程、研究和决策工作，我希望自己成为能够判断什么值得做、并且愿意为判断承担后果的人。这样的判断来自对行业、机构和人性的长期理解，也来自在具体问题里反复核对证据的习惯。完成这次作业时，真正耗费心力的部分在于确认某一条数字来自哪张表、哪一年、哪家公司，生成一句流畅的答案相对容易。我希望十年后的自己仍然保有这种对来源和证据的敏感，并且在别人急于接受一个看起来合理的结论时，能够停下来问清楚依据。

我现在的抱负大多围绕理解力和贡献展开，这些抱负在 AI 时代仍然重要。我希望真正理解市场如何定价、企业如何创造现金流、制度如何影响人的行为，因为这种理解决定了我会提出什么问题，也决定了我会不会直接接受一个现成的答案。我仍然希望做出对真实的人有用的东西，希望在一门手艺上达到可以信赖的水平，也希望把经验传给后来的人。执行成本下降之后，选择做什么、拒绝做什么以及如何承担结果，会比以往更加重要，这些抱负也因此更值得坚持。

除了职业上的成绩，我希望别人记住我是一个公正、可靠、愿意说真话的人。面对不确定的信息，我会说明自己掌握的事实和仍然存在的疑问，并且愿意花时间把不清楚的地方弄清楚。团队里出现分歧时，我希望自己能够先听完各方理由，再给出有依据的判断，并且在判断出错时及时承认和修正。这样的品质无法由 AI 助手代替，因为它们发生在人与人的长期相处之中，也需要一个人在具体情境里反复选择。

即使 AI 助手可以迅速给出结果，我仍然愿意为理解、手艺和关系付出缓慢而具体的努力。亲自推导一遍公式、读完一份年报的附注、把一个模型从错误改到可靠，这些过程会改变我观察世界的方式，也会让我在下次遇到相似问题时更快发现异常。与人的关系同样需要这种努力，认真听一个人说完、在对方困难时到场、在冲突之后修复信任，这些经历会塑造一个人的耐心和分寸。AI 可以节省时间，这些过程仍然需要我亲自经历，因此我愿意把一部分时间留给这些暂时低效的事情。

在未来仍然有价值的技能包括提出好问题、分辨证据质量、在复杂情境中做取舍、把判断清楚地表达出来，以及与他人建立信任。领域知识依然重要，因为理解业务的人才知道哪个数字异常、哪个假设缺少支持。审美和品味也会变得重要，因为当生成变得廉价，选择哪些内容值得保留、哪些方案值得推进，会成为稀缺能力。伦理判断同样重要，因为 AI 助手会按照被设定的目标行动，设定目标的人需要清楚什么可以接受、什么必须拒绝。

如果社会不再主要用产出衡量一个人的价值，我会用注意力、承诺和品格来定义自己。我愿意把注意力放在具体的人身上，记住他们说过的话，回应他们的需要；我愿意遵守承诺，在没有人监督的时候也把该做的事情做好；我愿意保持好奇，在现有解释之外继续寻找更完整的解释。这些内容很难被量化，同时决定了我如何与他人相处，也决定了我是否值得被信任。

从明天开始，我每天早晨会先手写当天最重要的一个决定、支持这个决定的证据以及可能推翻它的信息，然后再打开任何 AI 工具。接下来的四十五分钟里，我会独立完成最困难的那部分工作，之后才让 AI 助手帮助检索、检查和补充。这个习惯可以让我在日常工作里持续练习核对证据和承担结果，也可以让我在依赖工具之前先形成自己的判断。
