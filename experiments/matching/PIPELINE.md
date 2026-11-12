# Address matching pipeline

Current unified scoring run:
[525733446208790](https://dbc-031efa75-388b.cloud.databricks.com/jobs/632401166051077/runs/525733446208790?o=7474651042679944).
The saved candidate generators and embeddings are reused. Counts below are pairs
unless explicitly labeled records. Exact-baseline pairs are included in blocking
and the final union; they are not added again as a separate candidate source.

```mermaid
flowchart TD
    A["Silver cleaned records + accepted translations"] --> B["Validated, versioned snapshot<br/>33,677 records; original text and hashes retained"]

    B --> X["Deterministic exact rule<br/>Company + full street address + city + region + country<br/>5,544 baseline pairs tagged; all records retained"]
    X --> C["Original five blocking passes<br/>Exact baseline; country/region/city/building;<br/>company prefix; company rescue; city/street prefix<br/>37,139 unique pairs, including the baseline"]
    B --> E["Two embedding views per record<br/>Company + address; address only<br/>Multilingual MiniLM: 384-dimensional, unit-length vectors"]
    E --> I["Two FAISS HNSW indexes from the saved embedding job<br/>One index per embedding view"]
    I --> N["Nearest-neighbor retrieval<br/>Up to 20 new neighbors per view<br/>Cosine similarity at least 0.70"]
    C -.->|Exclude already-scored blocked pairs| N
    N --> D["Combine views; deduplicate unordered pairs<br/>984,646 HNSW-only candidates"]
    C --> U["Saved candidate union<br/>1,021,785 unique pairs"]
    D --> U

    B --> R["Cleaned per-record comparison views<br/>Normalized company and city; street components v3<br/>House, unit, building, and floor kept separately<br/>Road numbers preserved; ambiguity retained"]
    R --> J["Pairwise string comparisons and levels<br/>Company/street: exact, JW 0.95 / 0.85 / 0.70, disagreement<br/>City: exact, JW 0.95 / 0.85, disagreement<br/>Missing values contribute no evidence"]
    U --> J
    J -.->|Diagnostics only| L["Levenshtein retained<br/>Provisional field-label comparisons<br/>No separate Fellegi-Sunter weight"]

    R --> T["Fit Splink on the full record population<br/>u: seeded random sample targeting 1 million pairs<br/>m: EM on country/company and country/clean street<br/>Company-name frequency adjustment; assumed prior<br/>EM converged in 15 and 9 iterations"]
    J --> S["Unified Fellegi-Sunter inference<br/>Register the exact saved pair list with Splink<br/>Score all 1,021,785 pairs, including HNSW and exact pairs<br/>New fs_* probabilities and evidence weights"]
    T --> S

    R --> G["Independent safeguards<br/>Existing house, suite, numeric, region, country checks<br/>Parsed house/unit/building/floor conflicts<br/>Partial, missing, or ambiguous components require review"]
    S --> Q["Decision rules<br/>Keep exact-baseline tag first; then apply safeguards<br/>Proposal requires probability at least 0.99<br/>and all three JW similarities at least 0.85"]
    G --> Q
    X -.->|Exact-rule flag| Q

    Q --> O1["Exact baseline<br/>5,544 pairs"]
    Q --> O2["Proposed fuzzy matches<br/>6,193 pairs<br/>6,192 blocked + 1 HNSW-only"]
    Q --> O3["Review: conflict<br/>1,005,563 pairs"]
    Q --> O4["Review: uncertain<br/>3,141 pairs"]
    Q --> O5["Likely nonmatches<br/>1,344 pairs"]

    O1 --> V["Versioned Delta outputs, review samples, and MLflow artifacts<br/>Historical evidence preserved; probabilities uncalibrated<br/>No automatic merging"]
    O2 --> V
    O3 --> V
    O4 --> V
    O5 --> V
```

The full model, guard rules, before/after counts, validation, and query are in
[UNIFIED_SCORING.md](UNIFIED_SCORING.md). The 384-dimensional embedding vectors
remain intact; cosine is a retrieval similarity, and the Fellegi–Sunter score
comes from the learned string-comparison evidence. The 150 model-reviewed field
judgments are diagnostic only and were not used as entity-match training labels.
