# Findings Summary — HealthifyMe vs MyFitnessPal
## Validated numbers only. No recommendations. Last updated: 2026-09-16.

---

## 0. Dataset Overview

| Metric | HealthifyMe | MyFitnessPal |
|---|---|---|
| Rows after cleaning | 63,147 | 653,650 |
| Topic-scope rows (≥20 char, MFP stratified) | 34,765 | 34,765 |
| MFP full topic-scope (before sampling) | — | 532,171 |

---

## 1. Rating Distribution (full cleaned datasets)

| Stars | HealthifyMe | MyFitnessPal |
|---|---|---|
| 1★ | 13.7% | 8.0% |
| 2★ | 2.6% | 3.0% |
| 3★ | 4.7% | 4.4% |
| 4★ | 13.8% | 16.8% |
| 5★ | 65.3% | 67.8% |

---

## 2. Developer Response Rate (full cleaned datasets)

| Metric | HealthifyMe | MyFitnessPal |
|---|---|---|
| Overall dev response rate | 58.1% | 10.7% |
| Dev response rate on 1★ reviews | 93.6% | 62.4% |

---

## 3. Topic Structure (BERTopic on topic-scope datasets)

| Metric | HealthifyMe | MyFitnessPal |
|---|---|---|
| Total documents | 34,765 | 34,765 |
| Real topics found | 59 | 28 |
| Docs assigned to real topics | 24,876 (71.6%) | 21,974 (63.2%) |
| Outlier (-1) docs |  9,889 (28.4%) | 12,791 (36.8%) |
| Silhouette score (real topics only, 5K sample, seed=42) | −0.2789 | −0.4686 |
| Silhouette score (outliers included as own cluster) | −0.3962 | −0.5353 |
| LDA coherence C_v (original course project baseline) | 0.51 | 0.51 |

BERTopic parameters (identical for both apps): UMAP n_neighbors=15, n_components=5,
min_dist=0.0, metric=cosine, random_state=42; HDBSCAN min_cluster_size=40,
min_samples=10, metric=euclidean, cluster_selection_method=eom;
BERTopic nr_topics=auto, top_n_words=10; embeddings: all-MiniLM-L6-v2 (384-dim).

### 3a. HealthifyMe — Top 5 Most Negative Topics (RoBERTa % negative)

| Topic | % Negative | Top words |
|---|---|---|
| T22 | 97.9% | worst, ever, fake, app, dont, bad, seen, download |
| T34 | 90.8% | call, coach, coaches, you, will, no, money |
| T55 | 90.7% | waste, money, time, wasting, space, absolute, no |
| T38 | 90.1% | customer, support, service, worst, care, no, team, response |
| T8  | 88.6% | refund, plan, cancel, smart, subscription, my, money |

### 3b. HealthifyMe — Top 5 Most Positive Topics (lowest RoBERTa % negative)

| Topic | % Negative | Top words |
|---|---|---|
| T54 | 0.0% | health, track, tracking, application, monitoring, good, way |
| T56 | 0.0% | using, started, far, used, good, been, its |
| T46 | 0.0% | diet, helpful, helps, balanced, maintain, control, perfect |
| T4  | 1.1% | health, healthy, good, lifestyle, life, its, helpful |
| T19 | 1.3% | experience, good, day, first, till, very, nice, far |

### 3c. MyFitnessPal — Top 5 Most Negative Topics (RoBERTa % negative)

| Topic | % Negative | Top words |
|---|---|---|
| T14 | 94.7% | server, sync, connect, unable, cant, to, the, wont |
| T10 | 89.7% | data, personal, dark, information, email, web, permissions |
| T5  | 89.3% | cancel, subscription, trial, charged, refund, premium, not |
| T1  | 89.0% | sign, email, account, cant, log, let, in, up |
| T23 | 88.0% | search, network, error, food, fix, now, it, fine |

### 3d. MyFitnessPal — Top 5 Most Positive Topics (lowest RoBERTa % negative)

| Topic | % Negative | Top words |
|---|---|---|
| T15 | 0.0% | use, easy, excellent, awesome, and, very, good |
| T25 | 0.0% | tool, great, valuable, its, helpful, use, very, easy |
| T4  | 0.6% | helpful, very, helped, me, useful, really, lot |
| T8  | 1.6% | works, well, it, does, for, working, work, job |
| T9  | 1.7% | it, everyday, use, using, years, been, love, for |

---

## 4. Sentiment Model Comparison (topic-scope datasets, 34,765 rows each)

### 4a. VADER label distribution

| Label | HealthifyMe | MyFitnessPal |
|---|---|---|
| Positive | 74.3% | 80.2% |
| Neutral | 10.3% | 8.4% |
| Negative | 15.4% | 11.4% |

### 4b. RoBERTa label distribution

| Label | HealthifyMe | MyFitnessPal |
|---|---|---|
| Positive | 64.1% | 73.1% |
| Neutral | 12.0% | 9.2% |
| Negative | 23.9% | 17.8% |

### 4c. Agreement and disagreement

| Metric | HealthifyMe | MyFitnessPal |
|---|---|---|
| Overall agreement (same label) | 78.9% | 78.1% |
| Cat 1: VADER pos/neu → RoBERTa neg (count) | 3,738 | 3,605 |
| Cat 1: VADER pos/neu → RoBERTa neg (% of scope) | 10.75% | 10.37% |
| Cat 1 forward/reverse ratio | 4.7× | 2.6× |
| Cat 2: VADER-positive → RoBERTa neg/neu (count) | 4,571 | 4,316 |
| Cat 2: VADER-positive → RoBERTa neg/neu (% of scope) | 13.15% | 12.41% |

Cat 1 forward/reverse ratio = (VADER pos/neu → RoBERTa neg) ÷ (VADER neg → RoBERTa pos/neu),
computed on topic-scope datasets.

### 4d. Mixed-sentiment flagged rows

**Full cleaned datasets (true prevalence)**

| App | Flagged count | % of cleaned dataset |
|---|---|---|
| HealthifyMe | 964 | 1.53% |
| MyFitnessPal | 19,252 | 2.95% |

**Topic-scope datasets (subset scored by RoBERTa/VADER in §4c — diluted by MyFitnessPal's stratified sampling; not comparable to the full-dataset row above)**

| App | Flagged count | % of topic-scope |
|---|---|---|
| HealthifyMe | 962 | 2.77% |
| MyFitnessPal | 1,258 | 3.62% |

Flag criteria: rating ≤ 3 AND review_description matches at least one temporal/contrast regex pattern.