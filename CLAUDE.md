# HealthifyMe vs MyFitnessPal — Text Analytics Rebuild

## Goal
Rebuild a course project (originally done in Orange Data Mining) in Python.
Upgrades: LDA → BERTopic, VADER-only → VADER + transformer sentiment compared,
no cluster validation → real silhouette scores.
Output: comparative product diagnosis + RICE-prioritized recommendation memo,
for a GitHub portfolio project.

## Data
healthifyme_reviews.csv, myfitnesspal_reviews.csv — raw Play Store exports, uncleaned.
Known issues: duplicates, missing text, non-English reviews, HTML/mojibake artifacts,
rating-text mismatches.

## Stack
pandas, sentence-transformers (all-MiniLM-L6-v2), bertopic, vaderSentiment,
transformers (cardiffnlp/twitter-roberta-base-sentiment-latest), scikit-learn,
matplotlib/seaborn.

## Constraints
- Time-boxed build — working beats perfect.
- Cache expensive steps (embeddings) to disk, never recompute.
- Analyze HealthifyMe and MyFitnessPal separately; combine only for comparison tables.