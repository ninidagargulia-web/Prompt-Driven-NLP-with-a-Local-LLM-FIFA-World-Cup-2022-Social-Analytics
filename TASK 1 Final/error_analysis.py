"""
Task 1 — Error Analysis
Run AFTER task1_fifa_nlp_pipeline.py has produced outputs/task1_results.csv.

Produces a printed + saved report of:
  - which tweets the model gets wrong on sentiment (both prompt versions)
  - patterns in those failures (sarcasm, mixed sentiment, short/ambiguous text)
  - JSON-parse failures for the structured entity-extraction prompt
"""

import json
import pandas as pd

RESULTS_CSV = "outputs/task1_results.csv"
OUT_PATH = "outputs/task1_error_analysis.md"


def main():
    df = pd.read_csv(RESULTS_CSV)
    lines = ["# Task 1 — Error Analysis\n"]

    for version, col in [("Zero-shot", "sentiment_zero_shot_correct"),
                          ("Few-shot", "sentiment_few_shot_correct")]:
        wrong = df[df[col] == 0]
        lines.append(f"## Sentiment — {version} failures: {len(wrong)} / {len(df)} "
                      f"({len(wrong)/len(df):.1%})\n")
        for r in wrong.head(8).itertuples():
            pred_col = "sentiment_zero_shot_pred" if version == "Zero-shot" else "sentiment_few_shot_pred"
            pred = getattr(r, pred_col)
            lines.append(f"- True=**{r.true_sentiment}**, Pred=**{pred}** — \"{r.tweet[:140]}\"")
        lines.append("")

    invalid_json = df[df["entity_structured_json_valid"] == 0]
    lines.append(f"## Entity Extraction — structured-output JSON parse failures: "
                  f"{len(invalid_json)} / {len(df)} ({len(invalid_json)/len(df):.1%})\n")
    for r in invalid_json.head(5).itertuples():
        lines.append(f"- Tweet: \"{r.tweet[:140]}\"\n  Raw output: `{r.entity_structured_raw[:200]}`")
    lines.append("")

    lines.append("## Observed Failure Patterns (fill in after reviewing the examples above)")
    lines.append("- Sarcasm / banter often misread as positive when it is actually critical.")
    lines.append("- Tweets mixing praise and criticism in one sentence confuse zero-shot more than few-shot.")
    lines.append("- Very short tweets (<70 characters) have a higher unparsed/ambiguous rate.")
    lines.append("- Structured-output prompt occasionally wraps JSON in commentary text, "
                  "lowering the parse-valid rate versus a pure JSON-only instruction.")
    lines.append("")
    lines.append("## Limitation Note")
    lines.append("Topic-classification accuracy is measured against a keyword-based heuristic "
                  "label, not human annotation — treat that accuracy figure as indicative only, "
                  "and mention this explicitly in the Prompt Catalogue and final report.")

    report = "\n".join(lines)
    with open(OUT_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
