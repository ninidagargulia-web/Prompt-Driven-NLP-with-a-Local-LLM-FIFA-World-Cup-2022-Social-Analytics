import os
import json
import time
import re
import pandas as pd

# =====================================================================
# 1. CONFIG SECTION
# =====================================================================

OLLAMA_MODEL = "llama3.2:latest"    # falls back to 1b if RAM-limited

CSV_PATH = "task1_selected_tweets_final.csv"          # 100 selected tweets
TEXT_COLUMN = "Tweet"
TRUE_SENTIMENT_COLUMN = "Sentiment"                    # ground-truth label in dataset

OUTPUT_DIR = "outputs"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "task1_results.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "task1_metrics.json")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "task1_business_summary.txt")

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TOPIC_LABELS = ["Match Play", "Referees", "Logistics/Hosting"]

# Limit for quick local testing; set to len(df) for a full run.
MAX_ROWS = None     # None = run on all 100 rows

SLEEP_BETWEEN_CALLS = 0.0   # increase (e.g. 0.5) if you hit local resource limits

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. LOAD DATA
# =====================================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    if MAX_ROWS:
        df = df.head(MAX_ROWS)
    print(f"[LOAD] Loaded {len(df)} tweets from {path}")
    print(f"[LOAD] Sentiment label distribution:\n{df[TRUE_SENTIMENT_COLUMN].value_counts()}\n")
    return df


# =====================================================================
# 3. PROMPT DEFINITIONS
# =====================================================================
# Two documented prompt versions per NLP task (Section 3.3 requirement).

# ---- Sentiment Analysis -------------------------------------------------
# v1: Zero-shot — instruction only, no examples.
def prompt_sentiment_zero_shot(tweet: str) -> tuple[str, str]:
    system = "You are a precise sentiment classifier."
    user = (
        f"Classify the sentiment of this World Cup tweet as exactly one word: "
        f"positive, negative, or neutral.\n\nTweet: \"{tweet}\"\n\nAnswer with one word only."
    )
    return system, user

# v2: Few-shot — worked examples shown before the real input.
# Rationale: few-shot anchors the model on the exact label vocabulary and
# disambiguates sarcasm/banter common in football tweets.
def prompt_sentiment_few_shot(tweet: str) -> tuple[str, str]:
    system = "You are a precise sentiment classifier for football fan tweets."
    user = (
        "Classify each tweet's sentiment as positive, negative, or neutral.\n\n"
        "Tweet: \"What a goal! Best World Cup ever!\"\nSentiment: positive\n\n"
        "Tweet: \"This referee is a disgrace, ruined the match.\"\nSentiment: negative\n\n"
        "Tweet: \"The match starts at 18:00 local time.\"\nSentiment: neutral\n\n"
        f"Tweet: \"{tweet}\"\nSentiment:"
    )
    return system, user


# ---- Topic Classification -----------------------------------------------
# v1: Zero-shot.
def prompt_topic_zero_shot(tweet: str) -> tuple[str, str]:
    system = "You are a topic classifier."
    user = (
        "Classify this World Cup tweet into exactly one category: "
        "'Match Play', 'Referees', or 'Logistics/Hosting'.\n\n"
        f"Tweet: \"{tweet}\"\n\nAnswer with the category name only."
    )
    return system, user

# v2: System-role prompt — persona constrains the model's domain lens.
# Rationale: giving the model a sports-operations persona reduces drift
# toward generic "general comment" classifications.
def prompt_topic_system_role(tweet: str) -> tuple[str, str]:
    system = (
        "You are a sports-operations analyst at a tournament media monitoring desk. "
        "You only ever answer with one of these three category labels: "
        "'Match Play', 'Referees', 'Logistics/Hosting'. "
        "'Match Play' = goals, scorelines, player performance, tactics. "
        "'Referees' = officiating decisions, VAR, cards, penalties. "
        "'Logistics/Hosting' = tickets, stadiums, travel, accommodation, host-country operations."
    )
    user = f"Tweet: \"{tweet}\"\nCategory:"
    return system, user


# ---- Entity Extraction ---------------------------------------------------
# v1: Zero-shot, free-text instruction.
def prompt_entity_zero_shot(tweet: str) -> tuple[str, str]:
    system = "You extract named entities from football tweets."
    user = (
        f"Extract all player names, team/country names, and venues mentioned in this tweet.\n\n"
        f"Tweet: \"{tweet}\"\n\nList them."
    )
    return system, user

# v2: Structured-output — forces valid JSON with a fixed schema.
# Rationale: downstream analytics (Task 2 style dashboards) need machine-
# parseable entities, not free text; JSON schema also makes evaluation of
# the hallucination/parse-failure rate possible.
def prompt_entity_structured(tweet: str) -> tuple[str, str]:
    system = (
        "You extract named entities from football tweets and reply with VALID JSON ONLY — "
        "no prose, no markdown fences."
    )
    user = (
        "Extract entities from the tweet below. Return ONLY a JSON object with exactly these keys:\n"
        '{"players": [], "teams": [], "venues": []}\n'
        "If a category has no entities, return an empty list for it.\n\n"
        f"Tweet: \"{tweet}\""
    )
    return system, user


# =====================================================================
# 4. LLM CALLER FUNCTION
# =====================================================================

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Sends system + user prompt to the local Ollama model and returns raw text."""
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


def safe_call_llm(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    """Wraps call_llm with basic retry/error handling so one bad call doesn't kill the run."""
    for attempt in range(retries + 1):
        try:
            return call_llm(system_prompt, user_prompt)
        except Exception as e:
            if attempt == retries:
                print(f"[WARN] LLM call failed after {retries+1} attempts: {e}")
                return ""
            time.sleep(1.0)
    return ""


# =====================================================================
# 5. TASK LOOP — run every tweet through every task/prompt-version
# =====================================================================

def normalize_sentiment(raw: str) -> str:
    raw = raw.lower().strip()
    for label in SENTIMENT_LABELS:
        if label in raw:
            return label
    return "unparsed"


def normalize_topic(raw: str) -> str:
    raw_low = raw.lower()
    for label in TOPIC_LABELS:
        if label.lower() in raw_low:
            return label
    return "unparsed"


def parse_entity_json(raw: str) -> dict:
    """Attempts to parse a JSON entity object out of raw model output."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"players": [], "teams": [], "venues": [], "_parse_ok": False}
    try:
        obj = json.loads(match.group(0))
        obj.setdefault("players", [])
        obj.setdefault("teams", [])
        obj.setdefault("venues", [])
        obj["_parse_ok"] = True
        return obj
    except json.JSONDecodeError:
        return {"players": [], "teams": [], "venues": [], "_parse_ok": False}


def weak_topic_label(tweet: str) -> str:
    """
    Heuristic keyword-based PSEUDO gold-label for topic classification.
    The dataset has no human-annotated topic column, so this lightweight
    rule-based labeller acts as an approximate benchmark for the LLM
    comparison only — NOT a substitute for manual annotation. This
    limitation is reported explicitly in the error-analysis section.
    """
    t = tweet.lower()
    ref_kw = ["referee", " ref ", "var", "penalty", "red card", "yellow card", "foul"]
    logi_kw = ["ticket", "hotel", "stadium", "flight", "traffic", "security",
               "queue", "transport", "hosting", "accommodation", "visa", "fan zone", "metro"]
    if any(k in t for k in ref_kw):
        return "Referees"
    if any(k in t for k in logi_kw):
        return "Logistics/Hosting"
    return "Match Play"


def run_task_loop(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(df)
    for i, row in df.iterrows():
        tweet = str(row[TEXT_COLUMN])
        true_sentiment = str(row[TRUE_SENTIMENT_COLUMN]).lower().strip()
        weak_topic = weak_topic_label(tweet)

        print(f"[{i+1}/{total}] processing tweet_id={row.get('tweet_id', i)}")

        # ---- Sentiment: v1 zero-shot vs v2 few-shot ----
        s_sys1, s_user1 = prompt_sentiment_zero_shot(tweet)
        s_raw1 = safe_call_llm(s_sys1, s_user1)
        s_pred1 = normalize_sentiment(s_raw1)
        time.sleep(SLEEP_BETWEEN_CALLS)

        s_sys2, s_user2 = prompt_sentiment_few_shot(tweet)
        s_raw2 = safe_call_llm(s_sys2, s_user2)
        s_pred2 = normalize_sentiment(s_raw2)
        time.sleep(SLEEP_BETWEEN_CALLS)

        # ---- Topic: v1 zero-shot vs v2 system-role ----
        t_sys1, t_user1 = prompt_topic_zero_shot(tweet)
        t_raw1 = safe_call_llm(t_sys1, t_user1)
        t_pred1 = normalize_topic(t_raw1)
        time.sleep(SLEEP_BETWEEN_CALLS)

        t_sys2, t_user2 = prompt_topic_system_role(tweet)
        t_raw2 = safe_call_llm(t_sys2, t_user2)
        t_pred2 = normalize_topic(t_raw2)
        time.sleep(SLEEP_BETWEEN_CALLS)

        # ---- Entity Extraction: v1 zero-shot vs v2 structured/JSON ----
        e_sys1, e_user1 = prompt_entity_zero_shot(tweet)
        e_raw1 = safe_call_llm(e_sys1, e_user1)
        time.sleep(SLEEP_BETWEEN_CALLS)

        e_sys2, e_user2 = prompt_entity_structured(tweet)
        e_raw2 = safe_call_llm(e_sys2, e_user2)
        e_parsed2 = parse_entity_json(e_raw2)
        time.sleep(SLEEP_BETWEEN_CALLS)

        records.append({
            "tweet_id": row.get("tweet_id", i),
            "tweet": tweet,
            "true_sentiment": true_sentiment,

            "sentiment_zero_shot_raw": s_raw1,
            "sentiment_zero_shot_pred": s_pred1,
            "sentiment_zero_shot_correct": int(s_pred1 == true_sentiment),

            "sentiment_few_shot_raw": s_raw2,
            "sentiment_few_shot_pred": s_pred2,
            "sentiment_few_shot_correct": int(s_pred2 == true_sentiment),

            "weak_topic_label": weak_topic,

            "topic_zero_shot_raw": t_raw1,
            "topic_zero_shot_pred": t_pred1,
            "topic_zero_shot_correct": int(t_pred1 == weak_topic),

            "topic_system_role_raw": t_raw2,
            "topic_system_role_pred": t_pred2,
            "topic_system_role_correct": int(t_pred2 == weak_topic),

            "entity_zero_shot_raw": e_raw1,

            "entity_structured_raw": e_raw2,
            "entity_structured_json_valid": int(e_parsed2["_parse_ok"]),
            "entity_structured_players": json.dumps(e_parsed2["players"]),
            "entity_structured_teams": json.dumps(e_parsed2["teams"]),
            "entity_structured_venues": json.dumps(e_parsed2["venues"]),
        })

    return pd.DataFrame(records)


# =====================================================================
# 6. EVALUATION — accuracy per class, per prompt version
# =====================================================================

def evaluate(results: pd.DataFrame) -> dict:
    metrics = {}

    # Sentiment accuracy overall + per class, both prompt versions
    for version in ["zero_shot", "few_shot"]:
        col = f"sentiment_{version}_correct"
        metrics[f"sentiment_{version}_overall_accuracy"] = round(results[col].mean(), 4)
        per_class = {}
        for label in SENTIMENT_LABELS:
            subset = results[results["true_sentiment"] == label]
            if len(subset) > 0:
                per_class[label] = round(subset[col].mean(), 4)
        metrics[f"sentiment_{version}_per_class_accuracy"] = per_class

    # Topic accuracy vs weak/heuristic label, both prompt versions
    for version in ["zero_shot", "system_role"]:
        col = f"topic_{version}_correct"
        metrics[f"topic_{version}_overall_accuracy_vs_weak_label"] = round(results[col].mean(), 4)
        per_class = {}
        for label in TOPIC_LABELS:
            subset = results[results["weak_topic_label"] == label]
            if len(subset) > 0:
                per_class[label] = round(subset[col].mean(), 4)
        metrics[f"topic_{version}_per_class_accuracy"] = per_class

    # Entity extraction: JSON validity rate is the structured-output's
    # objective evaluation metric (no gold entity labels exist).
    metrics["entity_structured_json_valid_rate"] = round(
        results["entity_structured_json_valid"].mean(), 4
    )
    metrics["entity_structured_avg_players_per_tweet"] = round(
        results["entity_structured_players"].apply(lambda x: len(json.loads(x))).mean(), 2
    )
    metrics["entity_structured_avg_teams_per_tweet"] = round(
        results["entity_structured_teams"].apply(lambda x: len(json.loads(x))).mean(), 2
    )

    print("\n[EVAL] ===== Evaluation Summary =====")
    print(json.dumps(metrics, indent=2))
    return metrics


# =====================================================================
# 7. SUMMARY PROMPT — ask the LLM to synthesize a business insight
# =====================================================================

def generate_business_summary(metrics: dict, results: pd.DataFrame) -> str:
    sample_errors = results[results["sentiment_few_shot_correct"] == 0].head(5)
    error_examples = "\n".join(
        f"- \"{r.tweet[:120]}...\" (true={r.true_sentiment}, predicted={r.sentiment_few_shot_pred})"
        for r in sample_errors.itertuples()
    )

    system = "You are a data analyst writing a short business insight summary for tournament organizers."
    user = (
        "Here are evaluation metrics from an LLM-based sentiment/topic/entity pipeline run on "
        "FIFA World Cup 2022 fan tweets:\n\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        f"Example misclassified sentiment cases:\n{error_examples}\n\n"
        "In 4-6 sentences, summarize: (1) how well prompt engineering improved accuracy, "
        "(2) which prompt strategy worked best and why, (3) one practical recommendation for "
        "media analysts monitoring fan sentiment at the next tournament."
    )
    summary = safe_call_llm(system, user)
    if not summary:
        summary = "[Summary generation failed — LLM call returned no output. See metrics JSON instead.]"
    print("\n[SUMMARY]\n" + summary)
    return summary


# =====================================================================
# 8. SAVE OUTPUT
# =====================================================================

def _write_with_retry(write_fn, label: str, retries: int = 3, delay: float = 2.0):
    """Retries a file write a few times to survive transient timeouts
    (e.g. from cloud-synced folders like iCloud Drive/OneDrive/Dropbox)."""
    for attempt in range(retries + 1):
        try:
            write_fn()
            return
        except (TimeoutError, OSError) as e:
            if attempt == retries:
                print(f"[ERROR] Failed to write {label} after {retries+1} attempts: {e}")
                raise
            print(f"[WARN] Write to {label} failed ({e}); retrying in {delay}s...")
            time.sleep(delay)


def save_outputs(results: pd.DataFrame, metrics: dict, summary: str):
    _write_with_retry(lambda: results.to_csv(RESULTS_CSV, index=False), RESULTS_CSV)

    def _write_metrics():
        with open(METRICS_JSON, "w") as f:
            json.dump(metrics, f, indent=2)
    _write_with_retry(_write_metrics, METRICS_JSON)

    def _write_summary():
        with open(SUMMARY_TXT, "w") as f:
            f.write(summary)
    _write_with_retry(_write_summary, SUMMARY_TXT)

    print(f"\n[SAVE] Results  -> {RESULTS_CSV}")
    print(f"[SAVE] Metrics  -> {METRICS_JSON}")
    print(f"[SAVE] Summary  -> {SUMMARY_TXT}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print(f"[CONFIG] Backend = ollama ({OLLAMA_MODEL})")
    df = load_data(CSV_PATH)
    results_df = run_task_loop(df)
    metrics = evaluate(results_df)
    summary = generate_business_summary(metrics, results_df)
    save_outputs(results_df, metrics, summary)
    print("\nDone. Run `python error_analysis.py` next for the qualitative failure analysis.")
