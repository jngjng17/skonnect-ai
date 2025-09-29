from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, datetime, random
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import re

# ========================
# Load TensorFlow models & preprocessors
# ========================
faq_model = load_model("chatbot_model.h5")
event_model = load_model("event_recommender_model.h5")

# FAQ encoders
with open("tokenizer.pkl", "rb") as f:
    faq_tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    faq_label_encoder = pickle.load(f)

# Event encoders
with open("event_tokenizer.pkl", "rb") as f:
    event_tokenizer = pickle.load(f)
with open("event_label_encoder.pkl", "rb") as f:
    event_label_encoder = pickle.load(f)

# ========================
# Load datasets
# ========================
faq_df = pd.read_csv("skonnect_faq_dataset_intents.csv")
event_df = pd.read_csv("Events.csv").fillna("")  # avoid NaNs

# Normalize event_df string columns for matching
for col in ["Recommended Category", "PPAs", "Description", "Reference Code"]:
    if col in event_df.columns:
        event_df[col] = event_df[col].astype(str)

faq_max_len = 20
event_max_len = 50

# ========================
# Incremental learning backup
# ========================
vectorizer = HashingVectorizer(n_features=2**16)
clf = SGDClassifier(loss="log_loss")

if "intent" in faq_df.columns and "patterns" in faq_df.columns:
    X_init = vectorizer.transform(faq_df["patterns"].astype(str).tolist())
    y_init = faq_df["intent"].astype(str).tolist()
    try:
        clf.partial_fit(X_init, y_init, classes=np.unique(y_init))
    except Exception:
        # if partial_fit fails due to classes mismatch on first run, ignore
        pass

# ========================
# Logging
# ========================
LOG_FILE = "chat_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "user_message", "predicted_intent", "bot_response", "model_source"]).to_csv(LOG_FILE, index=False)

def log_conversation(user_message, predicted_intent, bot_reply, source="keras"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([[timestamp, user_message, predicted_intent, bot_reply, source]],
                          columns=["timestamp", "user_message", "predicted_intent", "bot_response", "model_source"])
    log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# Reload incremental learning from logs
try:
    log_data = pd.read_csv(LOG_FILE)
    if not log_data.empty:
        X_logs = vectorizer.transform(log_data["user_message"].astype(str).tolist())
        y_logs = log_data["predicted_intent"].astype(str).tolist()
        if len(set(y_logs)) > 0:
            clf.partial_fit(X_logs, y_logs, classes=np.unique(y_logs))
except Exception as e:
    print("Log reload skipped:", e)

# ========================
# Flask App
# ========================
app = Flask(__name__)
CORS(app)

# Response templates
templates = [
    "Here’s what I found: {answer}",
    "Good question! {answer}",
    "Sure thing 👍 {answer}",
    "Here’s the info you need: {answer}",
    "Absolutely! {answer}",
    "{answer} (hope that clears things up!)",
    "No worries — {answer}",
]

last_responses = {}

# ========================
# User Event Categorizer (keyword -> category)
# ========================
category_keywords = {
    "Education Support": ["education", "school", "scholarship", "study", "tuition", "exam", "learning"],
    "Environmental Protection": ["environment", "tree", "clean up", "recycle", "nature", "river", "eco"],
    "Health": ["health", "clinic", "hospital", "medical", "doctor", "checkup", "vaccine", "clinic"],
    "Sports Development": ["sports", "basketball", "volleyball", "football", "athletics", "soccer", "games"],
    "Capability Building": ["training", "seminar", "workshop", "capacity", "skills", "orientation"],
    "General Administration": ["admin", "office", "barangay", "coordination", "support", "meeting"],
    "Youth Empowerment": ["youth", "leadership", "empowerment", "volunteer", "talent"]
}

def categorize_user_interest(interest_text):
    """Map a raw interest string to one of the categories (title-cased)."""
    if not interest_text:
        return None
    interest = interest_text.lower()
    # exact phrase matching first
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', interest):
                return category
    # fallback: check substrings
    for category, keywords in category_keywords.items():
        if any(kw in interest for kw in keywords):
            return category
    return None

# ========================
# Helpers
# ========================
def ensure_minimum_words(text, min_words=40):
    words = text.split()
    if len(words) < min_words:
        filler = (
            " To provide more context, our SK council continuously develops programs "
            "and projects to benefit the youth and community. We encourage participation "
            "and feedback so everyone in Brgy. Buhangin benefits from our initiatives."
        )
        text = text + filler
    return text

def generate_dynamic_reply(base_reply, intent):
    chosen_template = random.choice(templates)
    reply = chosen_template.format(answer=base_reply)

    if intent in last_responses and last_responses[intent] == reply:
        alt_templates = [t for t in templates if t.format(answer=base_reply) != reply]
        if alt_templates:
            reply = random.choice(alt_templates).format(answer=base_reply)

    reply = ensure_minimum_words(reply, 40)
    last_responses[intent] = reply
    return reply

def classify_message(message, event_threshold=0.70, faq_threshold=0.40):
    # FAQ prediction
    faq_seq = faq_tokenizer.texts_to_sequences([message])
    faq_padded = pad_sequences(faq_seq, maxlen=faq_max_len, padding="post")
    faq_pred = faq_model.predict(faq_padded)
    faq_conf = float(np.max(faq_pred))
    faq_intent = faq_label_encoder.inverse_transform([np.argmax(faq_pred)])[0]

    # Event prediction
    event_seq = event_tokenizer.texts_to_sequences([message])
    event_padded = pad_sequences(event_seq, maxlen=event_max_len, padding="post")
    event_pred = event_model.predict(event_padded)
    event_conf = float(np.max(event_pred))
    event_intent = event_label_encoder.inverse_transform([np.argmax(event_pred)])[0]

    # explicit mention of the word event -> favor event (if confident)
    if "event" in message.lower():
        if event_conf >= event_threshold:
            return "event", event_intent, event_conf
        else:
            return "faq", faq_intent, faq_conf

    # otherwise prioritize FAQ if it's reasonably confident
    if faq_conf >= faq_threshold:
        return "faq", faq_intent, faq_conf

    # fallback to event if FAQ is weak and event confident
    if event_conf >= event_threshold:
        return "event", event_intent, event_conf

    # ultimate fallback -> FAQ
    return "faq", faq_intent, faq_conf

# ========================
# Event recommendation: dedupe + return all matches
# ========================
def clean_string(s):
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s)).strip()

def dedupe_events(df):
    """Remove duplicate events by Reference Code first, then by (PPA + Description) fingerprint."""
    if df.empty:
        return df
    # remove duplicates by reference code if present
    if "Reference Code" in df.columns:
        df = df.drop_duplicates(subset=["Reference Code"])
    # fingerprint to avoid near-duplicate rows
    if {"PPAs", "Description"}.issubset(df.columns):
        df["fingerprint"] = (df["PPAs"].astype(str).str.lower().str.strip() + "||" + df["Description"].astype(str).str.lower().str.strip())
        df = df.drop_duplicates(subset=["fingerprint"])
        df = df.drop(columns=["fingerprint"])
    return df

def sort_events(df):
    """Attempt to sort by Period of Implementation if present (simple alphanumeric)."""
    if "Period of Implementation" in df.columns:
        # keep as-is if empty or not sortable; otherwise sort ignoring empty strings
        try:
            df["__period_sort"] = df["Period of Implementation"].astype(str)
            df = df.sort_values(by="__period_sort", ascending=True)
            df = df.drop(columns=["__period_sort"])
        except Exception:
            pass
    return df

def make_summary_from_row(row):
    return (
        f"📌 {clean_string(row.get('PPAs',''))} ({clean_string(row.get('Recommended Category',''))})\n"
        f"📝 {clean_string(row.get('Description',''))}\n"
        f"🎯 Expected Result: {clean_string(row.get('Expected Result',''))}\n"
        f"📅 Implementation: {clean_string(row.get('Period of Implementation',''))}\n"
        f"👥 Responsible: {clean_string(row.get('Person Responsible',''))}\n"
        f"🔖 Reference: {clean_string(row.get('Reference Code',''))}"
    )

def recommend_event_all(category, limit=None):
    """Return all non-duplicate events matching the category (case-insensitive).
       If limit is provided, truncate to that many; otherwise return all."""
    if not category:
        return []
    cat_lower = category.strip().lower()
    # match rows where Recommended Category equals category (case-insensitive)
    if "Recommended Category" in event_df.columns:
        mask = event_df["Recommended Category"].astype(str).str.lower().str.strip() == cat_lower
        matches = event_df[mask].copy()
    else:
        matches = event_df[event_df.apply(lambda r: False, axis=1)].copy()

    if matches.empty:
        return []

    matches = dedupe_events(matches)
    matches = sort_events(matches)

    if limit is not None and len(matches) > limit:
        matches = matches.head(limit)

    recommendations = []
    for _, row in matches.iterrows():
        rec = {
            "reference_code": clean_string(row.get("Reference Code", "")),
            "ppa": clean_string(row.get("PPAs", "")),
            "description": clean_string(row.get("Description", "")),
            "expected_result": clean_string(row.get("Expected Result", "")),
            "period": clean_string(row.get("Period of Implementation", "")),
            "responsible": clean_string(row.get("Person Responsible", "")),
            "category": clean_string(row.get("Recommended Category", "")),
        }
        rec["summary"] = make_summary_from_row(row)
        recommendations.append(rec)

    return recommendations

# ========================
# Routes
# ========================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    message = (data.get("message") or "").strip()
    message_lc = message.lower()
    raw_interest = data.get("interest")  # optional: free-text interest from frontend
    requested_limit = data.get("limit")  # optional: limit number of recs if desired

    # categorize user interest (e.g., 'basketball' -> 'Sports Development')
    categorized_interest = categorize_user_interest(raw_interest) if raw_interest else None

    model_type, predicted_intent, confidence = classify_message(message_lc)

    bot_reply = "I'm not sure how to respond yet."
    recommendations = []

    # FAQ handling (unchanged)
    if model_type == "faq":
        if predicted_intent in faq_df["intent"].values:
            responses = faq_df[faq_df["intent"] == predicted_intent]["bot_response"].tolist()
            if responses:
                base_reply = random.choice(responses)
                bot_reply = generate_dynamic_reply(base_reply, predicted_intent)
        else:
            bot_reply = "Sorry, I couldn't find an FAQ answer for that."

    # Event handling — smarter, deduped, returns all matches for category
    else:  # model_type == "event"
        # priority 1: if user explicitly provided a categorized interest -> use that category
        if categorized_interest:
            recommendations = recommend_event_all(categorized_interest, limit=requested_limit)
            if recommendations:
                bot_reply = f"Based on your interest in **{categorized_interest}**, here are {len(recommendations)} event(s):\n\n" + \
                           "\n\n".join([r["summary"] for r in recommendations])
            else:
                bot_reply = f"Sorry — I couldn't find events for your interest '{categorized_interest}'."

        # priority 2: if message explicitly mentions a category keyword (like 'sports' or 'basketball')
        elif any(cat_keyword in message_lc for kwlist in category_keywords.values() for cat_keyword in kwlist):
            # try to map message itself to category
            mapped = categorize_user_interest(message)
            if mapped:
                recommendations = recommend_event_all(mapped, limit=requested_limit)
                if recommendations:
                    bot_reply = f"Here are {len(recommendations)} event(s) under **{mapped}**:\n\n" + \
                               "\n\n".join([r["summary"] for r in recommendations])
                else:
                    bot_reply = f"No events found under '{mapped}'."
            else:
                # fallback to predicted_intent
                recommendations = recommend_event_all(predicted_intent, limit=requested_limit)
                if recommendations:
                    bot_reply = f"I found {len(recommendations)} event(s) under '{predicted_intent}':\n\n" + \
                               "\n\n".join([r["summary"] for r in recommendations])
                else:
                    bot_reply = f"No events found for '{predicted_intent}'."

        # priority 3: use predicted_intent from event model
        else:
            recommendations = recommend_event_all(predicted_intent, limit=requested_limit)
            if recommendations:
                bot_reply = f"I found {len(recommendations)} event(s) under '{predicted_intent}':\n\n" + \
                           "\n\n".join([r["summary"] for r in recommendations])
            else:
                bot_reply = f"No events found for '{predicted_intent}'."

    # log conversation (keep model_type and predicted_intent for analytics)
    log_conversation(message, predicted_intent, bot_reply, model_type)

    return jsonify({
        "intent": predicted_intent,
        "confidence": float(confidence),
        "response": bot_reply,
        "recommendations": recommendations,
        "source": model_type,
        "categorized_interest": categorized_interest
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    message = (data.get("message") or "").lower()
    correct_intent = data.get("correct_intent")

    if message and correct_intent:
        X_new = vectorizer.transform([message])
        try:
            clf.partial_fit(X_new, [correct_intent])
        except Exception:
            pass

        log_conversation(message, correct_intent, "Corrected by user", source="feedback")
        return jsonify({"status": "updated", "new_intent": correct_intent})
    return jsonify({"status": "failed", "reason": "missing message or correct_intent"}), 400

# ========================
# Run Server
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
