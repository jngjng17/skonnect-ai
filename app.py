from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, datetime, random
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

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
event_df = pd.read_csv("skonnect_event_dataset.csv")

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
    clf.partial_fit(X_init, y_init, classes=np.unique(y_init))

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

def classify_message(message):
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

    return ("event", event_intent, event_conf) if event_conf > faq_conf else ("faq", faq_intent, faq_conf)

def recommend_event(predicted_category, top_n=3):
    matches = event_df[event_df["Recommended Category"].str.lower() == predicted_category.lower()]
    if matches.empty:
        return [{"message": f"No events found for category: {predicted_category}"}]

    sampled = matches.sample(min(top_n, len(matches)))
    recommendations = []
    for _, row in sampled.iterrows():
        recommendations.append({
            "reference_code": row.get("Reference Code", ""),
            "ppa": row.get("PPAs", ""),
            "description": row.get("Description", ""),
            "expected_result": row.get("Expected Result", ""),
            "period": row.get("Period of Implementation", ""),
            "responsible": row.get("Person Responsible", ""),
            "category": row.get("Recommended Category", "")
        })
    return recommendations

# ========================
# Routes
# ========================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "").lower()

    model_type, predicted_intent, confidence = classify_message(message)

    bot_reply = "I'm not sure how to respond yet."
    recommendations = []

    if model_type == "faq":
        if predicted_intent in faq_df["intent"].values:
            responses = faq_df[faq_df["intent"] == predicted_intent]["bot_response"].tolist()
            if responses:
                base_reply = random.choice(responses)
                bot_reply = generate_dynamic_reply(base_reply, predicted_intent)
    else:
        recommendations = recommend_event(predicted_intent, top_n=3)
        if recommendations and "message" not in recommendations[0]:
            bot_reply = f"I found {len(recommendations)} event(s) for category '{predicted_intent}'. Here are some suggestions!"
        else:
            bot_reply = recommendations[0]["message"]

    log_conversation(message, predicted_intent, bot_reply, model_type)

    return jsonify({
        "intent": predicted_intent,
        "confidence": confidence,
        "response": bot_reply,
        "recommendations": recommendations,
        "source": model_type
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    message = data["message"].lower()
    correct_intent = data["correct_intent"]

    X_new = vectorizer.transform([message])
    clf.partial_fit(X_new, [correct_intent])

    log_conversation(message, correct_intent, "Corrected by user", source="feedback")

    return jsonify({"status": "updated", "new_intent": correct_intent})

# ========================
# Run Server
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
