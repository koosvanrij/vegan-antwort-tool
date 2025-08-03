from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import json
import anthropic
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from langdetect import detect
from datetime import datetime, timezone
# ===== RATE LIMITER IMPORTS =====
from functools import wraps
import time
from collections import defaultdict

# .env-Datei laden
load_dotenv()

# Warnungen unterdrücken
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Database configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ===== RATE LIMITER SETUP =====
class SimpleRateLimiter:
    def __init__(self):
        # Store: {ip_address: [timestamp1, timestamp2, ...]}
        self.requests = defaultdict(list)

    def is_allowed(self, ip, max_requests=8, window_minutes=5):
        now = time.time()
        window_seconds = window_minutes * 60

        # Clean old requests outside the window
        self.requests[ip] = [
            req_time for req_time in self.requests[ip]
            if now - req_time < window_seconds
        ]

        # Check if under limit
        if len(self.requests[ip]) < max_requests:
            self.requests[ip].append(now)
            return True

        return False

    def get_remaining_requests(self, ip, max_requests=8, window_minutes=5):
        now = time.time()
        window_seconds = window_minutes * 60

        # Count recent requests
        recent_requests = [
            req_time for req_time in self.requests[ip]
            if now - req_time < window_seconds
        ]

        return max(0, max_requests - len(recent_requests))


# Initialize rate limiter
rate_limiter = SimpleRateLimiter()


def protect_from_bots(max_requests=8, window_minutes=5):
    """
    Rate limiter decorator to protect Claude API calls
    Default: 8 requests per 5 minutes per IP (good for legitimate users, blocks bots)
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get real IP address (handles proxies/load balancers)
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            if ',' in client_ip:  # Handle multiple proxies
                client_ip = client_ip.split(',')[0].strip()

            if not rate_limiter.is_allowed(client_ip, max_requests, window_minutes):
                remaining = rate_limiter.get_remaining_requests(client_ip, max_requests, window_minutes)
                print(f"RATE LIMIT: Blocked request from {client_ip}")

                return jsonify({
                    "error": "Zu viele Anfragen",
                    "message": "Bitte warten Sie einen Moment bevor Sie eine neue Frage stellen.",
                    "limit": f"{max_requests} Anfragen per {window_minutes} Minuten",
                    "remaining": remaining,
                    "quelle": "rate_limit"
                }), 429

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# ===== END RATE LIMITER SETUP =====


# Feedback Model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    email = db.Column(db.String(120))
    rating = db.Column(db.Integer)  # 1-5 star rating if you have that
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    def __repr__(self):
        return f'<Feedback {self.id}: {self.message[:50]}...>'

#Claude API log model
class ClaudeLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    argument = db.Column(db.Text, nullable=False)
    claude_response = db.Column(db.Text, nullable=False)
    detected_language = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    promoted_to_db = db.Column(db.Boolean, default=False)

# Claude-Client initialisieren
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Sentence Transformer Modell laden (einmalig beim Start)
print("Lade Semantic-Modell...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Unterstützt DE/EN/NL
print("Modell geladen!")

# JSON-Datenbank laden
with open("antworten.json", "r", encoding="utf-8") as f:
    antwort_db = json.load(f)

# Embeddings für alle Argumente in der DB erstellen (einmalig beim Start)
print("Erstelle Embeddings für Datenbank...")
db_arguments = [eintrag["argument"] for eintrag in antwort_db]
db_embeddings = model.encode(db_arguments)
print(f"Embeddings für {len(db_arguments)} Argumente erstellt!")


def translate_answer(english_answer, target_language):
    """
    Translate English answer to target language using Claude
    """
    if target_language == 'en':
        return english_answer

    # Fix common langdetect confusion: treat Afrikaans as Dutch
    if target_language == 'af':
        print(f"DEBUG: Detected Afrikaans, treating as Dutch instead")
        target_language = 'nl'

    lang_names = {
        'de': 'German',
        'nl': 'Dutch',
        'af': 'Afrikaans',  # Often confused with Dutch by langdetect
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'pl': 'Polish',
        'ru': 'Russian',
        'sv': 'Swedish',
        'da': 'Danish',
        'no': 'Norwegian',
        'fi': 'Finnish'
    }

    target_lang_name = lang_names.get(target_language, None)

    # If we don't recognize the language code, default to English
    if not target_lang_name:
        print(f"DEBUG: Unknown language code '{target_language}', defaulting to English")
        return english_answer

    prompt = f"""Please translate this vegan response to {target_lang_name}. Keep the same tone and meaning:

"{english_answer}"

Translation:"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]  # type: ignore
        )
        translated_text = message.content[0].text.strip()
        print(f"DEBUG: Successfully translated to {target_lang_name}: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"DEBUG: Translation failed: {e}")
        return english_answer  # Fallback to English if translation fails


def find_best_match(user_argument, threshold=0.75):
    """
    Findet das beste semantische Match in der Datenbank

    Args:
        user_argument (str): Das Benutzer-Argument
        threshold (float): Mindest-Ähnlichkeit (0-1)

    Returns:
        tuple: (best_match_entry, similarity_score) oder (None, 0)
    """
    # Embedding für Benutzer-Argument erstellen
    user_embedding = model.encode([user_argument])

    # Ähnlichkeiten berechnen
    similarities = cosine_similarity(user_embedding, db_embeddings)

    # Bestes Match finden
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]

    # Nur zurückgeben wenn über Threshold
    if best_score >= threshold:
        return antwort_db[best_idx], best_score
    else:
        return None, best_score


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Handle feedback form submissions"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            message = data.get('message', '')
            email = data.get('email', '')
            rating = data.get('rating')
        else:
            message = request.form.get('message', '')
            email = request.form.get('email', '')
            rating = request.form.get('rating')
            if rating:
                rating = int(rating)

        if not message.strip():
            if request.is_json:
                return jsonify({"error": "Feedback message is required"}), 400
            else:
                flash("Bitte geben Sie eine Nachricht ein.")
                return redirect(url_for('index'))

        # Save to database
        new_feedback = Feedback(
            message=message.strip(),
            email=email.strip() if email else None,
            rating=rating
        )

        db.session.add(new_feedback)
        db.session.commit()

        print(f"Feedback saved: ID {new_feedback.id}, Message: {message[:50]}...")

        if request.is_json:
            return jsonify({
                "success": True,
                "message": "Vielen Dank für Ihr Feedback!"
            })
        else:
            flash("Vielen Dank für Ihr Feedback!")
            return redirect(url_for('index'))

    except Exception as e:
        print(f"Error saving feedback: {e}")
        db.session.rollback()

        if request.is_json:
            return jsonify({"error": "Fehler beim Speichern des Feedbacks"}), 500
        else:
            flash("Fehler beim Speichern des Feedbacks.")
            return redirect(url_for('index'))


@app.route("/antwort", methods=["POST"])
@protect_from_bots(max_requests=8, window_minutes=5)
def antwort():
    daten = request.get_json()
    argument = daten["argument"]

    # Detect language of the argument
    try:
        detected_lang = detect(argument)
        print(f"DEBUG: Detected language '{detected_lang}' for argument: '{argument}'")
    except Exception as e:
        detected_lang = 'en'  # Default to English if detection fails
        print(f"DEBUG: Language detection failed, defaulting to English. Error: {e}")

    # Semantische Suche in der Datenbank
    best_match, similarity = find_best_match(argument)

    if best_match:
        # Match gefunden! Antwort aus Datenbank verwenden
        english_answer = best_match["antwort"]

        # Translate to detected language
        print(f"DEBUG: Database match - Translating from English to '{detected_lang}'")
        translated_answer = translate_answer(english_answer, detected_lang)
        # Remove quotes that translation API might add
        translated_answer = translated_answer.strip('"').strip("'")
        print(f"DEBUG: After translation: '{translated_answer}'")  # <-- HINZUFÜGEN
        print(f"DEBUG: Database translation result: '{translated_answer}'")

        return jsonify({
            "antwort": translated_answer,
            "quelle": "datenbank",
            "aehnlichkeit": float(round(similarity, 3)),
            "matching_argument": best_match["argument"],
            "detected_language": detected_lang
        })

    # Kein Match gefunden → Claude API verwenden

    # SCHRITT 1: Relevanz prüfen
    relevanz_prompt = f"""Is this argument related to veganism, soy, tofu, animal welfare, nutrition, environment, or ethics? Answer only "YES" or "NO".

Argument: "{argument}" """

    try:
        relevanz_message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[    # type: ignore
                {"role": "user", "content": relevanz_prompt}
            ]
        )
        relevanz_check = relevanz_message.content[0].text.strip().upper()

        if "NO" in relevanz_check:
            # Argument ist nicht vegan-relevant
            not_relevant_response = "The argument doesn't seem to be vegan-related. I'm happy to help with questions about vegan nutrition or animal welfare."

            # Nur übersetzen wenn nicht Englisch
            if detected_lang != 'en':
                translated_response = translate_answer(not_relevant_response, detected_lang)
                return jsonify({
                    "antwort": translated_response,
                    "quelle": "claude_api_not_relevant",
                    "aehnlichkeit": 0.0,
                    "detected_language": detected_lang
                })
            else:
                return jsonify({
                    "antwort": not_relevant_response,
                    "quelle": "claude_api_not_relevant",
                    "aehnlichkeit": 0.0,
                    "detected_language": detected_lang
                })

        # SCHRITT 2: Vegan-Antwort generieren (in der Original-Sprache)
        antwort_prompt = f"""Als erfahrene/r Veganer/in antwortest du respektvoll und sachlich auf Anti-Vegan-Argumente.

Argument: "{argument}"

Gib eine kurze, faktenbasierte Antwort (max. 50 Wörter) die:
- Höflich aber bestimmt ist
- Missverständnisse klärt  
- Zum Nachdenken anregt
- Keine Belehrung enthält

Answer in the same language as the argument."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[   # type: ignore
                {"role": "user", "content": antwort_prompt}
            ]
        )

        final_response = message.content[0].text.strip()
        # LOGGING antwort zu den database
        try:
            new_claude_log = ClaudeLog(
                argument=argument.strip(),
                claude_response=final_response.strip(),
                detected_language=detected_lang
            )
            db.session.add(new_claude_log)
            db.session.commit()
        except Exception as e:
            print(f"Claude logging failed: {e}")

        return jsonify({
            "antwort": final_response,
            "quelle": "claude_api",
            "aehnlichkeit": float(round(similarity, 3)),
            "detected_language": detected_lang
        })

    except Exception as e:
        return jsonify({
            "antwort": "Fehler bei der KI-Antwort: " + str(e),
            "quelle": "fehler"
        })

@app.route("/debug-matching", methods=["POST"])
def debug_matching():
    """Debug-Route um die Top-5 Matches zu sehen"""
    daten = request.get_json()
    argument = daten["argument"]

    user_embedding = model.encode([argument])
    similarities = cosine_similarity(user_embedding, db_embeddings)[0]

    # Top 5 Matches
    top_indices = np.argsort(similarities)[::-1][:5]

    results = []
    for idx in top_indices:
        results.append({
            "argument": antwort_db[idx]["argument"],
            "similarity": float(round(similarities[idx], 3))
        })

    return jsonify({"top_matches": results})


# ===== OPTIONAL: Rate limit status endpoint =====
@app.route("/api/rate-limit-status")
def rate_limit_status():
    """Optional: Check how many requests are remaining"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    remaining = rate_limiter.get_remaining_requests(client_ip, 8, 5)

    return jsonify({
        "remaining_requests": remaining,
        "limit": "8 requests per 5 minutes",
        "your_ip": client_ip
    })

@app.route("/create-tables")   #das hier alles nur für pycharm, nicht nötig in render
def create_tables():
    db.create_all()
    return "All tables created! Check your DB viewer now."

@app.route("/admin/claude-logs")
def claude_logs():
    logs = ClaudeLog.query.order_by(ClaudeLog.timestamp.desc()).limit(20).all()
    html = "<h1>Claude Logs</h1>"
    if not logs:
        html += "<p>No logs yet. Ask a question that goes to Claude first!</p>"
    for log in logs:
        html += f"""
        <div style="border: 1px solid #ccc; margin: 10px; padding: 10px;">
            <strong>Argument:</strong> {log.argument}<br>
            <strong>Response:</strong> {log.claude_response}<br>
            <strong>Language:</strong> {log.detected_language}<br>
            <strong>Time:</strong> {log.timestamp}<br>
        </div>
        """
    return html

if __name__ == "__main__":
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created!")
        print("Rate limiter activated: 8 requests per 5 minutes per IP")

    app.run(debug=True)