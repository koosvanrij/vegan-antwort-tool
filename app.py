from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
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
from deep_translator import GoogleTranslator
import deepl

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

# create DB
with app.app_context():
    db.create_all()

# Claude-Client initialisieren
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Sentence Transformer Modell laden (einmalig beim Start)
print("Lade Semantic-Modell...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Unterstützt DE/EN/NL
print("Modell geladen!")

# JSON-Datenbank laden
with open("antworten.json", "r", encoding="utf-8") as f:
    antwort_db = json.load(f)

# Antworten und Embeddings laden
with open("antworten.json", "r", encoding="utf-8") as f:
    antwort_db = json.load(f)

db_embeddings = np.load("db_embeddings.npy")



def translate_to_english(text, detected_language):
    """
    Translate user argument to English using Google Translate (no Claude API calls!)
    """
    if detected_language == 'en':
        return text

    # Fix common langdetect confusion: treat Afrikaans as Dutch
    if detected_language == 'af':
        detected_language = 'nl'

    try:
        translator = GoogleTranslator(source=detected_language, target='en')
        english_text = translator.translate(text)
        print(f"DEBUG: Google Translate '{text}' ({detected_language}) to English: '{english_text}'")
        return english_text
    except Exception as e:
        print(f"DEBUG: Google Translation to English failed: {e}")
        return text  # Fallback to original


def translate_from_english(english_text, target_language):
    """
    Translate English answer to target language using Google Translate (no Claude API calls!)
    """
    if target_language == 'en':
        return english_text

    # Fix common langdetect confusion: treat Afrikaans as Dutch
    if target_language == 'af':
        target_language = 'nl'

    try:
        translator = GoogleTranslator(source='en', target=target_language)
        translated_text = translator.translate(english_text)
        print(f"DEBUG: Google Translate '{english_text}' (en) to {target_language}: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"DEBUG: Google Translation from English failed: {e}")
        return english_text  # Fallback to English

import deepl

# DeepL Translator initialisieren (einmalig beim Start, wie dein Sentence Transformer)
print("Initialisiere DeepL Translator...")
try:
    deepl_translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
    print("DeepL Translator bereit!")
except Exception as e:
    print(f"DeepL Initialization failed: {e}")
    deepl_translator = None

# DeepL language code mapping
DEEPL_LANG_MAP = {
    'de': 'DE',
    'nl': 'NL',
    'fr': 'FR',
    'es': 'ES',
    'it': 'IT',
    'pt': 'PT',
    'ru': 'RU',
    'ja': 'JA',
    'zh': 'ZH',
    'pl': 'PL',
    'af': 'NL',  # Deine Afrikaans->Dutch Logik beibehalten
    'en': 'EN',  # Für Vollständigkeit
}

def translate_to_english_deepl(text, detected_language):
    """
    Translate user argument to English using DeepL (fallback to Google)
    """
    if detected_language == 'en':
        return text

    # Fix common langdetect confusion: treat Afrikaans as Dutch
    if detected_language == 'af':
        detected_language = 'nl'

    # Try DeepL first
    if deepl_translator and detected_language in DEEPL_LANG_MAP:
        try:
            source_lang = DEEPL_LANG_MAP[detected_language]
            result = deepl_translator.translate_text(text, source_lang=source_lang, target_lang='EN-US')
            english_text = result.text
            print(f"DEBUG: DeepL Translate '{text}' ({detected_language}) to English: '{english_text}'")
            return english_text
        except Exception as e:
            print(f"DEBUG: DeepL Translation to English failed: {e}, falling back to Google")

    # Fallback to Google Translate
    return translate_to_english(text, detected_language)

def translate_from_english_deepl(english_text, target_language):
    """
    Translate English answer to target language using DeepL (fallback to Google)
    """
    if target_language == 'en':
        return english_text

    # Fix common langdetect confusion: treat Afrikaans as Dutch
    if target_language == 'af':
        target_language = 'nl'

    # Try DeepL first
    if deepl_translator and target_language in DEEPL_LANG_MAP:
        try:
            target_lang = DEEPL_LANG_MAP[target_language]
            result = deepl_translator.translate_text(english_text, target_lang=target_lang)
            translated_text = result.text
            print(f"DEBUG: DeepL Translate '{english_text}' (en) to {target_language}: '{translated_text}'")
            return translated_text
        except Exception as e:
            print(f"DEBUG: DeepL Translation from English failed: {e}, falling back to Google")

    # Fallback to Google Translate
    return translate_from_english(english_text, target_language)

def find_best_match(user_argument, threshold=0.80):
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

@app.route('/faq_html_page')  # for the link to the faq page <--
def faq():
    return render_template('faq_html_page.html')


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

    # NEW: Translate user argument to English for database search (Google Translate - FREE!)
    english_argument = translate_to_english_deepl(argument, detected_lang)
    print(f"DEBUG: Searching database with English argument: '{english_argument}'")

    # Semantische Suche in der Datenbank (now with English argument)
    best_match, similarity = find_best_match(english_argument)

    if best_match:
        # Match gefunden! Antwort aus Datenbank verwenden
        english_answer = best_match["antwort"]

        # Translate answer to detected language (Google Translate - FREE!)
        print(f"DEBUG: Database match - Translating answer from English to '{detected_lang}'")
        translated_answer = translate_from_english_deepl(english_answer, detected_lang)
        print(f"DEBUG: Final translated answer: '{translated_answer}'")

        return jsonify({
            "antwort": translated_answer,
            "quelle": "datenbank",
            "aehnlichkeit": float(round(similarity, 3)),
            "matching_argument": best_match["argument"],
            "detected_language": detected_lang,
            "english_search_term": english_argument  # DEBUG info
        })

    # Kein Match gefunden → Claude API verwenden
    # Direkter Claude API Call für Vegan-Antwort
    antwort_prompt = f"""As an experienced vegan, respond respectfully and factually to anti-vegan arguments.

    Argument: "{argument}"

    Give a short, fact-based response (max 50 words) that:
    - Is polite but firm
    - Clarifies misconceptions  
    - Encourages reflection
    - Contains no lecturing

    CRITICAL: You MUST respond in the SAME language as the argument, even for non-relevant topics. 
- If argument is in German → respond in German
- If argument is in Spanish → respond in Spanish
- If argument is in French → respond in French
- etc.

NEVER respond in English unless the original argument was in English"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[  # type: ignore
                {"role": "user", "content": antwort_prompt}
            ]
        )

        # Claude antwortet direkt in der richtigen Sprache - keine Übersetzung nötig!
        final_response = message.content[0].text.strip()

        # LOGGING
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
            "detected_language": detected_lang,  # Für Info, aber nicht für Übersetzung verwendet
            "aehnlichkeit": 0.0
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