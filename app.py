import os
import time
import threading
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS

# Your analyzer implementation
from attention_count import RealTimeAttentionAnalyzer

from dotenv import load_dotenv
load_dotenv()


# ---------- Firebase Admin / Firestore ----------
import firebase_admin
from firebase_admin import credentials, firestore

FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "key.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()
# ------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Runtime state
analyzer: Optional[RealTimeAttentionAnalyzer] = None
analyzer_thread: Optional[threading.Thread] = None
saver_thread: Optional[threading.Thread] = None
is_tracking = False
lock = threading.Lock()

# Session/user context provided by frontend
current_uid: Optional[str] = None
current_session_id: Optional[str] = None

# Index of last row saved for the current (uid, sessionId)
last_saved_index = -1
stop_saver = threading.Event()

# Firestore paths
COL_SESSIONS = "attention_analysis_sessions"  # root collection
COL_SAMPLES = "samples"                       # subcollection per session


def ensure_analyzer():
    """Create analyzer and background threads once."""
    global analyzer, analyzer_thread, saver_thread
    if analyzer is None:
        analyzer = RealTimeAttentionAnalyzer()
        analyzer_thread = threading.Thread(target=analyzer.run, daemon=True)
        analyzer_thread.start()

        saver_thread = threading.Thread(target=_firestore_saver_loop, daemon=True)
        saver_thread.start()
    return analyzer


def latest_sample() -> Optional[Dict[str, Any]]:
    """Return the most recent row from analyzer.data as a flat dict."""
    if analyzer is None or not analyzer.data.get("timestamp"):
        return None
    idx = len(analyzer.data["timestamp"]) - 1
    return {k: analyzer.data[k][idx] for k in analyzer.data.keys()}


def _firestore_saver_loop():
    """
    Poll every ~1s. When analyzer has a new appended row (≈ every 10s post-calibration),
    write it to Firestore under:
      attention_analysis_sessions/{uid}_{sessionId}/samples/{autoId}
    """
    global last_saved_index
    while not stop_saver.is_set():
        try:
            # Only save when we have a uid and analyzer has data
            if analyzer and analyzer.data.get("timestamp") and current_uid:
                current_len = len(analyzer.data["timestamp"])
                # Save any unsaved rows (handles bursts or catch-up)
                while last_saved_index < current_len - 1:
                    next_idx = last_saved_index + 1

                    # Build the next document from the same index across keys
                    try:
                        doc = {k: analyzer.data[k][next_idx] for k in analyzer.data.keys()}
                    except Exception as inner_e:
                        print(f"[FirestoreSaver] Skipping index {next_idx}: {inner_e}")
                        last_saved_index = next_idx
                        continue

                    uid = current_uid
                    sess = current_session_id or "default_session"
                    sess_doc_id = f"{uid}"

                    # Upsert session metadata (idempotent)
                    session_ref = db.collection(COL_SESSIONS).document(sess_doc_id)
                    session_ref.set(
                        {
                            "studentId": uid,
                            "sessionId": sess,
                            "updatedAt": firestore.SERVER_TIMESTAMP,
                        },
                        merge=True,
                    )

                    # Write the sample
                    session_ref.collection(COL_SAMPLES).add(
                        {
                            **doc,
                            "studentId": uid,
                            "sessionId": sess,
                            "savedAt": firestore.SERVER_TIMESTAMP,
                        }
                    )

                    last_saved_index = next_idx

            time.sleep(1.0)
        except Exception as e:
            # Log and keep running
            print(f"[FirestoreSaver] Error: {e}")
            time.sleep(2.0)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Start/continue analysis using a uid provided by the frontend.
    Body JSON (uid required):
      {
        "uid": "<firebase_user_uid>",     # REQUIRED (you send from Angular)
        "sessionId": "2025-08-10-AM",     # optional
        "track": true                     # optional, default true
      }
    Returns latest sample if available.
    """
    global is_tracking, current_uid, current_session_id, last_saved_index

    body = request.get_json(silent=True) or {}
    uid = body.get("uid")
    if not uid:
        return jsonify({"error": "uid is required in request body"}), 400

    session_id = body.get("sessionId")
    track = bool(body.get("track", True))

    with lock:
        ensure_analyzer()

        # If uid or session changed, start saving from "now" for the new context
        uid_changed = (uid != current_uid)
        sess_changed = (session_id != current_session_id)
        if uid_changed or sess_changed:
            if analyzer and analyzer.data.get("timestamp"):
                # Don't backfill old rows into the new session
                last_saved_index = len(analyzer.data["timestamp"]) - 1
            else:
                last_saved_index = -1

            current_uid = uid
            current_session_id = session_id

        is_tracking = track
        analyzer.is_tracking = is_tracking  # honored inside process_frame()

    sample = latest_sample()
    return jsonify(
        {
            "status": "running" if is_tracking else "paused",
            "studentId": current_uid,
            "sessionId": current_session_id,
            "latest": sample if sample else "No data yet (calibration or first 10s window not complete).",
        }
    ), 200

@app.route("/overall/<uid>", methods=["GET"])
def latest_overall_for_user(uid):
    """
    Return the latest 'overall' value for the given uid
    (from Firestore) as:
    [
      {"overall": 71.68}
    ]

    Optional query params:
      - sessionId=<string>  -> only that session
    """
    try:
        session_id = request.args.get("sessionId")

        # Query latest sample for this user (and session if provided)
        q = db.collection_group(COL_SAMPLES).where("studentId", "==", uid)
        if session_id:
            q = q.where("sessionId", "==", session_id)

        # Get newest doc only
        q = q.order_by("savedAt", direction=firestore.Query.DESCENDING).limit(1)

        docs = list(q.stream())
        if not docs:
            return jsonify([]), 200

        result = []
        for d in docs:
            data = d.to_dict() or {}
            val = data.get("overall")
            if isinstance(val, (int, float)):
                result.append({"overall": round(float(val), 2)})

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/all_analysis", methods=["GET"])
def all_analysis():
    """Return full arrays from analyzer.data."""
    if analyzer is None:
        return jsonify({"message": "Analyzer not initialized. Call POST /analyze first."}), 400
    return jsonify(
        {
            "studentId": current_uid,
            "sessionId": current_session_id,
            "data": analyzer.data,
        }
    ), 200


def _shutdown():
    stop_saver.set()
    if saver_thread and saver_thread.is_alive():
        saver_thread.join(timeout=1.0)


if __name__ == "__main__":
    try:
        # Tip: set host="0.0.0.0" if calling from another device on LAN
        app.run(debug=True, threaded=True)
    finally:
        _shutdown()
