from flask import Flask, request, jsonify, render_template
from graph import graph
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True)

    # Validate input — beginner note: always check user input before using it
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body"}), 400

    question = data["question"]

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "'question' must be a non-empty string"}), 400

    clean_question = question.strip()

    try:
        result = graph.invoke({
            "question": clean_question,
            "intent": "",
            "answer": ""
        })
        answer = result.get("answer") or "I could not generate an answer at this time."
    except Exception as e:
        logger.error("Graph invoke failed: %s", e, exc_info=True)
        return jsonify({
            "error": str(e),
            "answer": "Sorry, an internal error occurred while generating the answer."
        }), 500

    return jsonify({
        "question": clean_question,
        "answer": answer
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # NEVER use debug=True in production — it enables a browser-based code executor
    app.run(host="0.0.0.0", port=5000, debug=False)