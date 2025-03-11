# website/endpoint.py
import io
from flask import Blueprint, request, jsonify
import PyPDF2
import docx
import numpy as np

from website import xgb
from website.roberta import roberta_classify

endpoint_bp = Blueprint('endpoint_bp', __name__)


def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(file_stream):
    doc = docx.Document(file_stream)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


@endpoint_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename.lower()
    mode = request.form.get('mode', 'general')

    if filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    text = ' '.join([i for i in text.split() if i.isalnum()])

    type, val = roberta_classify(text, "./models/RoBERTa")
    if type == 0:
        val = 1 - val
    value1 = min(100, log_transform(xgb.score(text)[0][1] * 100))
    value2 = val * 100
    value = str((value1 + value2) // 2)
    print(value1, value2, value)
    return jsonify({'value': value})


@endpoint_bp.route("/process", methods=["POST"])
def process_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    mode = data.get("mode", "general")

    if len(text) == 0:
        return jsonify({"error": "Empty text"}), 400

    type, val = roberta_classify(text, "./models/RoBERTa")
    if type == 0:
        val = 1 - val
    value1 = min(100, log_transform(xgb.score(text)[0][1] * 100))
    value2 = val * 100
    value = str((value1 + value2) // 2)
    print(value1, value2, value)
    return jsonify({"value": value})


def log_transform(x):
    # Constants
    a = 45.56
    b = 0.1

    # Apply logarithmic transformation
    y = a * np.log(b * x + 1)

    return y

    # print(x)
    # return 100-(-np.log10(x/100) * 30)


