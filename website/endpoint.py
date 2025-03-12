# website/endpoint.py
import io
from flask import Blueprint, request, jsonify
import PyPDF2
import docx
import numpy as np
import re

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

    p_text = parse_text(text)

    value1 = min(100, log_transform(xgb.score(p_text)[0][1] * 100))
    value2 = roberta_classify(text, "./models/RoBERTa")
    value = str((value1 + value2) // 2)
    print(value1, value2, value)
    print(mode)
    if mode == "both":
        return jsonify({'value': value})
    elif mode == "xgb":
        return jsonify({'value': str(int(value1))})
    elif mode == "roberta":
        return jsonify({'value': str(int(value2))})
    else:
        return jsonify({'error': 'Unsupported mode type'}), 400


@endpoint_bp.route("/process", methods=["POST"])
def process_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    p_text = parse_text(data["text"])
    text = data["text"]
    mode = data.get("mode", "general")
    print(text)
    if len(text) == 0:
        return jsonify({"error": "Empty text"}), 400

    value1 = min(100, log_transform(xgb.score(p_text)[0][1] * 100))
    value2 = roberta_classify(text, "./models/RoBERTa")
    value = str((value1 + value2) // 2)
    print(value1, value2, value)
    print(mode)
    if mode == "both":
        return jsonify({'value': value})
    elif mode == "xgb":
        return jsonify({'value': str(int(value1))})
    elif mode == "roberta":
        return jsonify({'value': str(int(value2))})
    else:
        return jsonify({'error': 'Unsupported mode type'}), 400



def log_transform(x):
    # Constants
    a = 45.56
    b = 0.1

    # Apply logarithmic transformation
    y = a * np.log(b * x + 1)

    return y

    # print(x)
    # return 100-(-np.log10(x/100) * 30)
    #print(x)
    # return 100-(-np.log10(x/100) * 30)
    #return x


def parse_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9.]+', ' ', text)
    return cleaned_text
