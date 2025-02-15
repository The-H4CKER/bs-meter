# Website/endpoint.py
from flask import Blueprint, request, jsonify

endpoint_bp = Blueprint('endpoint_bp', __name__)

@endpoint_bp.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data.get('text', '')
    # Process the text to compute a BS value (example logic)
    value = min(len(text) * 10, 100)  # For instance, using text length as a placeholder
    return jsonify({'value': value})
