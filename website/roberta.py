import torch
from transformers import RobertaForSequenceClassification
from model_training.data_processing import tokenize


def roberta_classify(text: str, model_name: str = "../model_training/Roberta_Model_testing_6") -> float:
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    inputs = tokenize(text, for_torch=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    term = abs((1 if predicted_class.item() == 0 else 0) - confidence.item())*100 - 50
    val = 0.00000000004*term**7 + 0.000000034*term**5 + 0.162*term + 50
    return val
