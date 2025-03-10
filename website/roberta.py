import torch
from transformers import RobertaForSequenceClassification
from model_training.data_processing import tokenize


def roberta_classify(text: str, model_name: str = "../model_training/Roberta_Model_testing_6", temperature: int = 5) -> tuple[int, float]:
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    inputs = tokenize(text, for_torch=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), confidence.item()

