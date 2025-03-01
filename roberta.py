import torch
from random import shuffle
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from data_processing import load_articles_from_folder, tokenize


def roberta_train(pure_articles_dir: str, bs_articles_dir: str, save_dir: str, model_name: str = "roberta_base") -> bool:
    pure_articles = load_articles_from_folder(pure_articles_dir, 0)[:20]
    bs_articles = load_articles_from_folder(bs_articles_dir, 1)[:20]
    shuffle(pure_articles)
    shuffle(bs_articles)

    n = int(len(pure_articles) * 0.8)
    m = int(len(bs_articles) * 0.8)
    train_texts, train_labels = zip(*(pure_articles[:n] + bs_articles[:m]))
    val_texts, val_labels = zip(*(pure_articles[n:] + bs_articles[m:]))

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    test_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    if trainer.evaluate()["eval_loss"] <= 1e-4:
        trainer.save_model(save_dir)
        return True
    return False


def roberta_classify(text: str, model_name: str) -> tuple[int, float]:
    model = RobertaForSequenceClassification(model_name)
    inputs = tokenize(text)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), confidence.item()


def main():
    # example usages
    roberta_train("dataset/nature_articles", "dataset/chatgpt_articles", "./Roberta_Model_testing")
    roberta_classify("Hello World!", "./Roberta_Model")


if __name__ == "__main__":
    main()
