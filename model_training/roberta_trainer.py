from random import shuffle
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from model_training.data_processing import load_articles_from_folder, tokenize
import math


def roberta_train(pure_articles_dir: str, bs_articles_dir: str, save_dir: str, model_name: str = "roberta-base", generalise=True, amount=1.0) -> bool:
    pure_articles = load_articles_from_folder(pure_articles_dir, 0)
    bs_articles = load_articles_from_folder(bs_articles_dir, 1)
    shuffle(pure_articles)
    shuffle(bs_articles)

    n = len(pure_articles)
    m = len(bs_articles)
    pure_articles = pure_articles[:int(amount*n)]
    bs_articles = bs_articles[:int(amount*m)]
    n = len(pure_articles)
    m = len(bs_articles)
    total_length = n + m
    length_weighting = 5 - math.floor(math.log(total_length, 10))
    num_epochs = max(length_weighting, 2)
    per_device_eval_batch_size = per_device_train_batch_size = (length_weighting - 1) * 8
    weight_decay = 0.01 * (10 if generalise else 1)
    learning_rate = 3e-5

    train_texts, train_labels = zip(*(pure_articles[:int(n*0.8)] + bs_articles[:int(m*0.8)]))
    val_texts, val_labels = zip(*(pure_articles[int(n*0.8):] + bs_articles[int(m*0.8):]))

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
    test_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    if trainer.evaluate()["eval_loss"] <= 0.1:
        trainer.save_model(save_dir)
        return True
    return False


def main():
    # example usages
    roberta_train("../dataset/nature_articles", "../dataset/chatgpt_articles", "./Roberta_Model_testing_2", generalise=False)
    roberta_train("../dataset/nature_articles", "../dataset/chatgpt_articles", "./Roberta_Model_testing_3", generalise=False, amount=0.5)
    roberta_train("../dataset/nature_articles", "../dataset/chatgpt_articles", "./Roberta_Model_testing_4", generalise=True, amount=0.5)
    roberta_train("../dataset/nature_articles", "../dataset/chatgpt_articles", "./Roberta_Model_testing_5", generalise=False, amount=0.25)
    roberta_train("../dataset/nature_articles", "../dataset/chatgpt_articles", "./Roberta_Model_testing_6", generalise=True, amount=0.25)


if __name__ == "__main__":
    main()
