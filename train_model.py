"""Train the SupportPilot classifier, save it, and evaluate on a separate set."""

from classifier import save_classifier, train_classifier

if __name__ == "__main__":
    from evaluation import evaluate

    model, split_accuracy = train_classifier()
    print(f"Kaggle held-out classification accuracy: {split_accuracy:.1%}")
    print(f"Saved model to {save_classifier(model)}")
    evaluate(model)
