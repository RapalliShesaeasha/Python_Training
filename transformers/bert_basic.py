from transformers import pipeline

def transformer():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    result = classifier("The product quality is really good")
    print(result)


if __name__ == "__main__":
    transformer()
