from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def seq2seq():
    model_name = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Input text (task prefix is important for T5)
    input_text = "summarize: Transformers are very powerful models for NLP tasks"

    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        max_length=30
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Output:", result)


if __name__ == "__main__":
    seq2seq()