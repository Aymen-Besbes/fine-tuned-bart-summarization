import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration

# Constants
MODEL_REPO = "AymB2/fine_tuned_bart_model"
MAX_INPUT_LENGTH = 133
MAX_SUMMARY_LENGTH = 31

# Load tokenizer and model
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_REPO)
    model = BartForConditionalGeneration.from_pretrained(MODEL_REPO)
    print(f"Tokenizer and model loaded from {MODEL_REPO}")
except Exception as e:
    raise RuntimeError(f"Failed to load model from Hugging Face: {e}")

# Summarization function
def summarize(text):
    if not text or len(text.strip()) == 0:
        return "Please enter some text."

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LENGTH
        ).to(model.device)

        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SUMMARY_LENGTH,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Enter text to summarize..."),
    outputs=gr.Textbox(),
    title="🧠 Fine-Tuned BART Summarizer",
    description="Summarize AI/LLM-related text using a fine-tuned BART model hosted on Hugging Face."
)

# Launch app
if __name__ == "__main__":
    demo.launch()
