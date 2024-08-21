from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=10, min_length=8, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    text = "Your extracted text goes here."
    summary = summarize_text(text)
    print(summary)
