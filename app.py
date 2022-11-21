import gradio as gr
from transformers import pipeline

pipe = pipeline("summarization", model="sshleifer/distilbart-xsum-12-3")

def main(in_text):
  print(in_text)
  answer = pipe(in_text, min_length=5, max_length=20)
  print(answer)
  return answer[0]["summary_text"]

with gr.Blocks() as demo:
  gr.Markdown("""# Summarization Engine!""")
  with gr.Row():
    with gr.Column():
      text1 = gr.Textbox(
            label="Input Text",
            lines=1,
        )
      output = gr.Textbox(label="Output Text")
      b1 = gr.Button("Summarize!")
      b1.click(main, inputs=[text1], outputs=output)
  gr.Markdown("""#### powered by *********""")


if __name__ == "__main__":
    demo.launch(debug=True)
