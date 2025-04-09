import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
import time
import pathlib
import json
import re
import ast

# Load environment variables
load_dotenv()

# Configure Gemini with API key from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model Initialization
model = genai.GenerativeModel("models/gemini-2.0-flash")

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return None

# Chunking
def chunk_text(text, max_tokens=1500):
    words = text.split()
    chunks = []
    current_chunk = []
    total_words = 0

    for word in words:
        current_chunk.append(word)
        total_words += 1
        if total_words >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            total_words = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Extract valid JSON blocks
def extract_json_blocks(text):
    json_blocks = []

    # Remove ```json and ``` markers
    cleaned_text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"```", "", cleaned_text).strip()

    # Extract possible JSON objects/lists
    possible_jsons = re.findall(r'(\{.*?\}|\[.*?\])', cleaned_text, re.DOTALL)

    for block in possible_jsons:
        try:
            parsed = json.loads(block)
            json_blocks.append(parsed)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(block)
                json_blocks.append(parsed)
            except Exception:
                continue

    return json_blocks

# Prompt Builder
def build_prompt(text, q_type, number, subject, tone, marks=None, response_json="{}"):
    instruction = f"""
    Text: {text}
    
    Based strictly on the language and style used in the above text, create {number} questions for {subject} students in a {tone.lower()} tone.
    Do not translate or change the language — use the same language as in the provided text.
    """

    if q_type == "MCQ":
        instruction += f"""
        Format: Multiple Choice Questions (MCQs).
        Each question should have 4 options with only one correct answer.
        Provide the output in this JSON format:
        {response_json}
        """
    elif q_type == "MSQ":
        instruction += f"""
        Format: Multiple Select Questions (MSQs).
        Each question should have multiple correct answers among 4 or more options.
        Provide the output in this JSON format:
        {response_json}
        """
    else:
        instruction += f"""
        Format: Descriptive questions of {marks} marks each.
        Include detailed answers based only on the text.
        Output in this JSON format:
        {response_json}
        """

    return instruction.strip()

# Evaluation Prompt
def evaluate_prompt(quiz, subject):
    return f"""
    Evaluate the following quiz for {subject} students:
    {quiz}
    Provide feedback on quality, difficulty, clarity, and possible improvements.
    """

# Streamlit UI
st.title("📘 PDF Question Bank Generator (Gemini)")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
subject = st.text_input("Subject")
tone = st.selectbox("Select Tone", ["Simple", "Medium", "Technical"])

question_types = st.multiselect(
    "Select Question Types to Generate",
    ["MCQ", "MSQ", "2-Mark", "5-Mark"],
    default=["MCQ"]
)

num_questions = {}
for q_type in question_types:
    num_questions[q_type] = st.number_input(f"Number of {q_type} Questions", min_value=1, max_value=20, value=5)

download_format = st.selectbox("Choose download format", ["CSV", "JSON"], index=0)

if st.button("Generate Questions") and pdf_file and subject:
    with st.spinner("Generating questions... Please wait."):

        file_text = extract_text_from_pdf(pdf_file)
        if not file_text:
            st.error("No text could be extracted from the uploaded PDF. Please upload a valid file.")
        else:
            chunks = chunk_text(file_text, max_tokens=1000)
            results = {}
            csv_data = []

            for q_type in question_types:
                if q_type in ["MCQ", "MSQ"]:
                    response_json = '{"question": "", "options": ["", "", "", ""], "answer": ""}'
                else:
                    response_json = '{"question": "", "answer": ""}'

                marks = int(q_type.split("-")[0]) if "-Mark" in q_type else None

                chunk_responses = []
                for chunk in chunks:
                    try:
                        prompt = build_prompt(chunk, q_type, num_questions[q_type], subject, tone, marks, response_json)
                        response = model.generate_content(prompt)
                        chunk_responses.append(response.text)
                    except Exception as e:
                        chunk_responses.append(f"Error generating for chunk: {e}")

                combined_output = "\n\n".join(chunk_responses)
                results[q_type] = combined_output

            for q_type, quiz in results.items():
                st.subheader(f"{q_type} Questions")
                st.text_area(f"Raw {q_type} Output", quiz, height=200)

                json_blocks = extract_json_blocks(quiz)
                if json_blocks:
                    formatted_output = json.dumps(json_blocks, indent=2, ensure_ascii=False)
                    st.text_area(f"Formatted {q_type} JSON", formatted_output, height=300)
                    csv_data.append({"Type": q_type, "Questions": formatted_output})
                else:
                    st.warning(f"No valid JSON found in {q_type} output.")

                try:
                    eval_prompt = evaluate_prompt(quiz, subject)
                    eval_response = model.generate_content(eval_prompt)
                    st.text_area(f"Evaluation of {q_type}s", eval_response.text, height=200)
                except Exception as e:
                    st.warning(f"Evaluation failed: {e}")

            if csv_data:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_dir = "saved_outputs"
                pathlib.Path(output_dir).mkdir(exist_ok=True)

                if download_format == "CSV":
                    df = pd.DataFrame(csv_data)
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)

                    st.download_button(
                        label="📥 Download Questions as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"generated_questions_{timestamp}.csv",
                        mime="text/csv"
                    )

                    local_filename = f"{output_dir}/questions_{timestamp}.csv"
                    with open(local_filename, "w", encoding="utf-8") as f:
                        f.write(csv_buffer.getvalue())

                elif download_format == "JSON":
                    full_json = {}
                    for item in csv_data:
                        q_type = item["Type"]
                        try:
                            full_json[q_type] = json.loads(item["Questions"])
                        except:
                            full_json[q_type] = item["Questions"]

                    json_data = json.dumps(full_json, indent=2, ensure_ascii=False)

                    st.download_button(
                        label="📥 Download Questions as JSON",
                        data=json_data,
                        file_name=f"generated_questions_{timestamp}.json",
                        mime="application/json"
                    )

                    local_filename = f"{output_dir}/questions_{timestamp}.json"
                    with open(local_filename, "w", encoding="utf-8") as f:
                        f.write(json_data)

                st.success(f"All questions generated and saved locally as: {local_filename}")


# Main Entry
if __name__ == "__main__":
    import sys
    import subprocess

    # Automatically run Streamlit server with network settings
    port = 8501  # Change if needed
    cmd = f"streamlit run {sys.argv[0]} --server.address=0.0.0.0 --server.port={port}"
    subprocess.run(cmd, shell=True)
