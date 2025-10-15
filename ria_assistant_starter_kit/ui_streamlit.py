import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"
TIMEOUT_S = 120

st.set_page_config(page_title="RIA Assistant — Local RAG", layout="centered")
st.title("RIA Assistant — Local RAG Prototype")

question = st.text_area(
    "Enter your regulatory question",
    height=120,
    placeholder="e.g., List donor screening requirements for ocular/cornea tissue and any deferral criteria."
)
top_k = st.slider("Top K passages", 3, 12, 6)

mode = st.radio(
    "Answer mode",
    ["Exact (clauses + citations)", "Concise", "Structured JSON"],
    index=0,
    help="Exact: bullet points with doc+page citations (deterministic, no LLM).\n"
         "Concise: 1-2 sentence summary with inline citations.\n"
         "Structured JSON: full object used for PDF."
)

col1, col2 = st.columns(2)

def post_text(path, data, expect_json=False):
    try:
        r = requests.post(f"{API_BASE}{path}", data=data, timeout=TIMEOUT_S)
        r.raise_for_status()
        return (r.json() if expect_json else r.text), None
    except requests.exceptions.JSONDecodeError:
        return None, "Backend returned non-JSON. Check server logs."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

if col1.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            if mode.startswith("Exact"):
                text, err = post_text("/ask_exact", {"question": question, "top_k": top_k})
                if err: st.error(err)
                else:
                    st.subheader("Answer")
                    st.write(text)
            elif mode == "Concise":
                text, err = post_text("/ask_concise", {"question": question, "top_k": top_k})
                if err: st.error(err)
                else:
                    st.subheader("Answer")
                    st.write(text)
            else:
                data, err = post_text("/ask", {"question": question, "top_k": top_k}, expect_json=True)
                if err: st.error(err)
                else:
                    st.subheader("Structured Answer")
                    st.json(data)

if col2.button("Generate PDF", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating report..."):
            try:
                r = requests.post(f"{API_BASE}/report", data={"question": question, "top_k": top_k}, timeout=TIMEOUT_S)
                r.raise_for_status()
                st.download_button(
                    "Download RIA Report PDF",
                    data=r.content,
                    file_name="ria_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except requests.exceptions.RequestException as e:
                st.error(f"PDF generation failed: {e}")

st.caption("Exact mode shows clause text with document+page citations. Always keep human review in the loop.")
