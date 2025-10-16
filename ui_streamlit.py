# ui_streamlit.py
import streamlit as st
import requests, json

st.set_page_config(page_title="RIA Assistant — Local RAG", layout="centered")
st.title("RIA Assistant — Local RAG Prototype")

question = st.text_area(
    "Enter your regulatory question",
    height=120,
    placeholder="e.g., List donor screening requirements for ocular/cornea tissue and any deferral criteria."
)
top_k = st.slider("Top K passages", 3, 12, 6)

mode = st.radio("Answer mode", ["Exact (clauses + citations)", "Concise", "Structured JSON"], index=2)

col1, col2, col3 = st.columns([1,1,1])





if col1.button("Ask", type="primary"):
    with st.spinner("Thinking..."):
        if mode.startswith("Exact"):
            resp = requests.post("http://127.0.0.1:8000/ask_exact", data={"question": question, "top_k": top_k})
            st.subheader("Answer")
            st.write(resp.json()["text"])
        elif mode == "Concise":
            resp = requests.post("http://127.0.0.1:8000/ask_concise", data={"question": question, "top_k": top_k})
            st.subheader("Answer")
            st.write(resp.json()["text"])
        else:
            resp = requests.post("http://127.0.0.1:8000/ask", data={"question": question, "top_k": top_k})
            st.subheader("Structured Answer")
            st.json(resp.json())
    st.caption("Tip: Concise/Exact are short, deterministic answers with inline citations. Structured JSON is the full schema for audit & PDF.")

if col2.button("Generate PDF"):
    with st.spinner("Generating report..."):
        resp = requests.post("http://127.0.0.1:8000/report", data={"question": question, "top_k": top_k})
        st.download_button("Download RIA Report", data=resp.content, file_name="ria_report.pdf", mime="application/pdf")

if col3.button("Rescan & Update Index"):
    with st.spinner("Re-indexing..."):
        r = requests.post("http://127.0.0.1:8000/reindex")
        st.success(r.json())
