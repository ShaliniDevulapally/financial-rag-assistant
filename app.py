import streamlit as st
from rag_pipeline import answer_question

st.set_page_config(page_title="JPMorgan Financial RAG Assistant", page_icon="📊", layout="wide")
st.title("📊 Financial Document AI Assistant")
st.caption("Grounded Q&A over JPMorgan Chase's 2025 Annual Report")
st.info("Ask a question about the indexed annual report. Answers are generated from retrieved document passages.")

question = st.text_area(
    "Your question",
    placeholder="Example: What were JPMorgan Chase's major sources of revenue in 2025?",
    height=100,
)
top_k = st.slider("Retrieved passages", 3, 8, 5)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the annual report and generating an answer..."):
                answer, sources = answer_question(question.strip(), top_k)
            st.subheader("Answer")
            st.markdown(answer)
            if sources:
                st.subheader("Retrieved sources")
                for i, source in enumerate(sources, 1):
                    score = source.get("score")
                    suffix = f" · Search score: {score:.3f}" if isinstance(score, (int, float)) else ""
                    with st.expander(f"Source {i} · Chunk {source['id']}{suffix}"):
                        st.write(source["content"])
        except Exception as exc:
            st.error("The assistant could not complete the request. Check the Azure environment variables.")
            st.exception(exc)

st.divider()
st.caption("Informational/educational use only. Not investment, legal, tax, or financial advice.")
