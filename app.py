import os
import streamlit as st
import requests
import time
import numpy as np
import pandas as pd

# When running in Docker, the frontend should talk to the api service name
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api")

st.set_page_config(page_title="Research Paper RAG System", layout="wide")

st.sidebar.title("🔬 Research Paper RAG System")
st.sidebar.info(
    "Upload papers, ask questions, review history, and explore popular research topics. "
    "Powered by FastAPI, Qdrant, and LLMs."
)
top_k = st.sidebar.slider("top_k (Results per query)", 1, 10, 5)
n_clusters = st.sidebar.slider("Topic Clusters (Analytics)", 2, 20, 10)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Tech stack:**  
[Streamlit](https://streamlit.io/) · [FastAPI](https://fastapi.tiangolo.com/) · [Qdrant](https://qdrant.tech/)  
[Ollama](https://ollama.ai/) / [DeepSeek](https://deepseek.com/)
""")

# Use literal emoji here for tab names!
tabs = st.tabs(["📄 Upload", "❓ Query", "🕑 History", "📊 Analytics"])


def show_notification(msg, type="info"):
    if type == "info":
        st.info(msg)
    elif type == "success":
        st.success(msg)
    elif type == "error":
        st.error(msg)
    elif type == "warning":
        st.warning(msg)


# === UPLOAD TAB ===
with tabs[0]:
    st.header("Upload Research Papers (PDF)")
    uploaded_files = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploads")
    if st.button("Upload Selected Files") and uploaded_files:
        with st.spinner("Uploading papers..."):
            for file in uploaded_files:
                try:
                    files = {"files": (file.name, file, "application/pdf")}
                    resp = requests.post(f"{API_BASE}/papers/upload", files=files)
                    if resp.status_code == 201:
                        show_notification(f"Uploaded: {file.name}", "success")
                    else:
                        show_notification(f"Failed to upload {file.name}: {resp.text}", "error")
                except Exception as e:
                    show_notification(f"Error uploading {file.name}: {e}", "error")
            time.sleep(0.5)

    st.subheader("Your Papers")
    resp = requests.get(f"{API_BASE}/papers")
    if resp.status_code == 200:
        papers = resp.json()
        for p in papers:
            label = f"📄 {p['title']} ({p['year'] or ''})"
            with st.expander(label, expanded=False):
                st.write(f"**Authors:** {p['authors']}")
                st.write(f"**Filename:** {p['filename']}")
                st.write(f"**Uploaded at:** {p['uploaded_at']}")
                if st.button(f"🗑️ Delete '{p['title']}'", key=f"del_{p['id']}"):
                    del_resp = requests.delete(f"{API_BASE}/papers/{p['id']}")
                    if del_resp.status_code == 204:
                        show_notification("Paper deleted.", "success")
                    else:
                        show_notification("Delete failed!", "error")
    else:
        show_notification("Could not fetch papers.", "error")


# === QUERY TAB ===
with tabs[1]:
    st.header("Ask a Question")
    query = st.text_input("Enter your research question...")
    resp = requests.get(f"{API_BASE}/papers")
    paper_opts = []
    paper_id_map = {}
    if resp.status_code == 200:
        for p in resp.json():
            opt = f"{p['title']} ({p['year'] or ''})"
            paper_opts.append(opt)
            paper_id_map[opt] = p['id']
    selected_papers = st.multiselect("Filter by papers (optional)", paper_opts)
    paper_ids = [paper_id_map[opt] for opt in selected_papers]

    if st.button("Ask Question") and query.strip():
        # Build query params
        params = {
            "query": query.strip(),
            "top_k": top_k,
        }
        # For multiple paper_ids, must do &paper_ids=...&paper_ids=...
        paper_id_params = "".join([f"&paper_ids={pid}" for pid in paper_ids]) if paper_ids else ""
        url = f"{API_BASE}/query?query={params['query']}&top_k={params['top_k']}{paper_id_params}"
        with st.spinner("Getting answer..."):
            try:
                res = requests.post(url)
                data = res.json()
                if res.status_code == 200:
                    st.success("Answer:")
                    st.markdown(f"<div style='background-color:#eef6fa;padding:1em;border-radius:10px;'>"
                                f"<b>{data['answer']}</b></div>", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {data.get('confidence', 0):.2f}")
                    for c in data.get("citations", []):
                        st.info(
                            f"Paper: {c['paper_title']} | Section: {c['section']} | "
                            f"Page: {c['page']} | Score: {c['relevance_score']:.2f}"
                        )
                else:
                    show_notification(f"Backend error: {data.get('detail', str(data))}", "error")
            except Exception as e:
                show_notification(f"Error: {e}", "error")


# === HISTORY TAB ===
with tabs[2]:
    st.header("Query History")
    resp = requests.get(f"{API_BASE}/query/history")
    if resp.status_code == 200:
        queries = resp.json()
        for q in queries:
            with st.container():
                st.markdown(f"**Q:** {q['text']}\n\n**A:** {q.get('answer', '')[:90]}{'...' if len(q.get('answer', '')) > 90 else ''}")
                st.write(f"At {q['created_at']} | Response: {q.get('response_time_ms',0):.0f}ms")
                rating = q.get("user_rating") or 0
                btn_cols = st.columns(5)
                for i in range(1, 6):
                    if btn_cols[i-1].button(f"{'⭐' if i<=rating else '☆'}", key=f"star_{q['id']}_{i}"):
                        post = requests.post(f"{API_BASE}/query/history/{q['id']}/rate?rating={i}")
                        if post.status_code == 204:
                            show_notification("Rating updated!", "success")
                        else:
                            show_notification("Error updating rating.", "error")
    else:
        show_notification("Failed to load history.", "error")


# === ANALYTICS TAB ===
with tabs[3]:
    st.header("Popular Topics (Semantic Clusters)")
    resp = requests.get(f"{API_BASE}/query/analytics/popular?n_clusters={n_clusters}")
    if resp.status_code == 200:
        data = resp.json()
        topic_strings = [d["topic_exemplar"] for d in data]
        df = pd.DataFrame({
            "Topic Exemplar": topic_strings,
            "Count": [d["count"] for d in data],
            "Examples": [", ".join(d["example_queries"]) for d in data]
        })
        st.bar_chart(df, x="Topic Exemplar", y="Count")
        st.dataframe(df, use_container_width=True)
    else:
        show_notification("Could not fetch analytics.", "error")

st.markdown(
    """
    <hr><div style='text-align:center;color:grey;font-size:90%'>Built with Streamlit · FastAPI · Qdrant · Ollama/DeepSeek</div>
    """,
    unsafe_allow_html=True
)
