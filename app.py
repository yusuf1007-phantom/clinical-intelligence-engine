import streamlit as st
from clinical_intelligence.services.engine import analyze

st.set_page_config(page_title="Clinical Intelligence Engine", page_icon="🧬", layout="wide")
st.title("🧬 Clinical Intelligence Engine")
st.caption("Biomedical NLP research rebuilt as a deployable AI application")

text = st.text_area(
    "Biomedical / clinical text",
    "Patients with type 2 diabetes received metformin for 12 weeks. HbA1c decreased significantly, while mild nausea was reported.",
    height=180,
)

if st.button("Analyze", type="primary"):
    with st.spinner("Running biomedical NLP pipeline..."):
        result = analyze(text)
    left, right = st.columns(2)
    with left:
        st.subheader("Scientific structure")
        if result["structure"]:
            st.metric(result["structure"]["label"], f'{result["structure"]["score"]:.2%} confidence')
        else:
            st.info("Optional fine-tuned structure classifier is not configured. Biomedical NER remains active.")
    with right:
        st.subheader("Biomedical entities")
        if result["entities"]:
            st.dataframe(result["entities"], use_container_width=True, hide_index=True)
        else:
            st.write("No entities detected.")
    with st.expander("Machine-readable JSON"):
        st.json(result)

st.divider()
st.caption("Research/portfolio demonstration only — not for clinical decision-making.")
