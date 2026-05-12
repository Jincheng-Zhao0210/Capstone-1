import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel
from torch import nn
import re


class MultiTaskDistilBERT(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.dim, 2)
        self.dropout = nn.Dropout(0.1)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return {"logits": logits}


st.set_page_config(page_title="AFRCC Case Note Checker", layout="wide")

st.title("📝 AFRCC Case Note Completeness Checker")
st.markdown("AI-assisted SOAPIE completeness review for AFRCC-style case notes.")

MODEL_PATH = "Danube1/Capstone"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = MultiTaskDistilBERT.from_pretrained(
        MODEL_PATH,
        ignore_mismatched_sizes=True
    )
    model.eval()
    return tokenizer, model


def check_soapie_sections(text):
    sections = {
        "Subjective": r"\b(subjective|subject)\b",
        "Objective": r"\b(objective|observed|observation)\b",
        "Assessment": r"\b(assessment|assess)\b",
        "Plan": r"\b(plan|next step|follow[- ]?up)\b",
        "Intervention": r"\b(intervention|provided|support|resource)\b",
        "Evaluation": r"\b(evaluation|outcome|response)\b",
    }

    found = {}
    lower_text = text.lower()

    for section, pattern in sections.items():
        found[section] = bool(re.search(pattern, lower_text))

    score = sum(found.values())
    return found, score


try:
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    case_note = st.text_area("Paste Case Note Summary:", height=280)

    if st.button("Check Note"):
        if case_note.strip():
            found_sections, soapie_score = check_soapie_sections(case_note)

            inputs = tokenizer(
                case_note,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                out = model(**inputs)

            temperature = 5.0
            probs = torch.softmax(out["logits"] / temperature, dim=-1)[0]
            prediction = torch.argmax(probs).item()
            raw_confidence = float(torch.max(probs))

            model_label = "Good" if prediction == 0 else "Incomplete"

            # Rule-based override
            if soapie_score >= 5 and len(case_note.split()) >= 40:
                final_label = "Good"
            elif soapie_score <= 3 or len(case_note.split()) < 25:
                final_label = "Incomplete"
            else:
                final_label = model_label

            # More realistic confidence estimate
            rule_confidence = 0.55 + (soapie_score / 6) * 0.35
            final_confidence = (raw_confidence * 0.4) + (rule_confidence * 0.6)
            final_confidence = max(0.60, min(final_confidence, 0.93))

            col1, col2, col3 = st.columns(3)
            col1.metric("Final Result", final_label)
            col2.metric("Confidence Estimate", f"{final_confidence:.1%}")
            col3.metric("SOAPIE Sections Found", f"{soapie_score}/6")

            if final_label == "Incomplete":
                st.error("🚩 Result: This note may need human review or more SOAPIE details.")
            else:
                st.success("✅ Result: This note appears to meet SOAPIE documentation standards.")

            st.subheader("SOAPIE Checklist")

            for section, found in found_sections.items():
                if found:
                    st.write(f"✅ {section}")
                else:
                    st.write(f"❌ {section}")

            with st.expander("Recommendation"):
                if soapie_score < 6:
                    missing = [s for s, v in found_sections.items() if not v]
                    st.write("Consider adding:")
                    for m in missing:
                        st.write(f"- {m}")
                else:
                    st.write("All major SOAPIE sections are present.")

            with st.expander("Debug Info"):
                st.write("Model prediction:", model_label)
                st.write("Raw model confidence:", raw_confidence)
                st.write("Final confidence estimate:", final_confidence)
                st.write("Class probabilities:", probs.tolist())
                st.write("SOAPIE score:", soapie_score)

        else:
            st.warning("Please enter note text.")

except Exception as e:
    st.error(
        f"Model loading failed: {e}. "
        "Check that the Hugging Face model path is correct and the repo is public."
    )
