import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel
from torch import nn


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
st.markdown("Checks whether a case note meets SOAPIE documentation standards.")

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


try:
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    case_note = st.text_area("Paste Case Note Summary:", height=250)

    if st.button("Check Note"):
        if case_note.strip():
            inputs = tokenizer(
                case_note,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                out = model(**inputs)

            probs = torch.softmax(out["logits"], dim=-1)[0]
            prediction = torch.argmax(probs).item()
            raw_confidence = float(torch.max(probs))

            # Cap displayed confidence to avoid unrealistic 100%
            display_confidence = min(raw_confidence, 0.95)

            # Your model appears to use:
            # 0 = Good
            # 1 = Incomplete
            label = "Good" if prediction == 0 else "Incomplete"

            col1, col2 = st.columns(2)
            col1.metric("Classification", label)
            col2.metric("Model Confidence Estimate", f"{display_confidence:.1%}")

            if label == "Incomplete":
                st.error("🚩 Result: This note may need human review or additional SOAPIE details.")
            else:
                st.success("✅ Result: This note appears to meet SOAPIE documentation standards.")

            with st.expander("SOAPIE Checklist"):
                st.write("A strong case note usually includes:")
                st.write("- Subjective: what the veteran reports")
                st.write("- Objective: observable facts or actions")
                st.write("- Assessment: coordinator judgment")
                st.write("- Plan: next steps")
                st.write("- Intervention: support/resources provided")
                st.write("- Evaluation: response or follow-up outcome")

            with st.expander("Debug Info"):
                st.write("Raw prediction:", prediction)
                st.write("Raw model confidence:", raw_confidence)
                st.write("Displayed confidence:", display_confidence)
                st.write("Class probabilities:", probs.tolist())

        else:
            st.warning("Please enter note text.")

except Exception as e:
    st.error(
        f"Model loading failed: {e}. "
        "Check that the Hugging Face model path is correct and the repo is public."
    )
