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


st.set_page_config(page_title="AFRCC Note Grader", layout="wide")
st.title("📝 AFRCC Case Note Quality Grader")
st.markdown("Evaluation based on SOAPIE standards and AFRCC training KPIs.")

MODEL_PATH = "Danube1/Capstone"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = MultiTaskDistilBERT.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    case_note = st.text_area("Paste Case Note Summary:", height=250)

    if st.button("Grade Note"):
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
            confidence = float(torch.max(probs))

            # Your model appears to use:
            # 0 = Good
            # 1 = Incomplete
            label = "Good" if prediction == 0 else "Incomplete"

            col1, col2 = st.columns(2)
            col1.metric("Classification", label)
            col2.metric("Confidence", f"{confidence:.1%}")

            if label == "Incomplete":
                st.error("🚩 Result: Fails quality check. Recommend human review.")
            else:
                st.success("✅ Result: Meets AFRCC documentation standards.")

            with st.expander("Debug Info"):
                st.write("Raw prediction:", prediction)
                st.write("Class probabilities:", probs.tolist())

        else:
            st.warning("Please enter note text.")

except Exception as e:
    st.error(
        f"Model loading failed: {e}. "
        "Check that the Hugging Face model path is correct and the repo is public."
    )
