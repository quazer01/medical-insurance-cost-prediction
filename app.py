"""
Demo: Medical Insurance Cost Predictor (CSC340 AI project)

Enter a person's details and get predicted medical insurance charges from
the trained XGBoost model, plus the 3 factors that pushed the prediction up or down
(SHAP values).

How to run:
    1. python train_model.py      (creates model.joblib)
    2. python app.py              (opens the demo at http://127.0.0.1:7860)
"""

import gradio as gr
import joblib
import pandas as pd
import shap

model = joblib.load("model.joblib")
prep = model.named_steps["prep"]
explainer = shap.TreeExplainer(model.named_steps["model"])
feature_names = [n.split("__")[-1] for n in prep.get_feature_names_out()]

LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "children": "Number of children",
    "smoker_yes": "Smoking status",
    "sex_male": "Sex",
    "region_northeast": "Region (northeast)",
    "region_northwest": "Region (northwest)",
    "region_southeast": "Region (southeast)",
    "region_southwest": "Region (southwest)",
}


def predict(age, sex, bmi, children, smoker, region):
    person = pd.DataFrame(
        [{"age": age, "bmi": bmi, "children": int(children), "sex": sex, "smoker": smoker, "region": region}]
    )
    cost = float(model.predict(person)[0])

    shap_values = explainer(pd.DataFrame(prep.transform(person), columns=feature_names)).values[0]
    top = sorted(zip(feature_names, shap_values), key=lambda p: abs(p[1]), reverse=True)[:3]
    lines = [
        f"- **{LABELS.get(name, name)}** {'raised' if value > 0 else 'lowered'} the estimate by ${abs(value):,.0f}"
        for name, value in top
    ]

    return (
        f"## Predicted medical cost: ${cost:,.0f}\n\n"
        f"### Top 3 factors (compared with the average prediction of ${float(explainer.expected_value):,.0f})\n" + "\n".join(lines) + "\n\n"
        "*This is an estimate from a model trained on simulated data, not a price quote.*"
    )


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 64, value=35, step=1, label="Age"),
        gr.Radio(["female", "male"], value="female", label="Sex"),
        gr.Slider(15, 55, value=28, step=0.1, label="BMI"),
        gr.Slider(0, 5, value=1, step=1, label="Number of children"),
        gr.Radio(["no", "yes"], value="no", label="Smoker"),
        gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], value="southeast", label="Region (US)"),
    ],
    outputs=gr.Markdown(),
    title="Medical Insurance Cost Predictor",
    description=(
        "CSC340 Artificial Intelligence project. XGBoost model trained on the "
        "Medical Cost Personal Datasets (Kaggle). Test RMSE $4,207, 29.4% lower than linear regression."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
