# Visit with Us - Wellness Tourism Package Prediction (MLOps)

Predicts whether a customer will buy the new **Wellness Tourism Package** before
the sales team contacts them, served through an automated MLOps pipeline.

## Pipeline
`push to main` -> **GitHub Actions** (`.github/workflows/pipeline.yml`)

| Job | Script | Output |
|---|---|---|
| register-dataset | `tourism_project/model_building/data_register.py` | schema validation + summary |
| data-prep | `tourism_project/model_building/prep.py` | `Xtrain/Xtest/ytrain/ytest.csv` (workflow artifact) |
| model-training | `tourism_project/model_building/train.py` | tuned XGBoost, MLflow runs, model committed to `tourism_project/deployment/` |

Streamlit Community Cloud redeploys `tourism_project/deployment/app.py` on every new commit.

## Repository structure
```
.github/workflows/pipeline.yml      CI/CD pipeline
tourism_project/
  data/tourism.csv                  registered dataset
  model_building/                   data_register.py, prep.py, train.py
  deployment/                       app.py, requirements.txt, trained model
  requirements.txt                  pipeline dependencies
```
