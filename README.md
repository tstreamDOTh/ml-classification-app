# Online Shoppers Purchasing Intention: ML Classification with Streamlit

BITS WILP, Machine Learning, Assignment 2
Name: Thiyagaraj T
BITS ID: 2025AC05747

## a. Problem statement

Online stores get thousands of visitors but only a small fraction of sessions
end in a purchase. If we can predict, from session behaviour alone, whether a
visitor is likely to buy, the store can act in real time (show an offer,
simplify checkout, prioritise support chat). The task here is binary
classification: given 17 features describing a browsing session, predict
whether the session ends with a purchase (Revenue = 1) or not (Revenue = 0).

Five classifiers are trained on the same dataset and compared using six
metrics. The best model is served through a Streamlit web app.

## b. Dataset description

Dataset: Online Shoppers Purchasing Intention, from the UCI Machine Learning
Repository (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

- Instances: 12,330 sessions (each row is one user session over a year)
- Features: 17 (10 numeric, 7 categorical)
- Target: Revenue (TRUE if the session ended in a purchase). About 15.5% of
  sessions are positive, so the classes are imbalanced.

Feature groups:

- Page visit counts and time spent: Administrative, Administrative_Duration,
  Informational, Informational_Duration, ProductRelated, ProductRelated_Duration
- Google Analytics signals: BounceRates, ExitRates, PageValues
- Session context: SpecialDay (closeness to a special day like Valentine's),
  Month, Weekend
- Visitor and technology info: OperatingSystems, Browser, Region, TrafficType,
  VisitorType (new, returning or other)

Preprocessing: Month and VisitorType are one-hot encoded, boolean columns are
converted to 0/1, and numeric features are standardised. The data is split
80/20 with stratification (random_state 42). The 20% test split (2,466 rows)
is saved as test_data.csv and is what the Streamlit app scores by default.

## c. GitHub repository link

https://github.com/tstreamDOTh/ml-classification-app

Live Streamlit app: https://STREAMLIT_APP_URL

Repository layout:

```
ml-classification-app/
|-- app.py                          Streamlit app
|-- requirements.txt
|-- README.md
|-- test_data.csv                   held-out 20% test split
|-- online_shoppers_intention.csv   full dataset
|-- model/
    |-- train_models.py             trains and saves all 5 pipelines
    |-- *.pkl                       fitted pipelines, one per model
    |-- metrics.json                test-set metrics for all models
```

To reproduce locally:

```
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## d. Models used

All five models were trained on the same 80% training split and evaluated on
the same 20% test split. Each model is a scikit-learn pipeline that includes
the preprocessing, so the app can score raw CSV rows directly.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8812 | 0.8877 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8556 | 0.7307 | 0.5330 | 0.5497 | 0.5412 | 0.4557 |
| kNN | 0.8739 | 0.7993 | 0.6715 | 0.3639 | 0.4720 | 0.4322 |
| Naive Bayes (Gaussian) | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Random Forest (Ensemble) | 0.8998 | 0.9211 | 0.7368 | 0.5497 | 0.6297 | 0.5814 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good accuracy (0.88) and the second best AUC (0.89), but recall is poor (0.36). The classes are imbalanced and a linear boundary favours the majority "no purchase" class, so it misses nearly two thirds of the actual buyers. Precision is high (0.74): when it does predict a purchase it is usually right. |
| Decision Tree | The single tree overfits the training data and generalises worst of all on ranking quality (AUC 0.73). Accuracy drops to 0.86. Its recall (0.55) is better than logistic regression because axis-aligned splits on PageValues capture buyer behaviour, but the predictions are noisy, which shows in the modest precision (0.53). |
| kNN | Behaves much like logistic regression: decent accuracy (0.87) but low recall (0.36). With 15.5% positives, the 7 nearest neighbours of a buyer are often non-buyers, so minority-class points get outvoted. Distance in the one-hot encoded space also dilutes the signal. AUC (0.80) is clearly below the linear model. |
| Naive Bayes (Gaussian) | The opposite trade-off: highest recall of all (0.79) but the worst precision (0.29) and accuracy (0.67). The conditional independence assumption is badly violated (the duration and count features are strongly correlated), so it over-predicts the purchase class. Useful only if missing a buyer is far more costly than a false alarm. |
| Random Forest (Ensemble) | Best on almost every metric: accuracy 0.90, AUC 0.92, F1 0.63 and MCC 0.58. Bagging over 200 trees removes the variance that hurt the single decision tree while keeping its ability to model non-linear feature interactions (for example PageValues combined with ExitRates and Month). Balanced precision (0.74) and recall (0.55). |
| Overall Winner for your dataset? | Random Forest. It has the best AUC, F1 and MCC, and MCC is the most reliable single number on an imbalanced dataset like this one. The ensemble handles the non-linear, mixed-type features better than any single model here. |

## Streamlit app features

- Upload option for test data (CSV). If nothing is uploaded the bundled
  test_data.csv is used.
- Dropdown to select any of the 5 trained models.
- Six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for the
  selected model on the uploaded data.
- Confusion matrix heatmap and the full classification report.
- A comparison table showing all 5 models scored on the same uploaded data.
