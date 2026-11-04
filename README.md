# Data Science Toolbox: Assessed Sprint 1 (Supervised Prediction)

University of Bristol, Data Science Toolbox. Our group predicts whether a hotel booking will be cancelled
(`is_canceled`) from the [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
data (Antonio, de Almeida & Nunes, 2019). Each member builds their own model; we agreed a common test, a performance
metric, and a way to compare the models.

## Project Group

* Luke D: random forest classifier; research into how to evaluate the models (metrics, tests, simulations, the shared scoring tool)
* Thomas F: gradient boosting model (XGBoost), EDA and data cleaning
* Morgan L: logistic regression model 

Equity split: equal, a third each (33/33/33).

## Reading order and requirements

**Preparation.** Python 3.12 or later. `pip install -r requirements.txt` installs the pinned versions of everything the
notebooks and scripts use (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost` and Jupyter).
The logistic regression (item 6) is in R: it needs R (tested with 4.6.1) and the packages installed by
`install.packages(c("rmarkdown", "knitr", "readr", "dplyr", "ggplot2", "tidyr", "scales"))`.
Run notebooks from the `report/` folder. `Data/hotel_bookings.csv` (raw) and `Data/hotel_bookings_clean.csv` (cleaned,
with the train/validation/test split) are included, so nothing needs downloading.

All report content is in `report/`, to be read in this order:

1. `report/01-Introduction.ipynb`: the question, the data, exploratory analysis.
2. `report/02-Preprocessing.ipynb`: removing columns recorded after the booking, cleaning, and the random
   train / validation / test split (70/15/15, stratified by the target, identical rows kept together). Saves `Data/hotel_bookings_clean.csv`.
3. `report/03-Performance_metrics.pdf`: the performance metric we agreed on (the Brier score) and why, how standardising
   the evaluation trades off real-world generalisability, and what "best" means. A short version of the research note in
   `LukeD/metrics/`; its figures come from `LukeD/metrics/simulations.py`.
4. `report/04-XGBoost.ipynb`: Thomas's gradient boosting model (about 7 minutes to run). Saves `ThomasF/test_preds.csv`.
5. `report/05-Random_forest.ipynb`: Luke's random forest, tuned and recalibrated on the validation set (about 13 minutes
   to run). Saves `LukeD/test_preds.csv`.
6. `report/06-Logistic_regression.pdf`, the output of `MorganL/06-Logistic_regression.Rmd`: Morgan's logistic
   regression, including model selection, splines, interactions, cross-validation and sensitivity analysis. To run it
   again (about 2 minutes; it writes a Word document), from the `MorganL/` folder: `Rscript -e 'rmarkdown::render("06-Logistic_regression.Rmd")'`.
   Its test predictions, `MorganL/test_preds.csv`, were saved from Morgan's working file `MorganL/v1.Rmd`; the report's
   final model reproduces them exactly.
7. `report/07-Conclusion.ipynb`: the comparison of the three models on the test set with the agreed metric, robustness
   checks, our conclusions and their limitations (written by Thomas). It reads the scoring output in `report/eval/`,
   made by the command in the next section.

## How we test our models

Every model outputs a probability of cancellation for the same 17,903 test bookings
(`split == "test"`), saved as `<Name>/test_preds.csv` with columns `id,prediction`, where `id` is the 0-based row number
in `Data/hotel_bookings_clean.csv`. Models are fitted on the training rows
(83,324) and tuned on the validation rows (17,331); the test rows are never used
for fitting, tuning or choosing a threshold. The primary metric is the Brier score, reported with a paired 95% interval
alongside log loss, ROC AUC, calibration, and threshold and cost-based metrics. "Best" requires a significant Brier
difference, no reversal on the Murphy (expected cost) diagram, and the same order in both hotels.

The full rules, including how the answer-key file `Data/test_truth.csv` is created, are in sections 10
and 11 of `LukeD/metrics/evaluation_metrics.md`. Once everyone has saved their predictions, run from the repository's
top folder:

```bash
python LukeD/metrics/evaluate.py --truth Data/test_truth.csv \
    --pred RF_Luke=LukeD/test_preds.csv --pred XGB_Thomas=ThomasF/test_preds.csv --pred Logit_Morgan=MorganL/test_preds.csv \
    --reference-rate 0.373 --group hotel --cluster block --n-boot 2000 --outdir report/eval
```

## Evaluation work

All in `LukeD/metrics/`:

* `evaluation_metrics.md` (also as `evaluation_metrics.pdf`): the research note with the mathematics, simulations and
  references behind every metric (the summary at its start is a two-minute read).
* `evaluate.py`: the shared scoring tool.
* `simulations.py`: reproduces every figure and number in the note
  (`python LukeD/metrics/simulations.py --data Data/hotel_bookings_clean.csv`, about five minutes), writing
  `sim_results.json` and the figures in `figures/`.
* `test_evaluate.py`: checks of the scoring tool against scikit-learn and SciPy (`python LukeD/metrics/test_evaluate.py`).

Code provenance: the scripts in `LukeD/metrics/` and the random forest notebook (`LukeD/05-Random_forest.ipynb`, copied
to `report/`) were drafted with the help of Claude (Anthropic); the scripts are tested by `test_evaluate.py`.


## Results and conclusions

All three models were evaluated on the same 17,903 held-out test bookings, using the Brier score as our primary metric. Lower scores indicate more accurate predicted cancellation probabilities.

| Model | Test Brier score |
|---|---:|
| XGBoost (Thomas) | 0.10142 |
| Random Forest (Luke) | 0.10209 |
| Logistic Regression (Morgan) | 0.12246 |

Both tree-based models outperformed logistic regression on the test set. XGBoost achieved the lowest Brier score, although its advantage over Random Forest was small and not statistically significant based on our paired comparison. Logistic regression was less accurate but provided a more interpretable model for examining relationships between booking characteristics and cancellation risk.

One limitation is that our random data split does not necessarily reflect how the models would perform on future bookings. A chronological split would be a useful extension to assess generalisation over time.

The full comparison, robustness checks and limitations are in `report/07-Conclusion.ipynb`.


## Evidence

Our individual working is shown in our own directories:

* `LukeD/`
* `ThomasF/`
* `MorganL/`
