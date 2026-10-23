# Data Science Toolbox: Assessed Sprint 1 (Supervised Prediction)

University of Bristol, Data Science Toolbox. Our group predicts whether a hotel booking will be cancelled
(`is_canceled`) from the [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
data (Antonio, de Almeida & Nunes, 2019). Each member builds their own model; we agreed a common test, a performance
metric, and a way to compare the models.

## Project Group

* Luke D: random forest classifier; research into how to evaluate the models (metrics, tests, simulations, the shared scoring tool)
* Thomas F: gradient boosting model (XGBoost), EDA and data cleaning
* Morgan L: regression model (to be confirmed)

Equity split: to be agreed by the group and noted here before submission.

## Reading order and requirements

**Preparation.** Python 3.10 or later with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` and `seaborn`
(`pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter`). Run notebooks from the `report/` folder.
`Data/hotel_bookings.csv` (raw) and `Data/hotel_bookings_clean.csv` (cleaned, with the train/validation/test split) are
included, so nothing needs downloading.

All report content is in `report/`, to be read in this order:

1. `report/01-Introduction.ipynb`: the question, the data, exploratory analysis.
2. `report/02-Preprocessing.ipynb`: removing columns recorded after the booking, cleaning, and the chronological
   train / validation / test split. Saves `Data/hotel_bookings_clean.csv`.
3. `report/03-Evaluation.ipynb`: how we test and compare the models: the agreed metric, the scoring of all submissions
   with the same code, the question of whether the best model is truly best, and checks of how the scores generalise.
   Until the three real models are submitted it runs on clearly labelled stand-in models.
4. Model notebooks, one per member (to be added).

## How we test our models

Every model outputs a probability of cancellation for the same 40,509 test bookings (arrivals 1 January to 31 August 2017,
`split == "test"`), saved as `<Name>/test_preds.csv` with columns `id,prediction`. Models are fitted on the training rows
(July 2015 to September 2016) and tuned on the validation rows (October to December 2016); the test rows are never used
for fitting, tuning or choosing a threshold. The primary metric is the Brier score, reported with a paired 95% interval
alongside log loss, ROC AUC, calibration, and threshold and cost-based metrics. "Best" requires a significant Brier
difference, no reversal on the Murphy (expected cost) diagram, and the same order in both hotels.

The full rules and commands are in [`evaluation/README.md`](evaluation/README.md). In short:

```bash
python evaluation/make_test_truth.py          # once: writes Data/test_truth.csv
python evaluation/evaluate.py --truth Data/test_truth.csv \
    --pred RF_Luke=LukeD/test_preds.csv --pred XGB_Thomas=ThomasF/test_preds.csv --pred Logit_Morgan=MorganL/test_preds.csv \
    --reference-rate 0.360 --group hotel --cluster block --n-boot 2000 --outdir results
```

## Evaluation work

* `evaluation/`: the shared scoring tool `evaluate.py`, the answer-key script, and the rules for submissions.
* `LukeD/metrics/evaluation_metrics.md` (PDF: `LukeD/metrics/tex/evaluation_metrics.pdf`): the research note with the
  mathematics, simulations and references behind every metric (about 12,000 words; the summary at its start is a
  two-minute read).
* `LukeD/metrics/simulations.py`: reproduces every figure and number in the note
  (`python LukeD/metrics/simulations.py --data Data/hotel_bookings_clean.csv`, about five minutes).
* `LukeD/metrics/test_evaluate.py`: 13 checks of the scoring tool against scikit-learn and SciPy.

Code provenance: `evaluation/evaluate.py`, `evaluation/make_test_truth.py`, `report/03-Evaluation.ipynb` and the scripts in
`LukeD/metrics/` were drafted with the help of Claude (Anthropic) and are tested by `test_evaluate.py`.

## Evidence

Our individual working is shown in our own directories:

* `LukeD/`
* `ThomasF/`
* `MorganL/`
