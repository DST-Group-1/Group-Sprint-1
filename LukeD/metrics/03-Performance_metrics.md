# Performance Metrics

The brief asks us to agree on a performance metric and test it, to consider how a standard metric trades real-world generalisability for comparability, and to examine in what sense our best model is truly best. This section sets out the metric we agreed on, the Brier score, and why. We tested it before using it, by simulation and on our data. The derivations, the other metrics we considered and the full references are in the research note behind this section, [`LukeD/metrics/evaluation_metrics.pdf`](../LukeD/metrics/evaluation_metrics.pdf) (also as a PDF). Figures 1 to 4, and the numbers that don't come from our model notebooks, are produced by `python LukeD/metrics/simulations.py --data Data/hotel_bookings_clean.csv`, run from the repository's top folder.

## 1. What a hotel needs from a prediction

Two features of how a hotel would use a cancellation model decide how it should be judged.

* **Hotels act on probabilities, not labels.** The expected number of cancellations on a night is $\sum_i \hat p_i$ over that night's bookings, so an overbooking level set from the model is only right if its probabilities are **calibrated**: among bookings given probability $p$, a share $p$ cancel. A metric that looks only at labels (accuracy) or only at rankings (AUC) cannot see this.
* **The two mistakes cost different amounts.** Expecting a guest to cancel and reselling the room risks turning away ("walking") a guest who arrives; expecting a guest to arrive leaves an empty room if they cancel. If walking a guest costs $c_{\mathrm{FP}}$ and an empty room $c_{\mathrm{FN}}$, a hotel with calibrated probabilities minimises its expected cost by treating booking $i$ as a likely cancellation when $\hat p_i > \theta$, where $\theta=c_{\mathrm{FP}}/(c_{\mathrm{FP}}+c_{\mathrm{FN}})$ is its **cost ratio** (Elkan, 2001). Nothing in our data tells us a hotel's costs, so a fair comparison shouldn't depend on one particular $\theta$.

Every model therefore outputs a probability of cancellation for the same 17,903 test bookings, and we score the probabilities themselves.

## 2. The primary metric: the Brier score

The Brier score is the mean squared error of the predicted probabilities (Brier, 1950):

$$
\mathrm{BS}=\frac{1}{n}\sum_{i=1}^{n}(\hat p_i-y_i)^2 .
$$

We chose it for four reasons.

**It is minimised only by the true probabilities.** If a booking cancels with probability $q$ and a model reports $p$, then, using $Y^2=Y$,

$$
\mathbb{E}\,(p-Y)^2=(p-q)^2+q(1-q),
$$

which is smallest at $p=q$ and nowhere else. A score with this property is **strictly proper** (Gneiting & Raftery, 2007): a model cannot improve its expected score by reporting anything other than the probability it believes. Figure 1 shows two common alternatives that fail. The absolute error of a probability, $\lvert p-y\rvert$, is minimised by reporting 0 or 1, so it rewards over-confidence; accuracy is flat on each side of ½, so it cannot tell a careful forecast from a careless one. Log loss, which we also report, is strictly proper too.

![Expected score against the reported probability for four scoring rules](../LukeD/metrics/figures/fig1_propriety.png)

*Figure 1. Expected score of a forecast p for a booking whose true cancellation probability is q = 0.37, our overall cancellation rate. The Brier score and log loss are minimised exactly at p = q. Source: `simulations.py`, S1.*

**It is bounded.** Each booking adds at most $1/n$ to the score. Log loss, $-\frac{1}{n}\sum_i\big[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\big]$, is unbounded: software clips probabilities at about $2.2\times10^{-16}$, so one cancelled booking that was predicted with probability exactly 0 adds about $36/n$. Random forests produce such predictions: before recalibration, our forest predicted exactly 0 or 1 for 1,418 of the 17,331 validation bookings (`05-Random_forest`, Section 5.1). Under log loss, a handful of them could decide a comparison.

**It separates calibration from ranking.** The CORP decomposition (Dimitriadis, Gneiting & Jordan, 2021) splits the score exactly into three parts,

$$
\mathrm{BS}=\mathrm{MCB}-\mathrm{DSC}+\mathrm{UNC}.
$$

UNC $=\bar y(1-\bar y)$ depends only on the data. MCB $\ge 0$ is the part of the score that the best non-decreasing recalibration of the model (isotonic regression) would remove, and DSC $\ge 0$ measures how well the model separates high-risk from low-risk bookings. The decomposition tells us whether a model is behind because it ranks bookings worse, which is hard to fix, or only because its probabilities are mis-scaled, which recalibration on validation data can fix.

**It averages the hotel's costs over all cost ratios.** Scale the costs so that $c_{\mathrm{FP}}+c_{\mathrm{FN}}=1$, and let $S_\theta(p,y)$ be the cost of acting on probability $p$ at cost ratio $\theta$: $1-\theta$ for a cancellation that was not flagged, $\theta$ for a flagged booking whose guest arrives, and 0 otherwise. Every proper score is a weighted average of these costs over $\theta$ (Schervish, 1989; Ehm et al., 2016), and for the two scores we report

$$
(p-y)^2=2\int_0^1 S_\theta(p,y)\,d\theta,
\qquad
-y\log p-(1-y)\log(1-p)=\int_0^1 \frac{S_\theta(p,y)}{\theta(1-\theta)}\,d\theta .
$$

The Brier score weights every cost ratio equally. Log loss weights $\theta$ by $1/(\theta(1-\theta))$: 4 at $\theta=1/2$, but about 101 at $\theta=0.01$ or $0.99$. Accuracy looks only at $\theta=1/2$. For a hotel, cost ratios between about 0.3 and 0.8 are plausible (a walked guest costing between about 0.4 and 4 times an empty room), and over that range the even weighting of the Brier score is closer to the hotel's decision than the extreme-heavy weighting of log loss. This is the main reason we made the Brier score primary rather than log loss.

To make it easier to read, we also report it as a skill score, $\mathrm{BSS}=1-\mathrm{BS}/\mathrm{BS}_{\mathrm{ref}}$: the share of error removed relative to always predicting the training cancellation rate, 0.373.

**A caveat: no metric is neutral between our models.** A classification tree that splits on the Gini criterion greedily minimises the Brier score of its training data, so the Brier score is the criterion our random forest is grown on, while XGBoost and logistic regression minimise log loss (`05-Random_forest`, Section 2.1). We therefore always report log loss next to the Brier score, and use the Murphy diagram (Section 5.2) as the tie-breaker. The random forest notebook also tested the concern directly: a forest grown on entropy, which minimises log loss instead, scores practically the same under both metrics (`05-Random_forest`, Section 4.3).

## 3. What we report alongside it

| Metric | What it measures | Why it isn't the primary metric |
| --- | --- | --- |
| Log loss | The quality of the probabilities, like the Brier score | Unbounded, and dominated by extreme cost ratios |
| ROC AUC, with DeLong's test | Ranking: the probability that a random cancelled booking gets a higher score than a random kept one | Blind to calibration: dividing every probability by ten leaves it unchanged |
| Calibration intercept and slope (Cox, 1958), reliability diagram | Whether the probabilities can be taken at face value; a slope below 1 means over-confident | Says nothing about ranking: always predicting the base rate is perfectly calibrated |
| MCB and DSC | The calibration and ranking parts of the Brier score | Components of the score, not a ranking |
| Accuracy, balanced accuracy, F1 and MCC at ½ | Decisions at one threshold | Each fixes one cost ratio. For calibrated probabilities the F1-maximising threshold is $F_1^{\ast}/2$, so F1's implied costs depend on the model (Lipton, Elkan & Narayanaswamy, 2014) |
| Expected cost at $\theta=2/3$ | Decisions when a walked guest costs twice an empty room | Depends on an assumed cost ratio |
| Murphy diagram (Ehm et al., 2016) | Expected cost at every cost ratio | A curve, not a number; used as the tie-breaker |

Every threshold is fixed before the test set is seen, because choosing it to maximise a metric on the test set overstates performance.

## 4. Standardisation against generalisability

To compare models fairly we standardise: one test set, one split, one primary score and one threshold for the decision metrics. Each choice buys comparability at a price in real-world generalisability. Here are three of those prices, shown by simulation and on our data.

### 4.1 One test set, one cancellation rate

Most metrics change when only the cancellation rate changes, even if the model doesn't. Under **label shift** (cancelled and kept bookings look the same as before, but the share that cancels moves), a model's recall and false-positive rate at a fixed threshold don't change, so neither do its AUC and balanced accuracy. Accuracy, precision, F1, MCC and the proper scores do change, and the probabilities become miscalibrated. This is not hypothetical for us: the cancellation rate is 28.0% at the resort and 41.9% at the city hotel, and quarterly rates range from 31% to 44%. Figure 2 holds one model fixed and changes only the test set's cancellation rate.

![Eight metrics against the cancellation rate of the test set](../LukeD/metrics/figures/fig2_prevalence_shift.png)

*Figure 2. One fixed model (AUC 0.85, calibrated at the training rate 0.373, threshold 0.5) scored on test sets that differ only in their cancellation rate. Shaded: the range of rates in our data. Source: `simulations.py`, S2.*

| Cancellation rate of the test set | 0.280 (resort) | 0.368 (our test set) | 0.419 (city) |
| --- | --- | --- | --- |
| Recall, balanced accuracy, AUC | 0.648, 0.755, 0.850 | unchanged | unchanged |
| Precision | 0.645 | 0.731 | 0.771 |
| F1 | 0.646 | 0.687 | 0.704 |
| Accuracy | 0.802 | 0.783 | 0.772 |
| Brier score | 0.138 | 0.149 | 0.156 |
| Calibration-in-the-large | −0.43 | −0.02 | +0.19 |

Without any change to the model, its precision differs by 0.13 between the two hotels, and its accuracy *falls* as the cancellation rate rises. Scores computed on different hotels, periods or deduplicated data are not comparable, so every model must be scored on the identical test set. That standardisation hides how a model would behave on a different population: a model calibrated at the overall rate over-predicts cancellations at the resort and under-predicts them in the city. The proper scores penalise this, which is right, because a hotel using those probabilities would overbook by the wrong amount. We therefore also report the metrics for each hotel (`07-Conclusion` and `report/eval/`). We had also planned to train each model on one hotel and test it on the other, but did not run that check.

### 4.2 One random split: duplicates and booking events

A random split treats test bookings as independent of training bookings, and ours aren't quite. 28.9% of the rows are exact copies of another row, most plausibly several rooms booked together (`02-Preprocessing`, Section 4). Rooms of one group booking that differ in a detail, such as the room type or the price, are not identical, but they share a **booking event**: the same hotel, arrival date, lead time, length of stay, market segment, channel, agent, country, deposit type and customer type. To measure how much a split flatters a model, we scored a "memoriser" that knows nothing but a lookup table. For each test booking it returns the cancellation rate of the training rows that match it (identical in every feature except the arrival date or, in the third row, from the same booking event), and the overall training rate if there are none.

| Split | Test rows with a match in training | Memoriser AUC | Memoriser Brier skill |
| --- | --- | --- | --- |
| Plain random split (80/20) | 35.6% | 0.813 | 0.400 |
| Our split (identical rows kept together) | 2.8% | 0.491 | −0.047 |
| Our split, matching on booking events | 24.9% | 0.654 | 0.120 |
| Chronological split, for comparison | 0.6% | 0.504 | −0.001 |

Under a plain random split, a model that cannot generalise at all would reach an AUC of 0.81, because copies of a booking sit on both sides of the split. Keeping identical rows together removes that leak: the memoriser does no better than the base rate. The random forest notebook shows the same leak inside a model. The forest's out-of-bag estimate, which does not keep copies together, gives bookings with identical copies a Brier score of 0.009, against 0.103 on the validation set (`05-Random_forest`, Section 3.2).

A smaller leak remains. A quarter of our test bookings share a booking event with a training or validation booking, and a memoriser that looks up events removes 12% of the benchmark's Brier score. Tree ensembles can pick out individual events from fine-grained features such as lead time and arrival week together, so this may tilt the comparison towards them. Rather than change the agreed split, we measure the tilt: `07-Conclusion` also reports the Brier score on the 13,450 test bookings whose booking event does not occur in training or validation, and the order of the three models and the gaps between them are almost the same there. The copies also mean that test rows are not independent, so intervals come from a cluster bootstrap that resamples each block of identical rows as one unit.

### 4.3 A random split cannot test the future

Our split spreads seasons, years and hotels evenly over the three sets (cancellation rates 37.3%, 37.5% and 36.8%), so differences in score reflect the models rather than the time of year. The price is that the test set only asks how well a model predicts bookings from the months and years it was trained on, whereas a hotel would apply it to next season's bookings.

Our data show what this can cost. In `04-XGBoost` (Section 3.2), adding the arrival month and week as features improves the validation log loss under our random split (0.330, against 0.336 without them). An earlier time-based split (training on 2015–16, validating on late 2016) gave the opposite answer (0.451 with them, against 0.412 without): the model had learned the seasonal pattern of particular years. **The standardised evaluation rewards features that a deployed model should drop.** To show how much of each model's skill depends on having seen the same months, we planned to refit each model with its chosen settings on arrivals up to 31 December 2016 and score it on the 40,509 arrivals of 2017, of which 38.8% cancelled (against 36.4% before 2017). We ran this check only for XGBoost, with settings tuned on a time-based split and without the arrival month and week (`ThomasF/ThomasF_test.ipynb`): on 2017 its AUC fell from 0.930 to 0.885, and it predicted 34.6% cancellations against an actual 38.8%. `07-Conclusion` discusses this among its limitations.

### 4.4 Summary

| Standardising choice | What it buys | What it can hide | How we expose it |
| --- | --- | --- | --- |
| One primary score (Brier) | A single, proper ranking | A hotel with extreme costs may prefer another model | Murphy diagram over $0.3\le\theta\le0.8$ |
| One threshold (½) for the decision metrics | Comparable confusion matrices | The hotel's own threshold may differ | Expected cost at $\theta=2/3$; Murphy diagram |
| One pooled test set | A large sample and tight intervals | Different behaviour in the two hotels | Metrics for each hotel (training on one hotel and testing on the other was planned but not run) |
| A random split, identical rows kept together | The same mix of seasons, years and hotels in every set | Predictions of future bookings are never tested; rooms of one booking can straddle the split | Metrics on unseen booking events; refit on 2015–16 and score 2017 (XGBoost only) |

## 5. What "truly best" means

### 5.1 One test set is noisy

Even 17,903 test bookings leave noticeable noise. We simulated two calibrated models with a small true difference (Brier score 0.1352 against 0.1384; AUC 0.876 against 0.871) and scored them on 1,000 independent test sets of each size (Figure 3).

![Probability of picking the wrong model, and power, against test-set size](../LukeD/metrics/figures/fig4_ranking_noise.png)

*Figure 3. Left: how often a single test set ranks the truly worse model first. Right: the power of the paired Diebold–Mariano test and of an unpaired test that ignores the pairing. Source: `simulations.py`, S4.*

* **Accuracy is the noisiest yardstick.** At our test-set size it picks the worse model 7% of the time, against about 2% for the Brier score, log loss and AUC, because it throws away how far each probability is from the truth.
* **Comparisons must be paired.** Every model is scored on the same bookings, and the models find the same bookings hard, so the difference between two models is known much more precisely than their separate scores suggest. At our test-set size, the paired Diebold–Mariano test (Diebold & Mariano, 1995) detects the difference 54% of the time, and an unpaired test only 25%.
* **"No significant difference" is a legitimate result.** Even with pairing, a real Brier difference of 0.003 goes undetected in almost half of all test sets of our size. If that happens to us, we report it rather than declare a winner.

### 5.2 Different metrics crown different winners

To see what disagreement between metrics looks like, we used `evaluate.py` to score three synthetic models on a synthetic test set the size of ours. Model A is calibrated but less informative; B is more informative but over-confident; C is as informative as B but reports vote fractions, as a random forest does. **These are not results from our models.**

| Metric | Model A | Model B | Model C |
| --- | --- | --- | --- |
| Brier score | 0.1266 | 0.1164 | **0.1087** |
| Log loss | 0.3958 | 0.4140 | **0.3524** |
| ROC AUC | 0.8907 | **0.9219** | 0.9195 |
| Calibration slope | **1.01** | 0.50 | 1.19 |
| F1 at ½ | 0.744 | **0.793** | 0.782 |

AUC and F1 prefer B, the proper scores prefer C, and the calibration slope prefers A. The Brier score and log loss even disagree about A against B, and both differences are significant: the Brier score prefers B (by 0.0101) and log loss prefers A (by 0.0182). The Murphy diagram (Figure 4) explains why. B's expected cost is lower than A's for cost ratios between 0.10 and 0.79 and higher at the extremes, which log loss weights heavily. The diagram also shows that the best model depends on the hotel's costs: at $\theta=0.85$, a walked guest costing about 5.7 times an empty room, the AUC winner B is the most expensive of the three. Finally, the CORP decomposition shows that B ranks bookings best but is the worst calibrated: after an ideal recalibration of both models, B's Brier score would be 0.1057 and C's 0.1075. This is why each of us may recalibrate on validation data, and why we report DSC separately.

![Murphy diagram and differences against the best-Brier model](../LukeD/metrics/figures/fig3_worked_murphy.png)

*Figure 4. Left: Murphy diagram, the mean cost per booking of acting at each cost ratio θ (lower is better). Right: each model's difference from C, the model with the best Brier score, with 95% pointwise bootstrap bands; above zero means worse than C. Source: `simulations.py`, S3.*

### 5.3 Our rule

We call one model **best** only if all three of these hold:

1. **Its Brier score is significantly lower than each rival's.** The paired 95% interval for each difference excludes zero, with Diebold–Mariano tests adjusted for the three pairwise comparisons by Holm's (1979) method, and intervals from the cluster bootstrap.
2. **There is no reversal on the Murphy diagram.** No rival is significantly cheaper at any cost ratio between 0.3 and 0.8.
3. **The same ordering holds in both hotels.**

Otherwise we report the models as statistically indistinguishable and say why. In the worked example, C passes all three against both rivals, but only just: a hotel whose costs sit near $\theta=0.4$ could not tell B and C apart.

## 6. The protocol we agreed

1. **Task.** Every model predicts the probability that a booking is cancelled (`is_canceled`), using only information available when the booking is made.
2. **Data and split.** `Data/hotel_bookings_clean.csv` and its `split` column (`02-Preprocessing`): fit on the 83,324 training bookings, tune and recalibrate on the 17,331 validation bookings, and predict the 17,903 test bookings. `arrival_date` and `arrival_date_year` are not used as features, and any cross-validation folds keep identical rows together.
3. **Submissions.** Each member saves `<Name>/test_preds.csv` with columns `id,prediction`, where `id` is the booking's 0-based row number in the cleaned file. Tuning, thresholds and recalibration use training and validation data only; the test labels are used once, to score the final models. Random-forest probabilities are recalibrated so that none is exactly 0 or 1.
4. **Scoring.** `LukeD/metrics/evaluate.py` scores every submission against the shared answer key, `Data/test_truth.csv`, with the metrics of Sections 2 and 3, overall and for each hotel. Its implementations are checked against scikit-learn and SciPy by `LukeD/metrics/test_evaluate.py`.
5. **Best.** The rule of Section 5.3.
6. **Generalisation checks.** The metrics on the test bookings whose booking event is unseen; each model refitted on arrivals up to the end of 2016 and scored on 2017; and each model trained on one hotel and tested on the other. Only the first was done for all three models; the second was run for XGBoost alone, and the third not at all (Section 4).

## 7. Limitations

* **The even weighting of cost ratios is a choice.** A hotel that knows its costs should read the Murphy diagram at its own $\theta$ rather than rely on the Brier score.
* **The Brier score barely distinguishes small probabilities.** Predicting 1% instead of 0.1% for a booking that is kept changes its Brier score by less than 0.0001. Log loss, which we also report, is far more sensitive there.
* **Our cost ratios are assumptions.** Nothing in our data tells us what a walked guest or an empty room costs, so the range $0.3\le\theta\le0.8$ and the case $\theta=2/3$ are illustrations.
* **Significant need not mean important.** With 17,903 bookings, differences too small to change any decision can be highly significant: in the worked example, B and C differ in AUC by only 0.002, with $p\approx2\times10^{-10}$.
* **Our test set is a random sample from the same two hotels and two years.** It measures how well a model fills gaps in the data it was trained on. The checks of Section 4 show only partly how it would do on future bookings, and not at all how it would do at a third hotel.

## References

* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
* Cox, D. R. (1958). [Two further applications of a model for binary regression](https://doi.org/10.1093/biomet/45.3-4.562). *Biometrika*, 45(3/4), 562–565.
* DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
* Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
* Dimitriadis, T., Gneiting, T. and Jordan, A. I. (2021). [Stable reliability diagrams for probabilistic classifiers](https://doi.org/10.1073/pnas.2016191118). *Proceedings of the National Academy of Sciences*, 118(8), e2016191118.
* Ehm, W., Gneiting, T., Jordan, A. and Krüger, F. (2016). [Of quantiles and expectiles: consistent scoring functions, Choquet representations and forecast rankings](https://arxiv.org/abs/1503.08195). *Journal of the Royal Statistical Society: Series B*, 78(3), 505–562.
* Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973–978.
* Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Lipton, Z. C., Elkan, C. and Narayanaswamy, B. (2014). [Optimal thresholding of classifiers to maximize F1 measure](https://doi.org/10.1007/978-3-662-44851-9_15). *Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2014*, LNCS 8725, 225–239.
* Schervish, M. J. (1989). A general method for comparing probability assessors. *The Annals of Statistics*, 17(4), 1856–1879.
