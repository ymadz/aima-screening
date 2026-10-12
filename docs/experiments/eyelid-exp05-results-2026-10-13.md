# Experiment 5 — verified CP-AnemiC transfer results (2026-10-13)

CP-AnemiC pretraining did not establish an improvement that justifies its added complexity on AIMA. The predefined joint policy detected **11/19 anemia cases (57.9% sensitivity) at 78% specificity**, compared with **10/19 (52.6%) at 84% specificity** for the predefined Experiment 4 robust color comparison. One additional detection came with six additional false positives. Balanced accuracy was slightly lower (67.95% versus 68.32%), ROC-AUC was 0.734211 versus 0.709474, and AP was 0.372546 versus 0.385148. The paired sensitivity difference had a conditional 95% interval of −21.1 to +31.6 percentage points; the specificity difference was −6.0 points [−11.0, −2.0]. This does not establish independent or clinically acceptable improvement.

The predefined CP-classification frozen-feature plus color candidate C detected **12/19 (63.2%) at 78% specificity**, with ROC-AUC 0.734211 and AP 0.372166. Its sensitivity 95% Wilson interval was 41.0–80.9%; specificity was 68.9–85.0%. Against the robust color comparator, its +10.5-point sensitivity difference had an interval of [−10.5, +36.8], while specificity fell by 6.0 points [−11.0, −2.0]. This is a predefined exploratory candidate comparison, not a replacement primary policy chosen from outer results. CP classification's actual inner-selected A/B/C policy detected 10/19 at 77% specificity; it selected C in four folds and B in one.

CP Hb-regression C and its inner-selected policy detected **11/19 at 79% specificity** (ROC-AUC 0.730000; AP 0.370429), with exactly the same binary OOF decisions as matched ImageNet C (11/19 at 79%; ROC-AUC 0.730526; AP 0.368331). CP classification C added one detection and one false positive versus matched ImageNet C; its paired ROC-AUC and AP intervals included zero. The frozen-feature performance is therefore compatible with ImageNet features plus robust color statistics without useful added CP pretraining. Head-only and partial CP adaptations with training-selected thresholds detected only 2/19 and 0/19 for classification, or 4/19 and 0/19 for regression. These outer results did not guide any redesign.

The public-only pediatric classifier detected **0/19 at 96% specificity** (ROC-AUC 0.459474; AP 0.150111). Public classification validation was weak (ROC-AUC 0.478125), and both bounded public objectives selected epoch 1. Public-only Hb prediction on AIMA had MAE 2.225671 g/dL, bias −1.905735 g/dL and R² −1.069180. Although its MAE was below the public-training-mean baseline of 2.801160, substantial adult underprediction and negative R² provide no evidence of a usable Hb estimator. Weak public fitting and observed acquisition/label differences are plausible explanations, not proven causes.

Retain robust color-only classification as the practical candidate for independent confirmation, with a prespecified acquisition, model and operating-rule policy and genuinely new positive and borderline cases. Frozen embeddings can remain a predefined comparator in that study. These repeated-cohort results do not justify additional CP fine-tuning complexity or choosing a new winner by outer sensitivity. The inner specificity constraint of 0.80 was a research threshold policy and did not guarantee 0.80 outer specificity. No clinical acceptability threshold or independent validation has been established.


## Completed source and methods

CP-AnemiC Version 2 was selected after official archive authentication (DOI 10.17632/m53vz6b7fx.2). All 503 local files match the release byte-for-byte. Its 251 canonical RGBA images contain 110 anemia and 141 non-anemia labels (43.8247% anemia); the 251 copies in All Categories are excluded. The true patient count is unavailable. The 251 curation IDs plus five near-image pairs form 246 conservative split groups. The grouped split has 199 training images (90 anemia/109 non-anemia) and 52 validation images (20 anemia/32 non-anemia), seed 20261012. Public validation is checkpoint-selected development evidence on image groups.

Version 1 was authenticated and inspected: 710 image files, 498 unique file hashes, 485 decoded rasters and 87 repeated-image groups. Of those groups, 86 contain conflicting Hb measurements and one crosses binary class labels. All 251 Version 2 images are byte-identical Version 1 subsets with matching source IDs and Hb; none belongs to a conflicting exact-image group. The releases were not combined. The original public pediatric target, Hb <11 g/dL, is retained separately from local male <13/female <12 labels. Observed local ages are 16–61, and non-pregnancy remains an owner attestation outside the Sheet. No individual demographics or patient linkage were borrowed from Version 1.

Public alpha=255 tissue and local approved manual 0/255 masks use the same tight bounding box, ImageNet-mean background before resizing, aspect-preserving 224-square bilinear-antialiased resize, and mean letterboxing. The input has three RGB channels. Partially transparent boundary pixels are excluded. Anatomical masks, glare and clinically meaningful color remain otherwise unchanged. Public untagged RGB is explicitly assumed sRGB for computation without evidence of calibration. Local neural RGB reuses verified ICC/DNG profile handling; classical robust18 exactly matches the historical original-rendering manual-mask features. Common processing reduces artificial transparency/background cues but cannot erase acquisition or geometry differences.

Both public objectives initialize verified ImageNet1K_V1 MobileNetV3-Small weights. Earlier blocks and BatchNorm statistics remain frozen; original and flipped early maps are cached without labels. Classification uses training-only weighted BCE and validation AP stopping. Regression uses training-only Hb mean/std, standardized SmoothL1 and physical-g/dL validation MAE stopping. Both use head LR 0.001, tail LR 0.0001, AdamW decay 0.001, batch size 32, maximum 24 epochs and patience 4. Both real public runs stopped after five epochs, selecting epoch 1. The bounded procedure was fixed before local evaluation.

For each CP objective and the matched ImageNet control, A replaces the public output with a new adult head and freezes the backbone (head LR 0.001). B replaces the head and adapts only the last two blocks (head LR 0.0003/tail LR 0.00001). A/B use fixed BatchNorm statistics, weighted BCE, maximum 12 epochs, patience 4, inner AP stopping and rounded-median inner best epochs for the outer refit. C uses frozen 576-dimensional embeddings plus exact robust18 color features: training-only embedding StandardScaler/PCA8, then a fitted StandardScaler on 26 inputs and L2 logistic regression with C=0.1, balanced weights, lbfgs and maximum 2,000 iterations. New adult heads use identical seeds across matched objectives; pediatric and regression heads are discarded.

The 119 subjects (19 anemia/100 non-anemia) and five outer folds are unchanged. Three inner folds use seed 20261011. Nine candidates and tie order were fixed before fitting. Per-objective and joint policies select pooled inner AP; the training API has no outer labels. Both fixed 0.5 and inner-selected thresholds are reported. The latter maximize sensitivity subject to specificity ≥0.80, with specificity/higher-cutoff ties. All scaling, PCA, class weights, checkpoint epochs and threshold decisions use the relevant training folds. No calibration, outer tuning or classifier holdout is introduced.

Uncertainty uses 95% Wilson intervals for class rates and 2,000 conditional stratified paired subject bootstrap draws for ranking metrics, F1, balanced accuracy and differences, seed 20261011. The same draws compare the robust color benchmark and matched CP versus ImageNet adaptations. Intervals condition on saved scores, fix prevalence and omit refit, selection and overlapping-fold uncertainty. Public-only pediatric scores on AIMA remain exploratory; Hb regression is reported in g/dL with bias and training-mean baselines. Local Hb and outer labels never guide public objective selection.

## Full fixed and training-selected comparisons

| Model / rule | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP | TP/FN/TN/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp01_fixed | 0.848739 | 0.105263 | 0.990000 | 0.547632 | 0.666667 | 0.181818 | 0.719474 | 0.387866 | 2/17/99/1 |
| lab_nested_selected | 0.764706 | 0.368421 | 0.840000 | 0.604211 | 0.304348 | 0.333333 | 0.719474 | 0.387866 | 7/12/84/16 |
| exp02_selected | 0.739496 | 0.368421 | 0.810000 | 0.589211 | 0.269231 | 0.311111 | 0.575789 | 0.247161 | 7/12/81/19 |
| exp03_selected | 0.714286 | 0.052632 | 0.840000 | 0.446316 | 0.058824 | 0.055556 | 0.463684 | 0.143881 | 1/18/84/16 |
| exp04_joint_selected | 0.756303 | 0.368421 | 0.830000 | 0.599211 | 0.291667 | 0.325581 | 0.656842 | 0.352228 | 7/12/83/17 |
| exp04_robust_selected | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.709474 | 0.385148 | 10/9/84/16 |
| exp04b_glare_weighted_selected | 0.789916 | 0.526316 | 0.840000 | 0.683158 | 0.384615 | 0.444444 | 0.720526 | 0.392086 | 10/9/84/16 |
| cp_public_only_fixed | 0.806723 | 0.000000 | 0.960000 | 0.480000 | 0.000000 | 0.000000 | 0.459474 | 0.150111 | 0/19/96/4 |
| cp_classification_A_fixed | 0.588235 | 0.526316 | 0.600000 | 0.563158 | 0.200000 | 0.289855 | 0.557368 | 0.202268 | 10/9/60/40 |
| cp_classification_A_threshold | 0.672269 | 0.105263 | 0.780000 | 0.442632 | 0.083333 | 0.093023 | 0.557368 | 0.202268 | 2/17/78/22 |
| cp_classification_B_fixed | 0.630252 | 0.157895 | 0.720000 | 0.438947 | 0.096774 | 0.120000 | 0.466842 | 0.163613 | 3/16/72/28 |
| cp_classification_B_threshold | 0.739496 | 0.000000 | 0.880000 | 0.440000 | 0.000000 | 0.000000 | 0.466842 | 0.163613 | 0/19/88/12 |
| cp_classification_C_fixed | 0.722689 | 0.631579 | 0.740000 | 0.685789 | 0.315789 | 0.421053 | 0.734211 | 0.372166 | 12/7/74/26 |
| cp_classification_C_threshold | 0.756303 | 0.631579 | 0.780000 | 0.705789 | 0.352941 | 0.452830 | 0.734211 | 0.372166 | 12/7/78/22 |
| cp_classification_selected_fixed | 0.655462 | 0.526316 | 0.680000 | 0.603158 | 0.238095 | 0.327869 | 0.695789 | 0.305089 | 10/9/68/32 |
| cp_classification_selected_threshold | 0.731092 | 0.526316 | 0.770000 | 0.648158 | 0.303030 | 0.384615 | 0.695789 | 0.305089 | 10/9/77/23 |
| cp_regression_A_fixed | 0.512605 | 0.631579 | 0.490000 | 0.560789 | 0.190476 | 0.292683 | 0.540526 | 0.193936 | 12/7/49/51 |
| cp_regression_A_threshold | 0.655462 | 0.210526 | 0.740000 | 0.475263 | 0.133333 | 0.163265 | 0.540526 | 0.193936 | 4/15/74/26 |
| cp_regression_B_fixed | 0.672269 | 0.210526 | 0.760000 | 0.485263 | 0.142857 | 0.170213 | 0.445263 | 0.164125 | 4/15/76/24 |
| cp_regression_B_threshold | 0.731092 | 0.000000 | 0.870000 | 0.435000 | 0.000000 | 0.000000 | 0.445263 | 0.164125 | 0/19/87/13 |
| cp_regression_C_fixed | 0.714286 | 0.631579 | 0.730000 | 0.680789 | 0.307692 | 0.413793 | 0.730000 | 0.370429 | 12/7/73/27 |
| cp_regression_C_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730000 | 0.370429 | 11/8/79/21 |
| cp_regression_selected_fixed | 0.714286 | 0.631579 | 0.730000 | 0.680789 | 0.307692 | 0.413793 | 0.730000 | 0.370429 | 12/7/73/27 |
| cp_regression_selected_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730000 | 0.370429 | 11/8/79/21 |
| imagenet_control_A_fixed | 0.512605 | 0.473684 | 0.520000 | 0.496842 | 0.157895 | 0.236842 | 0.518421 | 0.177612 | 9/10/52/48 |
| imagenet_control_A_threshold | 0.722689 | 0.210526 | 0.820000 | 0.515263 | 0.181818 | 0.195122 | 0.518421 | 0.177612 | 4/15/82/18 |
| imagenet_control_B_fixed | 0.680672 | 0.157895 | 0.780000 | 0.468947 | 0.120000 | 0.136364 | 0.426842 | 0.157141 | 3/16/78/22 |
| imagenet_control_B_threshold | 0.747899 | 0.000000 | 0.890000 | 0.445000 | 0.000000 | 0.000000 | 0.426842 | 0.157141 | 0/19/89/11 |
| imagenet_control_C_fixed | 0.731092 | 0.631579 | 0.750000 | 0.690789 | 0.324324 | 0.428571 | 0.730526 | 0.368331 | 12/7/75/25 |
| imagenet_control_C_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730526 | 0.368331 | 11/8/79/21 |
| imagenet_control_selected_fixed | 0.731092 | 0.631579 | 0.750000 | 0.690789 | 0.324324 | 0.428571 | 0.730526 | 0.368331 | 12/7/75/25 |
| imagenet_control_selected_threshold | 0.756303 | 0.578947 | 0.790000 | 0.684474 | 0.343750 | 0.431373 | 0.730526 | 0.368331 | 11/8/79/21 |
| joint_selected_fixed | 0.722689 | 0.631579 | 0.740000 | 0.685789 | 0.315789 | 0.421053 | 0.734211 | 0.372546 | 12/7/74/26 |
| joint_selected_threshold | 0.747899 | 0.578947 | 0.780000 | 0.679474 | 0.333333 | 0.423077 | 0.734211 | 0.372546 | 11/8/78/22 |

`_threshold` means inner-only specificity≥.80 selection; `_fixed` means.5. `_selected` inside a name means strategy selection by inner AP. The E4 robust row is a predefined comparison, not E4 primary selection. Public-only fixed.5 is not an adult calibration. All counts are TP/FN/TN/FP; per-fold metrics and all eight metric intervals are in private metrics.json.

## Conditional uncertainty

| Model/rule | Sensitivity 95% | Specificity 95% | ROC-AUC 95% | AP 95% |
| --- | --- | --- | --- | --- |
| cp_public_only_fixed | 0.000000–0.168179 | 0.901629–0.984337 | 0.325776–0.596842 | 0.125566–0.231330 |
| cp_classification_A_threshold | 0.029359–0.313941 | 0.689296–0.849987 | 0.418947–0.693197 | 0.147932–0.353104 |
| cp_classification_B_threshold | 0.000000–0.168179 | 0.801879–0.930006 | 0.318421–0.613711 | 0.126624–0.291019 |
| cp_classification_C_threshold | 0.410395–0.808505 | 0.689296–0.849987 | 0.600500–0.840013 | 0.235087–0.557620 |
| cp_classification_selected_threshold | 0.317078–0.726702 | 0.678456–0.841567 | 0.567368–0.806842 | 0.201042–0.478447 |
| cp_regression_A_threshold | 0.085077–0.433343 | 0.646290–0.815953 | 0.397368–0.669474 | 0.141784–0.341638 |
| cp_regression_B_threshold | 0.000000–0.168179 | 0.790196–0.922428 | 0.292066–0.597368 | 0.122743–0.304773 |
| cp_regression_C_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.593671–0.838421 | 0.231820–0.553327 |
| cp_regression_selected_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.593671–0.838421 | 0.231820–0.553327 |
| imagenet_control_selected_threshold | 0.362759–0.768581 | 0.700200–0.858343 | 0.594197–0.838434 | 0.231911–0.553629 |
| joint_selected_threshold | 0.362759–0.768581 | 0.689296–0.849987 | 0.598408–0.841079 | 0.235130–0.558791 |

| Model/rule minus predefined E4 robust | Δ sensitivity [95%] | Δ specificity [95%] | Δ ROC-AUC [95%] | Δ AP [95%] |
| --- | --- | --- | --- | --- |
| cp_public_only_fixed | -0.526316 [-0.736842, -0.315789] | 0.120000 [0.040000, 0.200000] | -0.250000 [-0.432105, -0.045789] | -0.235037 [-0.430837, -0.054237] |
| cp_classification_A_threshold | -0.421053 [-0.684211, -0.157895] | -0.060000 [-0.160000, 0.030000] | -0.152105 [-0.325263, 0.035276] | -0.182879 [-0.363717, 0.006277] |
| cp_classification_B_threshold | -0.526316 [-0.736842, -0.315789] | 0.040000 [-0.060000, 0.140000] | -0.242632 [-0.442645, -0.044711] | -0.221535 [-0.408007, -0.039862] |
| cp_classification_C_threshold | 0.105263 [-0.105263, 0.368421] | -0.060000 [-0.110000, -0.020000] | 0.024737 [-0.066882, 0.116316] | -0.012981 [-0.104865, 0.058028] |
| cp_classification_selected_threshold | 0.000000 [-0.263158, 0.263158] | -0.070000 [-0.120000, -0.020000] | -0.013684 [-0.124224, 0.096842] | -0.080059 [-0.226595, 0.050723] |
| cp_regression_A_threshold | -0.315789 [-0.578947, -0.052632] | -0.100000 [-0.210000, 0.000000] | -0.168947 [-0.342671, 0.007908] | -0.191212 [-0.372819, -0.003110] |
| cp_regression_B_threshold | -0.526316 [-0.736842, -0.315789] | 0.030000 [-0.070250, 0.130000] | -0.264211 [-0.471092, -0.058395] | -0.221022 [-0.416234, -0.022033] |
| cp_regression_C_threshold | 0.052632 [-0.210526, 0.315789] | -0.050000 [-0.100000, -0.010000] | 0.020526 [-0.069474, 0.108947] | -0.014718 [-0.105051, 0.056344] |
| cp_regression_selected_threshold | 0.052632 [-0.210526, 0.315789] | -0.050000 [-0.100000, -0.010000] | 0.020526 [-0.069474, 0.108947] | -0.014718 [-0.105051, 0.056344] |
| imagenet_control_selected_threshold | 0.052632 [-0.210526, 0.315789] | -0.050000 [-0.100000, -0.010000] | 0.021053 [-0.069474, 0.111579] | -0.016817 [-0.109757, 0.053251] |
| joint_selected_threshold | 0.052632 [-0.210526, 0.315789] | -0.060000 [-0.110000, -0.020000] | 0.024737 [-0.066855, 0.116316] | -0.012601 [-0.104835, 0.057075] |

| CP minus same ImageNet adaptation | Δ sensitivity [95%] | Δ specificity [95%] | Δ ROC-AUC [95%] | Δ AP [95%] |
| --- | --- | --- | --- | --- |
| cp_classification_A_threshold_minus_imagenet_control_A_threshold | -0.105263 [-0.315789, 0.105263] | -0.040000 [-0.130000, 0.050000] | 0.038947 [-0.061066, 0.137368] | 0.024656 [-0.040556, 0.114491] |
| cp_classification_B_threshold_minus_imagenet_control_B_threshold | 0.000000 [0.000000, 0.000000] | -0.010000 [-0.050000, 0.020000] | 0.040000 [-0.051053, 0.148961] | 0.006471 [-0.049962, 0.060626] |
| cp_classification_C_threshold_minus_imagenet_control_C_threshold | 0.052632 [0.000000, 0.157895] | -0.010000 [-0.030000, 0.000000] | 0.003684 [-0.003684, 0.013158] | 0.003836 [-0.003857, 0.020886] |
| cp_classification_selected_threshold_minus_imagenet_control_selected_threshold | -0.052632 [-0.157895, 0.000000] | -0.020000 [-0.060000, 0.020000] | -0.034737 [-0.097895, 0.032105] | -0.063242 [-0.183308, 0.029560] |
| cp_regression_A_threshold_minus_imagenet_control_A_threshold | 0.000000 [0.000000, 0.000000] | -0.080000 [-0.130000, -0.030000] | 0.022105 [-0.018421, 0.063684] | 0.016324 [-0.032011, 0.093165] |
| cp_regression_B_threshold_minus_imagenet_control_B_threshold | 0.000000 [0.000000, 0.000000] | -0.020000 [-0.070250, 0.040000] | 0.018421 [-0.025263, 0.060000] | 0.006984 [-0.013459, 0.037810] |
| cp_regression_C_threshold_minus_imagenet_control_C_threshold | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | -0.000526 [-0.007895, 0.005789] | 0.002099 [-0.006588, 0.014801] |
| cp_regression_selected_threshold_minus_imagenet_control_selected_threshold | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | -0.000526 [-0.007895, 0.005789] | 0.002099 [-0.006588, 0.014801] |

Only19 positives: one detection changes sensitivity by5.3 percentage points. These intervals do not establish independent clinical improvement.

## Public objectives and unadapted Hb

| Evaluation | MAE g/dL [95%] | RMSE g/dL | Bias predicted−lab g/dL [95%] | R² |
| --- | --- | ---: | --- | ---: |
| Public CP regression validation | 1.699905 [1.324415, 2.107379] | 2.251749 | 0.435266 [-0.141504, 1.059850] | -0.001030 |
| Public validation training-mean baseline | 1.802457 [1.452457, 2.173687] | 2.251072 | -0.046542 [-0.621735, 0.603506] | -0.000428 |
| Public-only regression on AIMA | 2.225671 [1.965814, 2.498436] | 2.631343 | -1.905735 [-2.215274, -1.578677] | -1.069180 |
| Public-training mean on AIMA | 2.801160 [2.545484, 3.072611] | 3.154166 | -2.569538 [-2.882164, -2.241765] | -1.973120 |

Public pediatric classification validation (52 images,20 anemia/32 non-anemia): sensitivity 0.200000, specificity 0.750000, ROC-AUC 0.478125, AP 0.429875. Checkpoint selected on this same validation set; true patient separation is unverified. Class-specific score distributions are preserved in public_validation/metrics.json and public_only/metrics.json.

## Frozen inner policies by outer fold

| Fold | CP classification | CP Hb representation | ImageNet | Joint nine-candidate policy |
| --- | --- | --- | --- | --- |
| 1 | cp_classification_C | cp_regression_C | imagenet_control_C | cp_classification_C |
| 2 | cp_classification_C | cp_regression_C | imagenet_control_C | cp_classification_C |
| 3 | cp_classification_C | cp_regression_C | imagenet_control_C | imagenet_control_C |
| 4 | cp_classification_B | cp_regression_C | imagenet_control_C | imagenet_control_C |
| 5 | cp_classification_C | cp_regression_C | imagenet_control_C | cp_regression_C |

## Domain observations and limits

Public/local median opaque tissue counts are4,231/104,517 pixels; median native public dimensions76×213 versus local photos4032×3024, with median local tissue bbox293×825. Median tissue LAB means are public[62.36,24.93,17.27] and local[65.69,46.29,21.07]. Local median any-channel clipping fraction is.3777 versus public0; this is a clipping proxy, not a glare mask or proven error cause. Median public partial-alpha/visible fraction is.04882 (max.10669). The private shared-input sheet shows residual shape/orientation/brightness differences. No diagnostic statistic selected a model or excluded a case. Local Hb5.9–18.4, median13.5;19 anemia cases comprise11 mild/6 moderate/2 severe under the retained descriptive policy. Public Hb4.2–15.0, full mean/median/std and descriptive pediatric bands are in audit/public_audit.json. Capture calibration, patient linkage, measurement noise and repeated development remain limitations.

## Preservation and reproducibility

Training commit `cf735ed2623f1f62fbb5ab7dd80f15249fd0d679`; branch `experiment/eyelid-exp05-cp-transfer`; worktree `/Users/madz/Documents/aima-screening-exp05-cp-transfer`. Completed2 public neural models,120 local neural models and60 classical pipelines. Checksum-first independent verification reproduced all182 models, public/inner/outer predictions, frozen parameters, train-only Hb normalization/PCA/scalers, decisions and all pooled/per-fold metrics and paired intervals, with fresh all370 inputs and both-view early maps. 637 files are covered by the sealed manifest;8 historical bundles and all source hashes remain unchanged. Peak RSS521.6MB; runtime624.8s; artifact size1294.4MiB before final checksum manifest.

Private artifacts: `data/processed/local_eyelid_classification_exp05_cp_transfer_2026-10-12_v1/` (models, inputs, metrics, report, domain panel, figures, checksums, copied implementation/environment and REPRODUCTION.md); independent evidence: `data/processed/exp05_verification_2026-10-13_v1/`. Code/tests/aggregate documents are committed; datasets, clinical rows, masks, predictions, images and checkpoints remain ignored. Previous worktrees and preservation refs are untouched; no external push.

Tests: full suite122 tests,119 passed/3 legacy-source skips; all11 new CP tests pass, compileall and Git whitespace checks pass. Central documentation validator checks source-root artifact manifests and aggregate metrics without reading patient rows.
