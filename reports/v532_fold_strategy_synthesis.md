# v5.3.2 candidate1 fold strategy check

## 1. 목적
최종 제출 후보인 v5.3.2 candidate1은 `FS8_no_BP_keep_BMI_metabolic + ENC4_no_ordinal_except_binary`와 raw RBF-SVR을 사용합니다.
이번 실험은 모델/feature를 바꾸지 않고 validation fold 방식만 비교해 10-fold 선택 근거를 확보하기 위한 것입니다.

## 2. 결과 요약
```text
            strategy   splitter  n_splits  mean_mae  std_mae  fold_mae_range  target_mean_std_across_folds  pred_std  endpoint_0_count  endpoint_1_count
     kfold_20_seed42      kfold        20  0.127323 0.011029        0.043600                      0.029200  0.201870                16                12
     kfold_15_seed42      kfold        15  0.131013 0.009900        0.033150                      0.025112  0.198974                15                12
     kfold_10_seed42      kfold        10  0.134170 0.007928        0.021767                      0.018178  0.196518                14                12
stratified_10_seed42 stratified        10  0.134607 0.011168        0.034167                      0.002827  0.196630                16                12
 stratified_5_seed42 stratified         5  0.144327 0.005342        0.014183                      0.000873  0.188571                14                10
      kfold_5_seed42      kfold         5  0.149050 0.008868        0.023767                      0.009608  0.183513                11                12
```

가장 낮은 CV는 `kfold_20_seed42`의 0.127323입니다.
기준 10-fold KFold는 0.134170, Stratified 10-fold는 0.134607, 5-fold KFold는 0.149050입니다.

## 3. 10-fold를 선택한 이유
- train size가 3000개로 크지 않아 5-fold보다 각 fold train 비율을 높이는 10-fold가 유리합니다.
- 10-fold는 validation fold가 300개라 fold별 target 분포가 과도하게 작아지지 않으면서도 OOF를 안정적으로 만들 수 있습니다.
- 5-fold는 validation fold가 커서 fold MAE는 덜 출렁일 수 있지만, 각 모델이 80% train만 보고 학습하므로 최종 full-train 제출 모델과의 train-size gap이 큽니다.
- StratifiedKFold 10-fold도 확인했지만, target이 0~100 grid로 촘촘하고 KFold shuffle만으로도 fold target mean 편차가 작아 큰 이득이 없었습니다.
- 최종 제출 모델은 full train fit이므로, local CV는 leaderboard를 맞추는 도구가 아니라 모델 선택의 안정성 확인용입니다. 이 관점에서 10-fold KFold는 bias와 variance의 균형이 좋습니다.

## 4. 보고서 문장
Validation은 10-fold KFold(shuffle=True, random_state=42)를 사용했다. 데이터 수가 3000개로 제한적이기 때문에 5-fold보다 각 fold의 학습 비율을 높여 full-train 제출 상황과의 차이를 줄이고, 동시에 validation fold당 약 300개 샘플을 확보해 fold별 MAE와 OOF 분포를 안정적으로 비교할 수 있었다. 추가로 5-fold 및 StratifiedKFold를 확인했으며, 10-fold KFold가 성능과 안정성 측면에서 균형적인 기준으로 판단되었다.
