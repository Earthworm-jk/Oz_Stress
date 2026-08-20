# v5.3.3 RBF representation final check

## 1. v5.3.2 best candidate 재현
v5.3.2 candidate1 재현 CV MAE는 0.134170입니다.
기준 0.134170 근처로 재현되었으며, 기존 candidate1 제출과의 diff는 `{'left': 'v532_candidate_1_fs8_no_bp_keep_bmi_metabolic_enc4_no_ordinal_except_binary.csv', 'right': 'v533_base_reproduce', 'different_row_count': 0, 'mean_abs_diff': 0.0, 'max_abs_diff': 0.0, 'prediction_correlation': 1.0, 'left_pred_mean': 0.49649333333333334, 'left_pred_std': 0.1981157571723382, 'right_pred_mean': 0.49649333333333334, 'right_pred_std': 0.1981157571723382, 'left_endpoint_0_count': 4, 'left_endpoint_1_count': 5, 'right_endpoint_0_count': 4, 'right_endpoint_1_count': 5}`입니다.

## 2. C/gamma narrow tuning
가장 좋은 C/gamma 후보는 `cg_c0.85_g1`이고 CV MAE는 0.134170, base 대비 개선폭은 0.000000입니다.

```text
     candidate        C    gamma  mean_mae  improvement_vs_v532_base  pred_std candidate_level
   cg_c0.85_g1 3.369001 1.063162  0.134170                  0.000000  0.196518            hold
      cg_c1_g1 3.963531 1.063162  0.134170                  0.000000  0.196518            hold
   cg_c1.15_g1 4.558060 1.063162  0.134170                  0.000000  0.196518            hold
cg_c0.85_g1.15 3.369001 1.222636  0.134213                 -0.000043  0.196147            hold
   cg_c1_g1.15 3.963531 1.222636  0.134213                 -0.000043  0.196147            hold
cg_c1.15_g1.15 4.558060 1.222636  0.134213                 -0.000043  0.196147            hold
   cg_c1_g0.85 3.963531 0.903687  0.134220                 -0.000050  0.196985            hold
cg_c0.85_g0.85 3.369001 0.903687  0.134220                 -0.000050  0.196985            hold
cg_c1.15_g0.85 4.558060 0.903687  0.134220                 -0.000050  0.196985            hold
```

## 3. Numeric scaler / distribution representation
가장 좋은 scaler 후보는 `SCALE0_current`이고 CV MAE는 0.134170, 개선폭은 0.000000입니다.
X feature의 QuantileTransformer는 target quantile이 아니라 numeric distribution normalization 실험입니다.

```text
                    candidate                 scaler  mean_mae  improvement_vs_v532_base  pred_std candidate_level
               SCALE0_current                current  0.134170                  0.000000  0.196518            hold
                SCALE2_robust                 robust  0.134170                  0.000000  0.196518            hold
              SCALE1_standard               standard  0.135577                 -0.001407  0.194124            hold
      SCALE3_power_yeojohnson       power_yeojohnson  0.145667                 -0.011497  0.179056            hold
SCALE4_quantile_normal_X_only quantile_normal_x_only  0.168227                 -0.034057  0.153607            hold
```

## 4. Metabolic representation
가장 좋은 metabolic 후보는 `MET4_log_product_plus_raw_ratio`이고 CV MAE는 0.134157, 개선폭은 0.000013입니다.
ratio/product가 서로 다른 정보를 주는지, log product가 거리 구조를 안정화하는지 확인했습니다.

```text
                      candidate             metabolic_mode  mean_mae  improvement_vs_v532_base  pred_std      candidate_level
MET4_log_product_plus_raw_ratio log_product_plus_raw_ratio  0.134157                  0.000013  0.196510 tiny_micro_candidate
     MET3_log_ratio_log_product      log_ratio_log_product  0.134160                  0.000010  0.196522 tiny_micro_candidate
     MET0_current_ratio_product      current_ratio_product  0.134170                  0.000000  0.196518                 hold
              MET2_product_only               product_only  0.134293                 -0.000123  0.196713                 hold
                MET1_ratio_only                 ratio_only  0.134357                 -0.000187  0.196620                 hold
      MET5_no_metabolic_derived       no_metabolic_derived  0.134583                 -0.000413  0.196899                 hold
```

## 5. One-hot block weight
가장 좋은 categorical block weight 후보는 `CATW3_1.25`이고 CV MAE는 0.134047, 개선폭은 0.000123입니다.
weight를 낮춘 후보가 좋아지면 one-hot block 영향이 다소 강했다는 해석이 가능하고, 1.0이 유지되면 기존 균형이 충분하다고 봅니다.

```text
 candidate  cat_weight  mean_mae  improvement_vs_v532_base  pred_std candidate_level
CATW3_1.25        1.25  0.134047                  0.000123  0.196476 micro_candidate
CATW0_1.00        1.00  0.134170                  0.000000  0.196518            hold
CATW1_0.75        0.75  0.134403                 -0.000233  0.196818            hold
CATW2_0.50        0.50  0.135433                 -0.001263  0.199186            hold
```

## 6. Sentinel micro
가장 좋은 sentinel 후보는 `SENT150`이고 CV MAE는 0.134150, 개선폭은 0.000020입니다.
설명 가능성 기준으로는 sentinel99를 우선 유지하고, sentinel150은 micro-gamble로만 봅니다.

```text
candidate  sentinel_value  mean_mae  improvement_vs_v532_base  pred_std      candidate_level
  SENT150           150.0  0.134150                  0.000020  0.196556 tiny_micro_candidate
   SENT99            99.0  0.134170                  0.000000  0.196518                 hold
   SENT50            50.0  0.134347                 -0.000177  0.196175                 hold
```

## 7. 최종 제출 후보
`v533_candidate_1_catw3_1p25_fs8_enc4.csv`를 micro 후보로 볼 수 있습니다.

```text
 rank                       candidate      group  mean_mae  improvement_vs_v532_base                                               submission_file
    1                      CATW3_1.25 cat_weight  0.134047                  0.000123                      v533_candidate_1_catw3_1p25_fs8_enc4.csv
    2                         SENT150   sentinel  0.134150                  0.000020                         v533_candidate_2_sent150_fs8_enc4.csv
    3 MET4_log_product_plus_raw_ratio  metabolic  0.134157                  0.000013 v533_candidate_3_met4_log_product_plus_raw_ratio_fs8_enc4.csv
```

baseline v5.3.2 candidate1 대비 test prediction diff:

```text
           left                           right  different_row_count  mean_abs_diff  max_abs_diff  prediction_correlation  left_pred_mean  left_pred_std  right_pred_mean  right_pred_std  left_endpoint_0_count  left_endpoint_1_count  right_endpoint_0_count  right_endpoint_1_count
v532_candidate1                      CATW3_1.25                  374       0.001370          0.03                0.999792        0.496493       0.198116         0.496537        0.198083                      4                      5                       4                       5
v532_candidate1                         SENT150                   11       0.000037          0.01                0.999995        0.496493       0.198116         0.496490        0.198119                      4                      5                       4                       5
v532_candidate1 MET4_log_product_plus_raw_ratio                   57       0.000190          0.01                0.999976        0.496493       0.198116         0.496490        0.198095                      4                      5                       4                       5
```

## 8. 0.12999 진입 가능성
CV 개선폭이 0.0005 이상이고 prediction std와 endpoint가 유지되는 후보라면 기존 LB 0.13023에서 0.12999 이하 진입 가능성을 기대할 근거가 있습니다.
다만 이번 실험은 LB fitting이 아니라 fold-safe local CV와 설명 가능한 representation 개선에 근거한 후보 선별입니다.

## 9. PPT 보고 문장
- 최종 단계에서는 RBF-SVR의 거리 기반 특성을 고려해 feature representation을 추가 점검했습니다.
- 범주형 변수는 one-hot으로 분리하되, one-hot block의 거리 영향이 과도하지 않은지 확인했습니다.
- 대사 관련 조합 변수는 ratio와 product가 서로 다른 정보를 제공하는지 ablation으로 확인했습니다.
- 최종 후보는 단순한 변수 추가 모델이 아니라, RBF가 해석 가능한 거리 구조를 학습하도록 feature space를 정리한 모델입니다.
