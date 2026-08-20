# v5.3.2 final micro synthesis

## 1. Feature ablation 결론
기준 CV MAE는 0.134917이고, 재현 baseline은 v5.3 sentinel99와 같은 설정입니다.

가장 좋은 feature set은 `FS8_no_BP_keep_BMI_metabolic`이며 CV MAE는 0.134527, 기준 대비 개선폭은 0.000390입니다.

```text
                       feature_set  mean_mae  improvement_vs_baseline  pred_std      candidate_level
      FS8_no_BP_keep_BMI_metabolic  0.134527                 0.000390  0.196609 tiny_micro_candidate
FS3_raw_plus_BMI_metabolic_product  0.134767                 0.000150  0.196897      micro_candidate
  FS4_raw_plus_BMI_metabolic_ratio  0.134810                 0.000107  0.196700      micro_candidate
               FS0_baseline_S2_all  0.134917                 0.000000  0.195513                 hold
      FS7_no_product_keep_ratio_BP  0.134983                -0.000067  0.195632                 hold
             FS1_raw_plus_BMI_only  0.134993                -0.000077  0.197105                 hold
      FS6_no_ratio_keep_product_BP  0.135117                -0.000200  0.195696                 hold
               FS2_raw_plus_BMI_BP  0.135283                -0.000367  0.195827                 hold
           FS5_raw_only_no_derived  0.135527                -0.000610  0.197553                 hold
```

## 2. RBF 거리 구조를 해친 파생변수 여부
파생변수 제거/정리 실험의 목적은 feature를 더 늘리는 것이 아니라 RBF 거리 구조에 불필요한 축이 있는지 보는 것입니다.
`FS8_no_BP_keep_BMI_metabolic` 결과를 기준으로 보면, CV 개선폭이 0.0003 이상이면 의미 있는 정리 후보로 볼 수 있고, 그보다 작으면 baseline S2 전체가 충분히 안정적이라고 해석합니다.

## 3. Categorical encoding 결론
가장 좋은 encoding은 `ENC5_missing_explicit_for_edu_only`이며 CV MAE는 0.134170, 기준 대비 개선폭은 0.000747입니다.

```text
                          encoding                  feature_set  mean_mae  improvement_vs_baseline  pred_std      candidate_level
ENC5_missing_explicit_for_edu_only FS8_no_BP_keep_BMI_metabolic  0.134170                 0.000747  0.196518 submission_candidate
     ENC4_no_ordinal_except_binary FS8_no_BP_keep_BMI_metabolic  0.134170                 0.000747  0.196518 submission_candidate
                   ENC1_all_onehot FS8_no_BP_keep_BMI_metabolic  0.134203                 0.000713  0.196487 submission_candidate
                  ENC0_v53_current FS8_no_BP_keep_BMI_metabolic  0.134527                 0.000390  0.196609 tiny_micro_candidate
                ENC3_sleep_revised FS8_no_BP_keep_BMI_metabolic  0.134527                 0.000390  0.196609 tiny_micro_candidate
               ENC2_clinical_order FS8_no_BP_keep_BMI_metabolic  0.134527                 0.000390  0.196609 tiny_micro_candidate
```

all-onehot 계열이 좋아지면 category 간 임의 순서보다 분리 표현이 RBF에 적합하다는 뜻이고, current가 유지되면 기존 단순 encoding만으로도 거리 구조가 충분하다는 해석입니다.

## 4. 최종 조합 후보
```text
 rank                  feature_set                           encoding  mean_mae  improvement_vs_baseline                                                                      submission_file      candidate_level
    1 FS8_no_BP_keep_BMI_metabolic      ENC4_no_ordinal_except_binary  0.134170                 0.000747      v532_candidate_1_fs8_no_bp_keep_bmi_metabolic_enc4_no_ordinal_except_binary.csv submission_candidate
    2 FS8_no_BP_keep_BMI_metabolic ENC5_missing_explicit_for_edu_only  0.134170                 0.000747 v532_candidate_2_fs8_no_bp_keep_bmi_metabolic_enc5_missing_explicit_for_edu_only.csv submission_candidate
    3 FS8_no_BP_keep_BMI_metabolic                    ENC1_all_onehot  0.134203                 0.000713                    v532_candidate_3_fs8_no_bp_keep_bmi_metabolic_enc1_all_onehot.csv submission_candidate
```

baseline sentinel99 제출과의 test prediction diff는 아래와 같습니다.

```text
               left                                                                  right  different_row_count  mean_abs_diff  max_abs_diff  prediction_correlation  left_pred_mean  left_pred_std  right_pred_mean  right_pred_std  left_endpoint_0_count  left_endpoint_1_count  right_endpoint_0_count  right_endpoint_1_count
baseline_sentinel99      rank1_FS8_no_BP_keep_BMI_metabolic__ENC4_no_ordinal_except_binary                  668       0.003203           0.1                0.999123        0.496717       0.197169         0.496493        0.198116                      4                      5                       4                       5
baseline_sentinel99 rank2_FS8_no_BP_keep_BMI_metabolic__ENC5_missing_explicit_for_edu_only                  668       0.003203           0.1                0.999123        0.496717       0.197169         0.496493        0.198116                      4                      5                       4                       5
baseline_sentinel99                    rank3_FS8_no_BP_keep_BMI_metabolic__ENC1_all_onehot                  669       0.003227           0.1                0.999110        0.496717       0.197169         0.496483        0.198070                      4                      5                       4                       5
```

## 5. 제출 판단
최종 판단: submission candidate.

CV 개선폭이 0.0005 이상이면 제출 후보, 0.001 이상이면 strong 후보로 봅니다. 개선폭이 0.0001~0.0003이면 micro 후보이며, prediction std 축소나 endpoint 약화가 있으면 제출을 보류합니다.

0.12999 진입 가능성은 CV 개선폭과 test prediction diff가 함께 충분할 때만 주장할 수 있습니다. diff가 지나치게 작으면 LB도 거의 같을 가능성이 높고, diff가 크지만 CV 개선이 작으면 과적합 가능성을 함께 표시해야 합니다.

## 6. PPT 반영 문장
- 추가 성능 개선을 위해 feature를 무작정 늘리기보다, RBF-SVR의 거리 구조를 고려해 파생변수와 categorical encoding을 재검토했습니다.
- RBF 기반 모델에서는 단순 feature 추가보다 feature representation이 중요하며, 불필요한 파생변수는 거리 구조를 흐릴 수 있습니다.
- 최종적으로 성능, 해석 가능성, 제출 안정성을 함께 고려해 기존 sentinel99 모델 유지 또는 개선 후보 1개만 추가 제출 대상으로 선정했습니다.
