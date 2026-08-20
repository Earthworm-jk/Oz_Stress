# v8.1 hidden score rule deep dive

## 1. Mean-working conditional target lattice
```text
group_value  count  y100_mean  y100_std  y100_min  y100_max  unique_y100_count  top_y100  top_y100_share
        <=6    119  31.168067 18.231245         0        77                 49        59        0.067227
          9    537  46.109870 29.500333         0       100                101        39        0.018622
          7    318  46.572327 29.112413         0       100                 97        13        0.025157
         10    346  46.774566 29.453693         0       100                 97        76        0.023121
          8    451  48.232816 30.218190         0       100                 98        96        0.022173
    missing   1032  49.121124 28.006128         0       100                101        55        0.018411
         11    120  59.641667 24.758598         1       100                 62        72        0.050000
       >=12     77  72.363636 18.398304        40       100                 44        54        0.064935
```

해석: `<=6`, `11`, `>=12`가 평균 y100에서 뚜렷하게 분리되면 근무시간 구간형 score item 가능성이 살아납니다.

## 2. Residual score-rule hints
아래는 RBF OOF가 남긴 residual100 평균이 ±5점 이상이고 표본 수가 50개 이상인 그룹입니다.

```text
          rule_col rule_value  count  residual100_mean  residual100_mae  t_like_residual_mean
    endpoint_group  0.97~0.99     90         29.666667        29.666667             11.610052
mean_working_group       >=12     77         12.883117        13.454545              6.601431
mean_working_group         11    120          8.358333        13.341667              4.761997
mean_working_group        <=6    119         -8.226891        10.210084             -5.906629
    endpoint_group  0.01~0.03    101        -22.821782        22.821782             -9.765512
```

해석: 같은 방향의 residual이 큰 그룹은 RBF가 smooth하게 평균화했지만 실제 생성식에는 discrete bonus/penalty가 있을 가능성이 있습니다.

## 3. Fold-safe residual offset check
OOF residual을 다른 fold에서만 학습한 그룹 offset으로 보정했을 때 가장 큰 개선은 `edu_level`입니다.

```text
             exp_id              group_col  base_mae  base_round2_mae  adjusted_round2_mae  improvement_vs_base_round2  adjusted_pred_mean  adjusted_pred_std  adjusted_endpoint_0_count  adjusted_endpoint_1_count
v81_20260618_103517              edu_level   0.13417          0.13417             0.136503                   -0.002333            0.480247           0.196645                          9                         12
v81_20260618_103517          sleep_pattern   0.13417          0.13417             0.136640                   -0.002470            0.482417           0.196588                          6                         14
v81_20260618_103517               activity   0.13417          0.13417             0.137277                   -0.003107            0.482593           0.196440                         11                         16
v81_20260618_103517           smoke_status   0.13417          0.13417             0.137447                   -0.003277            0.482070           0.196462                         12                         16
v81_20260618_103517 family_medical_history   0.13417          0.13417             0.137707                   -0.003537            0.484530           0.196419                          2                         21
v81_20260618_103517        medical_history   0.13417          0.13417             0.138807                   -0.004637            0.483583           0.196505                          7                         16
v81_20260618_103517     mean_working_group   0.13417          0.13417             0.139010                   -0.004840            0.482000           0.201660                         18                         23
```

이 실험은 제출용 보정이 아니라 hidden score item 검증입니다. 개선이 있으면 해당 그룹 축에 아직 설명되지 않은 offset 구조가 있다는 뜻입니다.

## 4. Interaction excess
단일 marginal 평균으로 설명되지 않는 interaction excess가 큰 조합입니다.

```text
              left                  right left_value         right_value  count  y100_mean  expected_additive_y100  interaction_excess_y100
mean_working_group        medical_history         11         __MISSING__     54  66.259259               57.969396                 8.289863
mean_working_group        medical_history         10       heart disease     69  53.652174               46.776133                 6.876041
mean_working_group family_medical_history         11         __MISSING__     64  64.953125               58.518842                 6.434283
mean_working_group          sleep_pattern          7        oversleeping     44  50.022727               43.682440                 6.340287
mean_working_group           smoke_status         11      current-smoker     31  66.677419               60.905707                 5.771712
mean_working_group        medical_history          9            diabetes     93  53.268817               47.675526                 5.593291
mean_working_group        medical_history       >=12         __MISSING__     34  75.970588               70.691366                 5.279223
mean_working_group family_medical_history         10            diabetes     66  52.939394               47.768071                 5.171323
mean_working_group family_medical_history          9       heart disease     80  51.362500               46.374197                 4.988303
mean_working_group family_medical_history         10 high blood pressure     68  53.426471               48.746983                 4.679487
mean_working_group          sleep_pattern         10        oversleeping     45  48.444444               43.884680                 4.559765
mean_working_group               activity         10             intense     73  42.657534               47.041566                -4.384032
```

해석: excess가 크고 표본 수가 충분하면 구간/상호작용 score rule 후보입니다.

## 5. Shallow rule surrogate
가장 좋은 shallow tree surrogate는 `tree_depth3_leaf40`이며 RBF OOF와의 MAE는 0.123448, correlation은 0.168422입니다.

```text
             exp_id          surrogate  depth  mae_to_rbf_oof  correlation_to_rbf_oof  r2_to_rbf_oof
v81_20260618_103517 tree_depth2_leaf40      2        0.124134                0.127291       0.016203
v81_20260618_103517 tree_depth3_leaf40      3        0.123448                0.168422       0.028366
v81_20260618_103517 tree_depth4_leaf40      4        0.124320                0.192805       0.037174
v81_20260618_103517 tree_depth5_leaf40      5        0.127035                0.232851       0.054220
```

## 6. 종합
- 0~100 점수체계 가설은 유지됩니다.
- 단순 선형 변환 가설보다는, mean_working 극단부와 일부 범주 조합이 discrete score item으로 들어간 구간/상호작용형 생성식이 더 그럴듯합니다.
- 다만 residual offset은 OOF 기반 해석 도구이지 그대로 제출 보정으로 쓰면 과적합 위험이 있습니다.
- 성능이 아니라 생성식 흔적을 찾는 목적이라면 `mean_working_group`, `smoke_status`, `medical_history`, `family_medical_history`의 residual offset과 interaction excess가 가장 먼저 볼 축입니다.
