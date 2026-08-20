# v8.3 final suspicion check: tree/boost/rule models

## 1. y100 직접 모델 비교
```text
                   candidate              family  add_rules  mean_mae  pred_std  pred_min  pred_max
           extra_trees_leaf5       ensemble_tree      False  0.210857  0.094936      0.21      0.76
         random_forest_leaf5       ensemble_tree      False  0.212110  0.087895      0.22      0.74
     extra_trees_leaf5_rules ensemble_tree_rules       True  0.212640  0.111701      0.14      0.89
      hist_gradient_boosting            boosting      False  0.219760  0.099526      0.18      0.86
hist_gradient_boosting_rules      boosting_rules       True  0.219957  0.102598      0.21      0.92
  decision_tree_depth6_rules          tree_rules       True  0.242857  0.075477      0.24      0.80
    ridge_y100_rule_features        linear_rules       True  0.243587  0.070020      0.23      0.79
        decision_tree_depth4                tree      False  0.245010  0.068126      0.26      0.73
        decision_tree_depth6                tree      False  0.245343  0.089898      0.24      0.80
        decision_tree_depth8                tree      False  0.247067  0.122099      0.11      0.91
             ridge_y100_base              linear      False  0.250043  0.034386      0.33      0.59
```

가장 좋은 해석형 계열 후보는 `extra_trees_leaf5`이며 CV MAE는 0.210857입니다.
RBF의 0.13417에는 못 미치지만, tree/boost 계열이 Ridge보다 확실히 낫다면 단순 선형식보다 threshold/interaction 구조가 더 그럴듯합니다.

## 2. v8.2 rule 후보를 명시적으로 넣었을 때
```text
             exp_id         base_candidate               rule_candidate  base_mae  rule_mae  improvement_from_rules
v83_20260618_104526        ridge_y100_base     ridge_y100_rule_features  0.250043  0.243587                0.006457
v83_20260618_104526   decision_tree_depth6   decision_tree_depth6_rules  0.245343  0.242857                0.002487
v83_20260618_104526      extra_trees_leaf5      extra_trees_leaf5_rules  0.210857  0.212640               -0.001783
v83_20260618_104526 hist_gradient_boosting hist_gradient_boosting_rules  0.219760  0.219957               -0.000197
```

가장 큰 rule feature 개선은 `ridge_y100_rule_features`이며 개선폭은 0.006457입니다.
개선이 작거나 음수이면, v8.2 rule 후보는 해석 신호이지만 단순 binary feature 몇 개만으로 성능을 끌어올리는 구조는 아닙니다.

## 3. 실제 y100 ExtraTrees 중요도 상위
```text
                                feature  importance
          num__diastolic_blood_pressure    0.049958
                               num__bmi    0.049052
                            num__height    0.048023
                       num__cholesterol    0.046790
           num__systolic_blood_pressure    0.046236
                            num__weight    0.046083
       num__cholesterol_glucose_product    0.045068
         num__glucose_cholesterol_ratio    0.043739
                               num__age    0.043071
                      num__bone_density    0.042733
                           num__glucose    0.042696
                       num__gender_code    0.029249
                      num__mean_working    0.026514
             cat__activity_cat_moderate    0.025390
 cat__edu_level_cat_high school diploma    0.024487
              cat__activity_cat_intense    0.024201
                    num__mw_high_12plus    0.024152
cat__sleep_pattern_cat_sleep difficulty    0.023432
                cat__activity_cat_light    0.023425
          cat__sleep_pattern_cat_normal    0.023353
```

명시 rule/tail feature 중요도:

```text
                         feature  importance
             num__mw_high_12plus    0.024152
           num__mw_low_6_or_less    0.018232
   num__rule_mw11_family_missing    0.008464
      num__rule_mw9_med_diabetes    0.002946
        num__rule_mw10_med_heart    0.002503
  num__rule_mw10_family_diabetes    0.001969
      num__rule_mw11_med_missing    0.001808
num__rule_mw7_sleep_oversleeping    0.001526
                 num__mw_high_11    0.001349
  num__rule_mw12plus_med_missing    0.000911
   num__rule_mw11_current_smoker    0.000155
```

## 4. RBF surrogate
RBF OOF를 가장 잘 흉내낸 surrogate는 `rbf_surrogate_extra_trees_leaf3`이며 MAE는 0.035088, correlation은 0.990454입니다.

```text
             exp_id                       surrogate  mae_to_rbf  corr_to_rbf  r2_to_rbf
v83_20260618_104526       rbf_surrogate_tree_depth4    0.123583     0.210935   0.044494
v83_20260618_104526       rbf_surrogate_tree_depth6    0.125058     0.263954   0.069672
v83_20260618_104526       rbf_surrogate_tree_depth8    0.125993     0.344574   0.118731
v83_20260618_104526 rbf_surrogate_extra_trees_leaf3    0.035088     0.990454   0.933020
```

RBF surrogate ExtraTrees 중요도 상위:

```text
                               feature  importance
         num__diastolic_blood_pressure    0.054824
          num__systolic_blood_pressure    0.053253
                              num__bmi    0.052588
                           num__weight    0.050730
                      num__cholesterol    0.049858
      num__cholesterol_glucose_product    0.049610
                     num__bone_density    0.048143
                           num__height    0.046409
                          num__glucose    0.044420
                              num__age    0.043140
        num__glucose_cholesterol_ratio    0.042631
                      num__gender_code    0.030496
                     num__mean_working    0.029311
cat__edu_level_cat_high school diploma    0.028653
            cat__activity_cat_moderate    0.027714
   cat__edu_level_cat_bachelors degree    0.025795
             cat__activity_cat_intense    0.024217
               cat__activity_cat_light    0.023732
         cat__sleep_pattern_cat_normal    0.023272
            cat__edu_level_cat_Unknown    0.022752
```

## 5. 최종 판단
- 0~100 점수식 의심은 유지됩니다.
- 단순 선형식은 Ridge/target transform 계열이 약해서 가능성이 낮습니다.
- tree/boost가 Ridge보다 낫고, RBF surrogate를 ensemble tree가 잘 흉내내면 구간/상호작용/비선형 score surface 가설이 강화됩니다.
- 하지만 명시 rule feature 몇 개만으로 큰 개선이 없다면, 사람이 한두 줄로 쓸 수 있는 단순 rule list가 아니라 많은 약한 threshold와 interaction이 합쳐진 복합 생성식일 가능성이 큽니다.
- 따라서 제출 모델에서 RBF가 암묵적으로 먹은 것은 숨은 점수식의 일부 흔적이며, 이를 완전히 손으로 복원하기는 어렵다는 쪽으로 의심을 내려놓는 것이 합리적입니다.
