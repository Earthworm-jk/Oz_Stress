# v8.2 mean_working cut and interaction deep dive

## 1. mean_working cut 조정 필요성
가장 y100 분리도가 큰 cut scheme은 `current_fine`입니다.

```text
             exp_id                   scheme  n_groups  min_group_count  eta2_y100  eta2_residual100  between_group_y100_range  between_group_residual_range
v82_20260618_103948             current_fine         8               77   0.040113          0.022954                 41.195569                     21.110008
v82_20260618_103948           coarse_extreme         5               77   0.039641          0.021845                 41.195569                     21.110008
v82_20260618_103948 low_6_mid_7_8_high_9plus         5              119   0.036823          0.021294                 33.446146                     18.353794
v82_20260618_103948              high_11plus         4              119   0.036595          0.021123                 33.446146                     18.353794
v82_20260618_103948              high_12plus         4               77   0.032373          0.015756                 41.195569                     21.110008
v82_20260618_103948            low_7_high_11         4              197   0.028368          0.018094                 22.236639                     13.172670
```

best scheme의 그룹별 평균:

```text
             exp_id       scheme mw_group_candidate  count  y100_mean  y100_std  residual100_mean  residual100_mae  pred100_mean
v82_20260618_103948 current_fine                 10    346  46.774566 29.453693          0.173410        13.809249     46.601156
v82_20260618_103948 current_fine                 11    120  59.641667 24.758598          8.358333        13.341667     51.283333
v82_20260618_103948 current_fine                  7    318  46.572327 29.112413         -1.106918        13.635220     47.679245
v82_20260618_103948 current_fine                  8    451  48.232816 30.218190          1.199557        14.702882     47.033259
v82_20260618_103948 current_fine                  9    537  46.109870 29.500333         -0.931099        13.847300     47.040968
v82_20260618_103948 current_fine                <=6    119  31.168067 18.231245         -8.226891        10.210084     39.394958
v82_20260618_103948 current_fine               >=12     77  72.363636 18.398304         12.883117        13.454545     59.480519
v82_20260618_103948 current_fine            missing   1032  49.121124 28.006128          0.796512        12.808140     48.324612
```

해석: 제출용 RBF는 raw sentinel numeric을 쓰고 있어서 cut을 직접 바꿀 필요는 크지 않습니다. 다만 해석용 score rule로는 `<=6`, `7~10/11`, `>=12`, `missing`을 분리하는 구조가 가장 자연스럽습니다.

## 2. mean_working x categorical interaction 후보
count >= 30이며 interaction excess가 큰 조합입니다.

```text
                factor mw_cut        factor_value  count  y100_mean  expected_additive_y100  interaction_excess_y100  residual100_mean
       medical_history     11         __MISSING__     54  66.259259               57.969396                 8.289863         13.574074
       medical_history     10       heart disease     69  53.652174               46.776133                 6.876041          3.057971
family_medical_history     11         __MISSING__     64  64.953125               58.518842                 6.434283         11.078125
         sleep_pattern      7        oversleeping     44  50.022727               43.682440                 6.340287         -0.840909
          smoke_status     11      current-smoker     31  66.677419               60.905707                 5.771712         14.451613
       medical_history      9            diabetes     93  53.268817               47.675526                 5.593291          4.247312
       medical_history   >=12         __MISSING__     34  75.970588               70.691366                 5.279223         17.617647
family_medical_history     10            diabetes     66  52.939394               47.768071                 5.171323          1.393939
family_medical_history      9       heart disease     80  51.362500               46.374197                 4.988303          1.125000
             edu_level     11 high school diploma     31  54.354839               59.067195                -4.712356          4.483871
family_medical_history     10 high blood pressure     68  53.426471               48.746983                 4.679487          1.779412
         sleep_pattern     10        oversleeping     45  48.444444               43.884680                 4.559765          8.066667
              activity     10             intense     73  42.657534               47.041566                -4.384032          0.698630
       medical_history      9       heart disease     76  41.973684               46.111437                -4.137752         -5.842105
             edu_level     10     graduate degree     62  51.177419               47.116745                 4.060675          1.435484
```

## 3. 숨은 score rule 후보 문장
```text
                                                                                      rule_text  count  y100_mean  residual100_mean
       if mean_working 11 and medical_history == __MISSING__ then interaction shift ~ 8.29 y100     54  66.259259         13.574074
     if mean_working 10 and medical_history == heart disease then interaction shift ~ 6.88 y100     69  53.652174          3.057971
if mean_working 11 and family_medical_history == __MISSING__ then interaction shift ~ 6.43 y100     64  64.953125         11.078125
         if mean_working 7 and sleep_pattern == oversleeping then interaction shift ~ 6.34 y100     44  50.022727         -0.840909
       if mean_working 11 and smoke_status == current-smoker then interaction shift ~ 5.77 y100     31  66.677419         14.451613
           if mean_working 9 and medical_history == diabetes then interaction shift ~ 5.59 y100     93  53.268817          4.247312
     if mean_working >=12 and medical_history == __MISSING__ then interaction shift ~ 5.28 y100     34  75.970588         17.617647
   if mean_working 10 and family_medical_history == diabetes then interaction shift ~ 5.17 y100     66  52.939394          1.393939
```

## 4. 결론
- mean_working cut은 제출 모델에서 직접 조정하기보다, RBF가 raw numeric sentinel 공간에서 암묵적으로 학습하도록 두는 편이 안전합니다.
- 해석 관점에서는 `<=6`은 낮은 stress score 군, `11`과 `>=12`는 높은 stress score 군으로 뚜렷하게 분리됩니다.
- interaction 후보는 `medical_history`, `family_medical_history`, `smoke_status`, `sleep_pattern`과 결합될 때 더 강해집니다.
- 따라서 숨은 생성식은 단일 mean_working 점수표가 아니라 mean_working 구간과 건강/수면/흡연/가족력 항목의 조합 score item이 섞인 구조일 가능성이 큽니다.
