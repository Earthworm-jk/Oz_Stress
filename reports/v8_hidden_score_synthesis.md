# v8 hidden score generation diagnostic

## 1. Target lattice
- y100 unique count: 101
- max grid deviation: 0.0000000000
- exact integer y100 rows: 3000 / 3000
- most frequent y100 score: 12 with count 44

해석: target이 0~100 정수 점수에서 0~1로 변환되었다는 가설을 직접 지지합니다.

## 2. Duplicate determinism
- duplicate groups: 6
- conflicting duplicate groups: 0
- max duplicate target range: 0.000000

동일 feature row에서 target이 흔들리면 hidden noise나 누락 변수가 있다는 뜻이고, 거의 흔들리지 않으면 deterministic formula 복원 가능성이 커집니다.

## 3. Linear transform probe
가장 좋은 선형 target transform은 `ridge_raw`이며 CV MAE는 0.250043입니다.

```text
        candidate  mean_mae  pred_std  endpoint_0_count  endpoint_1_count
        ridge_raw  0.250043  0.034386                 0                 0
       ridge_y100  0.250043  0.034386                 0                 0
ridge_arcsin_sqrt  0.250317  0.041325                 0                 0
ridge_rank_normal  0.251170  0.056712                 0                 0
      ridge_logit  0.251960  0.063097                 0                 0
```

선형 변환 후에도 RBF 수준까지 오지 못하면, 단순 선형식을 monotonic nonlinear transform한 구조만으로는 부족하다고 봅니다.

## 4. Local neighbor consistency
가장 좋은 kNN 후보는 `knn_k5`이며 MAE는 0.250443입니다.

```text
            exp_id candidate  k  mean_mae  raw_mae  pred_mean  pred_std  neighbor_target_std_mean  neighbor_target_std_median
v8_20260618_102726    knn_k5  5  0.250443 0.250399   0.473807  0.146531                  0.260651                    0.266139
v8_20260618_102726   knn_k10 10  0.251500 0.251480   0.475777  0.102615                  0.275680                    0.278214
v8_20260618_102726   knn_k20 20  0.250673 0.250695   0.478897  0.071270                  0.282719                    0.284683
```

kNN이 Ridge와 비슷하게 약하므로 단순한 근접 평균만으로는 hidden score surface를 복원하기 어렵습니다. RBF의 이득은 nearest-neighbor averaging보다 kernel interpolation과 representation 정리에서 나온 것으로 보입니다.

## 5. Integer residual fingerprint
OOF RBF residual100 rounded 상위 빈도:

```text
 residual100_rounded  count             exp_id
                   0   1337 v8_20260618_102726
                  -1     47 v8_20260618_102726
                   1     35 v8_20260618_102726
                   6     29 v8_20260618_102726
                  50     26 v8_20260618_102726
```

잔차가 정수 근처에 몰리면 0~100 점수와 rounding의 흔적으로 볼 수 있습니다. 특정 그룹에서 residual mean이 일정하면 숨은 offset item 후보입니다.

## 6. Surrogate formula hint
가장 fidelity가 좋은 surrogate는 `extra_trees_leaf5`이며 RBF OOF 예측과의 MAE는 0.060055, correlation은 0.968691입니다.

```text
            exp_id            surrogate  mae_to_rbf_oof  correlation_to_rbf_oof  r2_to_rbf_oof
v8_20260618_102726 decision_tree_depth4        0.124320                0.192805       0.037174
v8_20260618_102726    extra_trees_leaf5        0.060055                0.968691       0.804117
```

## 7. 종합 해석
- target grid는 0~100 점수체계 가설을 강하게 지지합니다.
- 다만 선형 target transform probe가 RBF에 근접하지 못하면, 단순 선형 점수를 비선형 변환한 문제라기보다는 구간/상호작용/부드러운 비선형 latent score일 가능성이 큽니다.
- kNN과 surrogate 결과는 hidden score surface가 feature space의 국소 구조를 가진다는 점을 확인하는 용도입니다.
- 다음으로 수식 복원을 더 밀고 싶다면 residual group offset과 surrogate feature importance 상위 항목을 중심으로 작은 rule/threshold 후보를 만들 수 있습니다.
