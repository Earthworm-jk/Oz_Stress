# v5.3.1 final touch synthesis

## 1. v5.3 sentinel99 재현
- sentinel99 single 10-fold CV MAE는 0.134917입니다.
- 기존 v5.3 기준 CV 0.134917와 같은 수준으로 재현되었습니다.
- 기존 제출 파일과의 비교 상태: compared, mean_abs_diff=0.00000000, max_abs_diff=0.00000000.

## 2. seed ensemble 판단
- 5-seed OOF ensemble CV MAE는 0.135407입니다.
- single seed42 대비 개선폭은 -0.000490입니다.
- OOF pred std는 0.187468이고, single seed42 pred std 대비 변화량은 -0.008045입니다.
- 판단: not recommended; keep v5.3 sentinel99.

## 3. sentinel150 / sentinel999 비교
- best micro sentinel은 sentinel150이며 CV MAE는 0.134887입니다.
- 99, 150, 999는 모두 정상 근무시간 범위 밖으로 결측을 분리한다는 공통점이 있고, 성능 차이는 매우 작습니다.
- 따라서 99라는 숫자가 마법값이라기보다, mean_working 결측을 정상 근무시간 manifold 밖의 별도 상태로 두는 표현이 핵심입니다.

## 4. pairwise submission difference
       left       right  different_row_count  mean_abs_diff  max_abs_diff  prediction_correlation
 sentinel99 sentinel150                   18       0.000060          0.01                0.999992
 sentinel99 sentinel999                   29       0.000097          0.01                0.999988
sentinel150 sentinel999                   13       0.000043          0.01                0.999994

## 5. 왜 99999 같은 극단 sentinel은 쓰지 않는가
- 99/150/999만으로도 RobustScaler 이후 정상 관측 범위에서 충분히 멀리 분리됩니다.
- 더 극단적인 값은 RBF 거리 구조를 불필요하게 포화시킬 수 있고, LB에 맞춘 임의 보정처럼 보일 위험이 큽니다.
- 이번 최종 스토리는 out-of-range separation 가설이지 특정 거대 숫자 튜닝이 아닙니다.

## 6. 최종 제출 후보
- 1순위: `v531_single_sentinel99_reproduce.csv` 또는 기존 `v53_best_raw_rbf_B_mean_working_sentinel99.csv`.
- 2순위: seed ensemble이 CV에서 충분히 개선되면 `v531_sentinel99_seed5_ensemble.csv`; 개선이 작으면 보류.
- micro-gamble: sentinel150/999는 CV상 거의 같거나 아주 근소하게 좋지만, 설명 가능성은 sentinel150이 sentinel999보다 낫습니다.

## 7. PPT 설명 문장
- 최종 모델은 raw target RBF SVR이며, 0~1 범위의 0.01 단위 bounded grid target을 직접 학습한 뒤 clip/round2를 적용했습니다.
- S2 파생변수와 RobustScaler/OHE pipeline을 fold-safe하게 사용했고, test는 최종 predict에만 사용했습니다.
- mean_working 결측은 정상 근무시간 범위 밖 sentinel 99로 분리했습니다.
- 150/999도 유사한 결과를 보여 숫자 자체가 아니라 결측을 별도 latent state로 표현하는 것이 중요하다는 해석을 강화합니다.
- MLPRegressor도 비교했지만, 작은 tabular 데이터에서는 RBF SVR이 더 낮은 CV와 더 안정적인 분포를 보였습니다.
