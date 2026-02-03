# TF CSV 시간순 분석 요약

기준: `tf_csv` 폴더의 이벤트 CSV를 `wall_time` 기준으로 정렬해 비교.

## 시간순 이벤트 요약

| 시간 | 파일 | 측정길이(s) | odom->base_link JUMP | map->odom JUMP | MISSING | STALE | odom JUMP/분 | map JUMP/분 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 09:13:44 | tf_jump.csv | 196.6 | 0 | 0 | 0 | 88 | 0.00 | 0.00 |
| 09:28:23 | tf_jump_005.csv | 1.0 | 0 | 0 | 4 | 0 | 0.00 | 0.00 |
| 10:00:24 | tf_jump_sens.csv | 252.4 | 81 | 23 | 1 | 7 | 19.25 | 5.47 |
| 10:19:39 | tf_jump_sens_run2.csv | 261.1 | 16 | 29 | 1 | 8 | 3.68 | 6.66 |
| 10:43:45 | tf_jump_after_tuning.csv | 162.9 | 134 | 15 | 0 | 21 | 49.35 | 5.52 |
| 11:03:28 | tf_jump_sensitive_after_rtabmap_tune.csv | 251.9 | 12 | 0 | 6 | 6 | 2.86 | 0.00 |
| 13:10:05 | tf_jump_after_reg2.csv | 128.6 | 1 | 0 | 0 | 1 | 0.47 | 0.00 |
| 14:10:52 | tf_jump_final_sensitive.csv | 182.9 | 60 | 0 | 4 | 8 | 19.68 | 0.00 |
| 14:29:19 | tf_jump_after_ekf_xy.csv | 195.4 | 53 | 0 | 5 | 0 | 16.27 | 0.00 |
| 14:42:34 | tf_jump_after_vx_imu_split.csv | 198.3 | 11 | 0 | 1 | 2 | 3.33 | 0.00 |

## 개선 포인트(핵심)

1. 초기에는 `STALE` 중심(시간 정합 문제)에서 시작했으나, 이후 시간동기/TF 조정으로 크게 감소.
2. RTAB-Map 정합 보수화 이후 `map->odom JUMP`가 0으로 안정화됨.
3. EKF를 `wheel(vx)` + `IMU(yaw,vyaw)` 분리로 되돌린 뒤 `odom->base_link JUMP`가 `53 -> 11`로 개선.
4. `MISSING 1`은 시작 시점 transient(0.00s)로 확인된 케이스가 있어, 주행 중 반복 여부로 판단 필요.

## 현재 권장 고정 상태

- 시간동기: `lidar_offset_sec=0.036`
- RTAB-Map: 보수 정합(`Reg/Strategy=2`, `Reg/Force3DoF=true`, `LoopThr=0.25`, `Vis/MinInliers=20`, `OptimizeMaxError=0.3`)
- EKF: wheel pose/yaw 비활성 + `vx`만 사용, 회전은 IMU(`yaw`,`vyaw`) 사용

