# Поточний проєкт: synthetic GPS

## Мета

Формувати оцінку положення між рідкісними вимірюваннями Starlink/GPS на основі:

- оптичного потоку з направленої вниз тепловізійної камери;
- IMU;
- далекоміра/оцінки висоти;
- рідкісних абсолютних координат.

## Структура

### `notebooks/`

- `01_sparse_gps_fusion_local.ipynb` — поточний структурований локальний
  pipeline: preflight → extract topics → calibration → GTSAM fusion. Він
  використовує готовий derived bag із `/vision/velocity_frd`, ядро
  `Project CV (ROS 2 Humble)` і записує результати тільки в WSL artifacts.
- `00_environment_and_data_check.ipynb` — коротка незалежна перевірка
  середовища та читання даних.
- `opticalflow_sparse_noised_gps_fusion_v1(1).ipynb` — незмінний Colab-референс
  викладача з історичними outputs та upstream raw→LK→velocity експериментами.
  Локально його не слід запускати через `Run All`.

### `calibration/`

- `thermal_5_9x7_30mm_384x288_20260123_102051.yml` — внутрішня калібровка тепловізійної камери 384×288.
- `сam_to_imu_rot_mtrx.yml` — поворот між системами координат камери та IMU і часовий параметр `td`.

У назві другого файла перша літера `с` є кириличною. Назву збережено без змін, щоб не порушити наявні посилання у блокноті.

### `data/01_raw_k2r/`

- `K2R00005_20260607_194949_0.mcap` — первинний запис, який слід вважати незмінним джерелом даних.
- `metadata.yaml` — метадані саме цього запису.

### `data/02_derived_with_velocity/`

- `K2R00005_20260607_194949_with_velocity.mcap` — похідна копія запису, до якої вже додано результати оптичного потоку та оцінку швидкості.
- `metadata_velocity.yaml` — метадані похідного запису.

Похідний MCAP зручний як контрольна точка для швидкого продовження роботи, але його потрібно вміти відтворити з сирого MCAP.

## Робоче правило

Не змінювати й не перезаписувати вміст `data/01_raw_k2r/` та
`data/02_derived_with_velocity/`. Поточний fusion notebook читає derived bag,
а всі відтворювані CSV/JSON результати зберігає поза source data в
`$PROJECT_CV_ARTIFACTS/fusion_v1`.

## Локальне середовище

Робоче середовище винесене з notebook у `environment/`: WSL2, Ubuntu 22.04,
ROS2 Humble, Python 3.10, Jupyter та всі Python-залежності встановлюються й
перевіряються окремими ідемпотентними скриптами.

Для першої перевірки середовища запускай
`notebooks/00_environment_and_data_check.ipynb`; для подальшої роботи —
`notebooks/01_sparse_gps_fusion_local.ipynb`. Обидва використовують ядро
`Project CV (ROS 2 Humble)`.

Окремий відтворюваний preprocessing pipeline raw→LK→gyro compensation→velocity
ще не винесено з Colab-референсу. Він не потрібен для поточного запуску, бо
наданий derived MCAP уже містить необхідний velocity topic.
