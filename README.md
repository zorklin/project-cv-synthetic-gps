# Поточний проєкт: synthetic GPS

## Мета

Формувати оцінку положення між рідкісними вимірюваннями Starlink/GPS на основі:

- оптичного потоку з направленої вниз тепловізійної камери;
- IMU;
- далекоміра/оцінки висоти;
- рідкісних абсолютних координат.

## Структура

### `notebooks/`

- `00_environment_and_data_check.ipynb` — коротка незалежна перевірка
  середовища та читання даних.
- `01_build_vision_velocity.ipynb` — відтворюваний preprocessing:
  thermal frames → sparse LK → фільтр pixel flow → gyro compensation →
  `/vision/velocity_frd` → новий derived MCAP. За замовчуванням запускає
  лише безпечний 15-секундний smoke test.
- `02_sparse_gps_fusion.ipynb` — поточний baseline: preflight → extract
  topics → calibration → GTSAM fixed-lag fusion. Математику цього notebook на
  етапі відтворення preprocessing не змінено; за замовчуванням він читає
  готовий teacher-derived bag.
- `03_fusion_diagnostics.ipynb` — read-only аналіз уже створених
  `fusion_v1` артефактів: похибки XY/висоти, Starlink outliers/stale fixes,
  timing і використання vision factors. GTSAM повторно не запускає.
- `opticalflow_sparse_noised_gps_fusion_v1(1).ipynb` — незмінний Colab-референс
  викладача з історичними outputs та upstream raw→LK→velocity експериментами.
  Локально його не слід запускати через `Run All`.

### `src/project_cv/`

- `vision_velocity.py` — модульні стадії preprocessing з режимами
  `smoke`, `full` та повторним використанням fingerprint-matching
  артефактів.
- `preprocessing_validation.py` — потокова read-only звірка з golden MCAP:
  topics, типи, counts, 18-польова flow-схема, timestamps, frame та числові
  значення.

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
`data/02_derived_with_velocity/`. Це immutable raw і teacher-provided golden
reference. Усі нові CSV/NPZ/JSON/PNG/MCAP зберігаються тільки під
`$PROJECT_CV_ARTIFACTS` у файловій системі WSL:

- `preprocess_smoke/` — короткий тест без фінального MCAP;
- `preprocess_v1/` — повний preprocessing і `generated_with_velocity/`;
- `fusion_v1/` — baseline fusion;
- `fusion_v1/diagnostics/` — read-only діагностика baseline.

## Локальне середовище

Робоче середовище винесене з notebook у `environment/`: WSL2, Ubuntu 22.04,
ROS2 Humble, Python 3.10, Jupyter та всі Python-залежності встановлюються й
перевіряються окремими ідемпотентними скриптами.

Для першої перевірки середовища запускай
`notebooks/00_environment_and_data_check.ipynb`. Усі робочі notebooks
використовують ядро `Project CV (ROS 2 Humble)`.

Після перезавантаження Windows запусти Jupyter із кореня проєкту:

```powershell
powershell -ExecutionPolicy Bypass -File .\environment\windows\jupyter.ps1 start
```

У Cursor обери `Existing Jupyter Server`, встав URL із команди й вибери
`Project CV (ROS 2 Humble)`.

## Порядок запуску

1. `00_environment_and_data_check.ipynb` — середовище та доступність даних.
2. `01_build_vision_velocity.ipynb` — спочатку `smoke`. Для повної
   побудови встанови `MODE = "full"` або перед запуском задай
   `PROJECT_CV_PREPROCESS_MODE=full`; для безпечного повторного відкриття
   готових артефактів використовуй `reuse`.
3. `02_sparse_gps_fusion.ipynb` — baseline fusion. До завершення golden
   validation він навмисно використовує teacher-derived bag.
4. `03_fusion_diagnostics.ipynb` — зафіксувати baseline-помилки до будь-яких
   змін висотного каналу чи factor graph.

Поточний preprocessing відтворює legacy-математику, включно з явно
задокументованими сумнівними припущеннями. Їх виправлення є наступним
експериментальним етапом і не змішується з перевіркою відтворюваності.
