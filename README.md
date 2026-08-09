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
- `04_height_experiments.ipynb` — відтворюваний, baseline-preserving етап
  дослідження висоти: будує causal barometer diagnostics для legacy та
  `raw_start_reset`, а з `RUN_GTSAM=False` безпечно повторно використовує
  завершений corrected run. Явний повний запуск дозволений лише в порожню
  output-папку; dense GPS використовується тільки для порівняльних метрик.
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
- `height_experiments.py` — causal median→mean→IIR, точне відтворення
  legacy barometer alignment, виправлений `raw-at-start/filter-reset` варіант
  і diagnostic-only оцінювання висоти у двох режимах часу. GTSAM цей модуль
  не запускає.
- `fusion_experiment_runner.py` — hash-locked runner: читає зафіксовану
  `gtsam-fusion` клітинку з `02`, застосовує одну контрольовану AST-зміну
  та пише кожен повний запуск у нову ізольовану папку. Його CLI окремо
  підтримує `legacy_replay`, `no_p0_realign` і `raw_start_reset`.

`src/project_cv/__pycache__/` — автоматично згенерований Python bytecode cache.
Він не є частиною алгоритму, безпечно перебудовується Python і вже ігнорується
через `.gitignore` разом із `*.pyc`.

### Контрольні файли в корені

- `SHA256SUMS` — зафіксовані SHA-256 immutable MCAP, metadata та calibration.
- `.gitignore` — не дозволяє випадково додати великі MCAP, artifacts, Jupyter
  checkpoints і Python cache до Git.

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
- `height_experiments_v1/baro_alignment_v1/raw_start_reset/gtsam/` — стабільний
  corrected run, який повторно використовує `04`; його diagnostic/comparison
  CSV/JSON/PNG лежать у сусідніх `baro_alignment_v1/diagnostic/` та
  `baro_alignment_v1/comparison/`;
- `height_experiments_v1/<UTC>_<variant>/gtsam/` — стандартний ізольований
  output CLI runner: fusion CSV/JSON, точний `baro_graph_input.csv`, manifest
  із hashes/параметрами та stdout log.

Dense GPS у нових height-метриках є diagnostic-only reference і не додається
як factor. Водночас перші експерименти навмисно успадковують baseline
origin/calibration/p0/v0; deployment-pure варіант буде окремим етапом.

### Перевірка immutable входів

Після копіювання даних або перед серією експериментів перевір hashes у WSL:

```bash
source /home/fedor/project_cv_runtime/paths.env
cd "$PROJECT_CV_SOURCE"
sha256sum --check SHA256SUMS
```

Усі рядки мають завершитися `OK`. Metadata `02_sparse_gps_fusion.ipynb`
окремо зберігає історичний `source_sha256_at_migration` і поточний
`tracked_reference_sha256`, тому ці два різні стани більше не плутаються.

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
5. `04_height_experiments.ipynb` — залишити `RUN_GTSAM=False`, побудувати
   barometer-only diagnostics і звірити вже завершений `raw_start_reset`.
   Встановлювати `RUN_GTSAM=True` лише для явного запуску corrected variant
   у порожню output-папку; dry-run трьох CLI-варіантів виконується окремо
   через `fusion_experiment_runner.py`.

Поточний preprocessing відтворює legacy-математику, включно з явно
задокументованими сумнівними припущеннями. Їх виправлення є наступним
експериментальним етапом і не змішується з перевіркою відтворюваності.
