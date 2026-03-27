# ТЗ `module_target_predict` v1

## 1. Назначение модуля

`module_target_predict` отвечает за применение обученной модели к новым большим растрами и за получение геопривязанных растровых предсказаний, пригодных для последующего постпроцессинга и векторизации.

Модуль должен быть production-like: он не просто «прогоняет сеть по картинке», а воспроизводит train-time контракт признаков, корректно обрабатывает большие GeoTIFF, сохраняет геопривязку, учитывает NoData/valid и экспортирует предсказания в стабильном формате для `module_postprocess_vectorize`.

## 2. Границы ответственности

### 2.1. Модуль обязан

* читать обученный checkpoint и его metadata;
* читать новый GeoTIFF произвольного размера;
* восстанавливать feature contract, с которым модель была обучена;
* восстанавливать не только dataset-side feature mode, но и финальный model input contract;
* использовать checkpoint metadata так, чтобы predict pipeline не требовал ручных догадок о числе каналов, роли valid-mask и порядке сборки входного тензора.
* при необходимости автоматически строить `raw8_idx3` из исходного 8-band растра;
* выполнять tiled inference с overlap и blending;
* корректно учитывать `valid` / NoData;
* сохранять геопривязанные raster outputs и manifest предикта;
* экспортировать данные, пригодные для постпроцессинга.

### 2.2. Модуль не обязан

* обучать модель;
* изменять checkpoint;
* выполнять финальный постпроцессинг и polygonization;
* вычислять полный набор object-level итоговых метрик.

## 3. Входные данные

### 3.1. Обязательные входы

1. `checkpoint` обученной модели.
2. `checkpoint metadata` / `config_used` / `train manifest`.
3. Новый входной GeoTIFF:

   * ожидаемый основной кейс: 8-band raster;
   * произвольный spatial extent;
   * произвольные исходные dtype / NoData / CRS / tiling.

### 3.2. Опциональные входы

* AOI;
* predict config overrides;
* batch list нескольких входных растров.

## 4. Основной принцип работы

### 4.1. Checkpoint-driven inference

Модуль не должен полагаться на ручное задание критически важных параметров.

Основной источник истины для инференса:

* metadata checkpoint;
* `config_used.yaml` из train run;
* exported model manifest.

### 4.2. Что должно восстанавливаться из checkpoint metadata

* `feature_mode` (`raw8` или `raw8_idx3`);
* `in_channels`;
* архитектурный тип модели;
* схема нормализации;
* статистики нормализации;
* параметры target heads;
* ожидаемый patch/tile size (если он зафиксирован как train baseline);
* версия модели.
* channel semantics;
* признак использования `valid` как дополнительного входного канала;
* финальный assembled model input contract;
* runtime-параметры, критичные для безопасного predict-time восстановления train contract.

## 5. Входной feature contract

### 5.1. Поддерживаемые режимы

* `raw8`
* `raw8_idx3`
В рамках `module_target_predict` эти режимы рассматриваются как dataset-side feature modes, которые должны быть восстановлены из исходного 8-band raster.

Финальный model input baseline v1 собирается с обязательным учетом слоя `valid`:

* `raw8_valid = raw8 + valid`
* `raw8_idx3_valid = raw8_idx3 + valid`

### 5.2. Автоматический feature adapter

Если checkpoint ожидает:

* `8` каналов → используется `raw8`;
* `11` каналов → модуль обязан автоматически построить `raw8_idx3` из исходного 8-band растра.
Указанные выше значения относятся к числу feature-каналов до добавления `valid`.

Финальный assembled model input contract baseline v1:

* checkpoint ожидает `9` каналов → используется `raw8_valid`
* checkpoint ожидает `12` каналов → используется `raw8_idx3_valid`

Соответственно, модуль обязан:

* восстановить dataset-side feature stack (`raw8` или `raw8_idx3`);
* отдельно восстановить `valid`;
* собрать финальный model input tensor в соответствии с checkpoint metadata.

### 5.3. Зафиксированный состав `raw8_idx3`

* NDVI
* SAVI
* NDWI

### 5.4. Запрещено

* использовать feature mode, не соответствующий checkpoint metadata;
* тихо переставлять каналы или подменять band order без явной проверки.

## 6. Политика пространственной обработки

### 6.1. Общие принципы

* оригинальный растр не переписывается;
* чтение должно быть window-based / tiled;
* геопривязка должна сохраняться в outputs;
* spatial alignment между входом и выходом должен быть 1:1 по grid, если архитектура не меняет финальное разрешение.

### 6.2. AOI policy

Если AOI предоставлен:

* модуль должен уметь ограничивать обработку AOI+buffer областью;
* буфер должен быть совместим с общесистемной AOI policy;
* итоговые outputs могут быть сохранены либо в полном raster extent, либо в AOI-limited extent — это должно задаваться конфигом и отражаться в manifest.

Если AOI отсутствует:

* модуль должен обрабатывать весь raster.

## 7. NoData / valid policy

### 7.1. Общий принцип

Инференс обязан использовать ту же valid/nodata логику, что и `module_prep_data`.

### 7.2. Источники valid-mask

Приоритет:

1. dataset mask / GDAL valid data mask;
2. nodata metadata;
3. predict config override;
4. явная ошибка, если корректная интерпретация невозможна.

### 7.3. Использование valid-mask

`valid` должен использоваться:

* до нормализации;
* при вычислении derived indices;
* для фильтрации полностью invalid tiles;
* для подавления outputs в invalid областях;
* для экспорта итогового `valid.tif` рядом с предсказаниями.
Кроме служебной роли valid-mask, `valid` в baseline v1 должен рассматриваться и как дополнительный входной канал модели, если это следует из checkpoint metadata.

Модуль не должен терять эту двойную семантику при window reading, feature assembly, normalization и model forward.

### 7.4. Запрещено

* silently treat nodata as real background;
* использовать invalid пиксели при расчёте нормализации или индексов.

## 8. Нормализация и препроцессинг

### 8.1. Основной принцип

Predict-time препроцессинг должен быть идентичен train-time контракту.
Это требование относится не только к нормализации и derived features, но и к финальной сборке model input tensor, включая обязательное добавление `valid`, если оно использовалось при обучении.

### 8.2. Зафиксированная baseline-схема

Для v1 принимается:

* `valid` определяется до числовых преобразований;
* данные приводятся к `float32`;
* применяется `per-band robust percentile normalization` по статистикам train split, сохранённым в checkpoint metadata / manifest;
* clipping по `p0.5/p99.5`;
* масштабирование в `[0,1]`.

### 8.3. Важное правило

На predict нельзя пересчитывать новые «свои» статистики нормализации вместо train statistics, если только это не включено как отдельный experimental mode.

### 8.4. Финальная сборка model input

После восстановления feature stack и valid-mask модуль должен собрать финальный model input tensor в точном соответствии с checkpoint metadata.

Для baseline v1 принимается:

* `raw8_valid = raw8 + valid`
* `raw8_idx3_valid = raw8_idx3 + valid`

Любое несоответствие между:

* `feature_mode`,
* `in_channels`,
* `channel semantics`,
* фактически собранным input tensor

должно приводить к явной ошибке, а не к молчаливому fallback.

## 9. Стратегия tiled inference

### 9.1. Общие требования

* большие растры должны обрабатываться по окнам;
* tile size должен быть конфигурируемым;
* baseline должен быть совместим с train patch size.

### 9.2. Зафиксированный baseline v1

* baseline tile size: `512`;
* baseline overlap: `25%`;
* модуль должен поддерживать альтернативные tile sizes, если они совместимы с checkpoint и конфигом.

### 9.3. Padding policy

Если размеры растра не делятся на tile size/stride:

* модуль должен корректно обрабатывать border tiles;
* паддинг должен быть явным и отражённым в коде;
* итоговый output должен быть cropped обратно к исходному spatial extent.

## 10. Blending policy

### 10.1. Обязательное требование

Предсказания перекрывающихся окон не должны склеиваться простым overwrite.

### 10.2. Зафиксированный baseline v1

Для overlapping tiles по умолчанию принимается:

* `Gaussian blending`.

### 10.3. Обоснование

* Gaussian blending снижает влияние предсказаний у краёв окна;
* центральные области окна получают больший вес;
* это уменьшает seam artifacts по сравнению с равномерным смешиванием.

### 10.4. Допустимый fallback

`constant/equal weighting averaging` допускается только как fallback/ablation mode.

## 11. Tile filtering policy

### 11.1. Основная идея

Модуль должен уметь пропускать окна, не несущие полезной информации.

### 11.2. Обязательный baseline

* полностью invalid tiles не прогоняются через модель;
* их outputs формируются напрямую как invalid/zero-confidence.

### 11.3. Дополнительные режимы

Допускается конфигурируемый пропуск окон с `valid_ratio` ниже порога, но для v1 этот режим не должен быть default без отдельной проверки.

## 12. Модельный forward contract

### 12.1. Требования

Модуль должен поддерживать multitask outputs:

* `extent logits/probabilities`
* `boundary logits/probabilities`
* `distance prediction`
Если checkpoint metadata описывает multitask heads и их порядок, predict pipeline обязан валидировать соответствие фактического model output ожидаемому контракту до начала сохранения raster outputs.

### 12.2. Совместимость

Если модель имеет deep supervision outputs, predict pipeline должен использовать только финальные heads, а не промежуточные outputs.

## 13. Постобработка внутри predict-модуля

### 13.1. Что допускается

Внутри `module_target_predict` допускается только минимальная техническая постобработка, необходимая для корректного сохранения raster predictions:

* sigmoid / softmax;
* mask-based suppression;
* blending;
* cropping to original extent.

### 13.2. Что запрещено

* thresholding extent/boundary как final decision;
* skeletonization;
* breakpoint connection;
* watershed;
* polygonization.

Это относится к следующему модулю.

## 14. Выходные данные

### 14.1. Основные raster outputs

На каждый входной raster должны экспортироваться как минимум:

* `extent_prob.tif`
* `boundary_prob.tif`
* `distance_pred.tif`
* `valid.tif`

### 14.2. Дополнительные полезные outputs

По умолчанию рекомендуется сохранять также:

* `extent_logit.tif` (optional)
* `boundary_logit.tif` (optional)
* `preview_rgb.*`
* `preview_overlay.*`

### 14.3. Формат

Для v1 основным форматом выходных растров принимается GeoTIFF.

## 15. Manifest предикта

### 15.1. Модуль обязан сохранять

* `predict_manifest.json`
* `config_used.yaml`
* `summary.json`
* лог выполнения

### 15.2. Что должно быть в `predict_manifest`

* путь к checkpoint;
* model version;
* feature_mode;
* in_channels;
* normalization stats reference;
* tile size;
* overlap;
* blending mode;
* AOI usage;
* input raster metadata summary;
* output raster paths;
* timing / runtime summary;
* valid coverage summary.
* channel semantics;
* факт использования `valid` как входного канала;
* resolved model input channel count;
* device requested / device resolved;
* amp requested / amp used;
* tile batch size;
* OOM fallbacks applied;
* assembled input contract (`raw8_valid` или `raw8_idx3_valid`).

## 16. Batch mode

### 16.1. Требования

Модуль должен поддерживать:

* single-raster mode;
* batch mode по директории / списку файлов.

### 16.2. Batch manifest

При batch запуске должен формироваться общий summary по всем входам.

## 17. Производительность и память

### 17.1. Требования

* модуль должен читать растр по окнам, а не грузить весь raster в GPU;
* CPU/GPU memory usage должна быть контролируемой;
* batch size по tiles должен быть конфигурируемым.

### 17.2. Baseline strategy

При нехватке памяти:

* сначала уменьшается tile batch size;
* затем допускается уменьшение tile size;
* ухудшение feature contract или отключение valid-aware logic не допускается.

### 17.3. Hardware-adaptive runtime policy

Модуль должен автоматически адаптироваться к доступной среде выполнения.

Базовый приоритет устройств:

* `CUDA -> MPS -> CPU`

Обязательные правила:

* AMP включается автоматически там, где это безопасно и поддерживается средой;
* при нехватке памяти модуль должен сначала деградировать по runtime-параметрам;
* порядок baseline-реакции на OOM:
  1. уменьшение tile batch size;
  2. уменьшение tile size;
  3. переход к более медленному, но корректному режиму исполнения.
* ради экономии памяти запрещено:
  * ломать train/predict feature contract;
  * убирать `valid` из assembled model input, если checkpoint ожидает его наличие;
  * отключать valid-aware suppression;
  * пересчитывать свои normalization stats вместо train-derived statistics.

## 18. Optional quality modes

### 18.1. Допускаемые режимы

После baseline допускаются optional режимы:

* TTA (`hflip`, `vflip`, `rot90`) ;
* ensemble of checkpoints;
* averaging logits from multiple stochastic runs, если архитектура это поддерживает.

### 18.2. Статус для v1

Эти режимы не входят в обязательный baseline, но должны проектно не конфликтовать с модулем.

## 19. Ошибки и диагностика

Модуль обязан завершаться с явной ошибкой, если:

* checkpoint metadata неполна или несовместима;
* число каналов входа не позволяет построить ожидаемый feature mode;
* невозможно безопасно восстановить valid/nodata policy;
* normalization stats отсутствуют при обязательном их использовании;
* spatial metadata входного raster повреждена или недостаточна для корректного экспорта outputs.
* checkpoint metadata не позволяет однозначно восстановить assembled model input contract;
* отсутствует достаточная информация о channel semantics;
* невозможно корректно добавить `valid` как входной канал при ожидаемом model contract;
* assembled input tensor не согласован с `in_channels`.

## 20. Критерии приемки модуля

Модуль считается готовым, если:

1. Может применить обученный checkpoint к новому 8-band GeoTIFF без ручного переписывания feature pipeline.
2. Корректно восстанавливает `raw8` или `raw8_idx3` по metadata checkpoint.
3. Выполняет tiled inference с overlap и blending без seam-like артефактов как baseline-поведения.
4. Не использует invalid/nodata как полноценный фон.
5. Экспортирует геопривязанные raster outputs для следующего модуля.
6. Сохраняет полный predict manifest и достаточно диагностики для forensic-проверки.
7. Модуль корректно собирает финальный model input contract (`raw8_valid` или `raw8_idx3_valid`) по metadata checkpoint без ручного вмешательства.
8. Runtime-поведение остается корректным на разном железе за счет hardware-adaptive policy без нарушения valid-aware и feature-contract логики.

## 21. Зафиксированные baseline-решения v1

1. Основной режим: `checkpoint-driven inference`.
2. Default tile size: `512`.
3. Default overlap: `25%`.
4. Default blending: `Gaussian blending`.
5. Default output set: `extent_prob`, `boundary_prob`, `distance_pred`, `valid`.
6. Normalization: строго train-derived robust stats.
7. Feature adaptation: автоматический `raw8` / `raw8_idx3`.
8. Invalid-only tiles: пропускаются без model forward.
9. AOI: поддерживается, но не обязателен.
10. Финальный assembled model input baseline v1: `raw8_valid` или `raw8_idx3_valid`.
11. `valid` сохраняет двойную роль: служебная mask + дополнительный входной канал модели.
12. Runtime baseline по умолчанию hardware-adaptive.

## 22. Открытые решения перед кодингом

1. Нужен ли по умолчанию экспорт логитов наряду с probability maps.
2. Нужен ли baseline threshold-free confidence summary по выходам.
3. Какой default batch size по tiles считать стартовым под доступное железо.
4. Должен ли модуль уметь писать full-extent outputs всегда, или AOI-cropped outputs тоже могут быть default в некоторых сценариях.

## 23. Рекомендуемый порядок реализации

1. Checkpoint metadata reader.
2. Raster reader + valid mask resolver.
3. Feature adapter (`raw8` / `raw8_idx3`).
4. Predict-time normalization.
5. Sliding window inference engine.
6. Gaussian blending aggregator.
7. Output writers.
8. Predict manifest.
9. Single-raster smoke test.
10. Batch mode and AOI mode.
Да. Я добил research по этим пяти пунктам и могу дать окончательное решение для module_postprocess_vectorize.

Сразу честно: я попытался сразу внести это в документ модуля, но редактор не принял шаблон замены. Сами решения ниже уже финальные, и я буду держать их как принятые для следующей правки документа.

1. Какой boundary repair брать: morphology-first, graph-first или hybrid

Берём hybrid boundary repair.

Не чисто morphology-first и не чисто graph-first.

Почему. Свежие работы по parcel vectorization прямо показывают, что простая морфология полезна только для небольших локальных дефектов, но в сложных сценах её недостаточно для восстановления разорванных границ и сохранения топологии. В обзоре по APBD подчёркнуто, что после пиксельного boundary detection критично получать замкнутые и непрерывные parcels через post-processing, иначе остаются adhesion и разрывы. В работе CPVF отдельно сказано, что common methods используют watershed и morphology, но morphology ограничена по масштабу и гибкости, а более сильный вариант использует topological relation-based hanging line extension. В generalized framework 2024 oriented watershed дополнительно усиливает слабые границы и фактически работает как boundary connection.

Итоговое решение для v1:

этап 1: локальная morphology-first очистка и ограниченное closing/bridging мелких разрывов внутри extent-supported области;
этап 2: topology-aware endpoint connection только для оставшихся разрывов;
этап 3: connection разрешается только при проверке:
максимальной дистанции,
угловой согласованности,
отсутствия запрещённых пересечений,
согласия с extent_prob.

То есть baseline будет такой:
local morphology + constrained topological endpoint connection.

1. Какой должна быть точная формула marker generation

Markers нужно делать консервативными внутренними ядрами полей, а не просто threshold на одной карте.

Почему. Generalized framework 2024 прямо пишет, что distance-based seeds полезны, но сильно зависят от порогов и не всегда соответствуют полям один-к-одному; из-за этого могут появляться over-segmentation и ошибки разделения. Там же показано, что watershed по boundary probability лучше работает, когда слабые границы усиливаются, а затем идёт consolidation. Отдельно marker-controlled watershed и distance-transform-based seeding — стандартный класс решений для object separation.

Финальная baseline-формула для v1:

строим
extent_core = extent_prob >= T_core
вычитаем сильный boundary barrier:
boundary_barrier = boundary_prob >= T_boundary_strong
используем distance как внутренний сигнал удалённости от границы:
distance_seed = distance_pred >= T_dist_seed
получаем seeds:
markers = CC( extent_core AND NOT(boundary_barrier) AND distance_seed )
затем:
удаляем слишком маленькие компоненты;
при необходимости делаем split по локальным максимумам distance map;
запрещаем marker’ам касаться invalid.

То есть формально для baseline:

markers = CC((extent_prob >= T_core) AND (boundary_prob < T_boundary_strong) AND (distance_pred >= T_dist_seed))

с последующей фильтрацией по min_area.

Практический смысл:
extent даёт допустимую область поля, boundary не даёт маркеру переползти через сильную границу, distance выбирает внутренние устойчивые ядра. Это лучший baseline, чем seeds только по distance или только по extent.

1. Нужен ли polygon confidence уже в baseline

Да, нужен. Но не как сложная learned uncertainty model, а как простой rule-based QC score.

Почему. В practical parcel pipeline очень полезно сразу иметь способ ранжировать полигоны по надёжности для QC, ручной проверки и последующей фильтрации. При этом в baseline не стоит добавлять ещё одну обучаемую систему только ради confidence. Более рационально строить confidence из уже имеющихся raster evidences: extent, boundary, overlap, shape sanity. Это особенно уместно для MVP, где важна объяснимость. В современных vectorization workflows практическая ценность смещается к “ready-to-use” parcel instances и к контролю ошибок over-/under-segmentation, а не только к одной общей маске.

Финальное решение для v1:
вводим polygon_confidence как поле в атрибутах, но не делаем по нему жёсткое удаление по умолчанию.

Базовый score считаем из:

mean_extent_prob внутри полигона,
mean_boundary_prob_on_border,
mean_boundary_prob_inside_core с инверсией,
overlap_ratio с extent-support mask,
штрафа за контакт с invalid,
штрафа за сильную shape anomaly.

В простом виде:

confidence = w1*mean_extent_inside + w2*mean_boundary_on_border - w3*mean_boundary_inside_core + w4*overlap_ratio - penalties

Где веса фиксируются в конфиге.
Использование:

QC,
сортировка,
optional filtering,
downstream manual review.
4. Какие topology-fix операции включать по умолчанию

По умолчанию включаем только консервативный набор.

Почему. После polygonization нам нужен чистый и устойчивый слой, но нельзя превращать topology cleanup в агрессивную генерализацию, которая начинает “дорисовывать” поля. В generalized framework 2024 и review по APBD смысл postprocess именно в том, чтобы довести boundary-derived instances до закрытых и пригодных полигонов, а не переписать геометрию заново. GDAL polygonize сам по себе просто превращает connected regions в полигоны и умеет учитывать mask/NoData, но не решает топологические дефекты более высокого уровня.

Default topology-fix для v1:

удаление empty/invalid geometries,
make valid / исправление self-intersections,
удаление микродыр до порога площади,
удаление слишком маленьких островков и спайков,
нормализация multipart/singlepart policy,
консервативная topology-preserving simplification в метрах.

Что не включаем по умолчанию:

агрессивный smoothing,
массовый dissolve соседних полигонов,
сильный snapping,
тяжёлую генерализацию, которая заметно двигает границы.

То есть cleanup должен исправлять дефекты, а не менять картографический смысл результата.

1. Нужен ли Shapefile export по умолчанию параллельно с GPKG

Нет. По умолчанию — только GPKG. Shapefile оставляем как optional legacy export.

Почему. Официальная документация GDAL по Shapefile перечисляет очень неприятные для нашего проекта ограничения:

один geometry type на layer,
имена полей максимум 10 символов,
ограниченный набор типов атрибутов,
риск truncation строк и чисел,
многокомпонентный набор файлов .shp/.shx/.dbf/....
Для проекта, где мы хотим хранить нормальные имена полей, parcel_confidence, trace-поля, manifests и стабильный archive format, это плохой default. GeoPackage в официальной документации GDAL описан как OGC-стандарт на базе SQLite-контейнера с метаданными, spatial tables и более современными возможностями.

Итог:

основной рабочий и архивный формат: parcels.gpkg
optional export:
parcels.shp только по флагу,
GeoJSON только для лёгкого обмена, не как archive default.
Окончательная версия решений для module_postprocess_vectorize

Теперь финальный baseline модуля выглядит так:

Входы

extent_prob.tif
boundary_prob.tif
distance_pred.tif
valid.tif
optional AOI

Pipeline

input contract check
valid/AOI suppression
validation-calibrated thresholding
hybrid boundary repair
marker generation = extent-core ∩ low-boundary ∩ high-distance
marker-controlled watershed
region filtering / constrained merge
parcel_instance.tif
polygonization
conservative topology cleanup
polygon_confidence computation
export to GPKG

Обязательные outputs

parcel_instance.tif
parcels.gpkg
postprocess_manifest.json
summary.json
config_used.yaml

Рекомендуемые diagnostics

extent_mask.tif
boundary_mask.tif
boundary_repaired.tif
marker_map.tif
parcels_preview.*

Default format

GPKG
SHP только optional

Default geometry policy

topology cleanup обязателен,
но только консервативный.
Мои окончательные ответы коротко
boundary repair → hybrid
marker generation → extent-core + low-boundary + high-distance
polygon confidence → да, rule-based уже в baseline
topology fixes → консервативный default-набор
Shapefile default export → нет, только optional; default = GPKG

Это уже можно считать финальным freeze для module_postprocess_vectorize.
