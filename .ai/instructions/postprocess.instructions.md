# postprocess.instructions.md

## Назначение

Этот файл задаёт правила для разработки `module_postprocess_vectorize` и связанного кода, который превращает raster predictions в финальные parcel-level vector outputs.

Он обязателен для кода, который:

- читает `extent_prob`, `boundary_prob`, `distance_pred`, `valid`;
- применяет valid/AOI suppression;
- выполняет thresholding и marker generation;
- делает watershed или сходный instance-separation step;
- строит `parcel_instance.tif`;
- выполняет polygonization;
- делает conservative topology cleanup;
- экспортирует `GPKG` и связанные diagnostics;
- пишет `postprocess_manifest.json` и `summary.json`.

Этот файл дополняет:

- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `DECISIONS.md`
- ТЗ для `module_postprocess_vectorize` / общий architectural freeze v1

---

## Главная роль модуля

`module_postprocess_vectorize` не обучает модель и не меняет смысл upstream predictions.

Его задача:

1. принять probability/regression raster outputs модели;
2. применить фиксированные и воспроизводимые правила postprocess;
3. получить instance-like parcel raster;
4. перевести его в векторный результат;
5. сохранить provenance, thresholds и diagnostics.

Иными словами:

- predict отвечает за вероятностные карты;
- postprocess отвечает за отделение объектов и выпуск финальных полигонов.

---

## Контракт входов

Базовые обязательные входы:

- `extent_prob`
- `boundary_prob`
- `distance_pred`
- `valid`

Опционально:

- AOI
- manifests/configs upstream run-ов
- threshold overrides, если они явно допускаются конфигом

### Входной код должен проверять

- spatial alignment всех raster inputs;
- одинаковые width/height/transform/CRS там, где это требуется;
- совместимость `valid` с prediction outputs;
- отсутствие silent resampling без явной фиксации;
- что значения реально выглядят как probability/regression maps, а не произвольные слои.

---

## Главные принципы postprocess

### 1. Valid-first

Никогда не делать object extraction без учёта `valid`.

`valid` — обязательная маска допустимой области.
Нельзя трактовать invalid-пиксели как фон, пригодный для обычной сегментации.

### 2. AOI-aware, но без скрытой магии

Если AOI используется, его роль должна быть явно определена:

- ограничение области постпроцессинга;
- suppression вне AOI;
- reporting extent;
- или их комбинация, если это отдельно зафиксировано.

### 3. Threshold provenance обязателен

Любой threshold должен быть:

- явно сохранён;
- воспроизводим;
- привязан к config/manifest;
- различим как fixed / calibrated / overridden.

### 4. Boundary-aware separation

Модуль должен использовать boundary-сигнал как важнейший источник для разделения смежных parcel-ов.
Нельзя сводить задачу к простому бинарному threshold `extent_prob` без boundary-aware логики.

### 5. Conservative topology cleanup

Cleanup должен исправлять очевидные артефакты, но не перепридумывать геометрию агрессивно.

---

## Базовый pipeline v1

Рекомендуемый baseline flow:

1. valid/AOI suppression
2. thresholding по `extent_prob` и вспомогательным картам
3. hybrid boundary repair
4. marker generation
5. marker-controlled watershed
6. region filtering
7. constrained merge при необходимости
8. запись `parcel_instance.tif`
9. polygonization
10. conservative topology cleanup
11. экспорт в `GPKG`
12. запись manifests / summary / diagnostics

Если конкретная реализация меняет этот flow, это должно быть явно отражено в config и manifest.

---

## Marker generation

Markers — критическая часть separation logic.

В baseline v1 marker generation должен опираться на сочетание:

- high-confidence `extent` core;
- low-boundary zones;
- `distance_pred` как сигнал внутренности parcel-а.

Рекомендуемый принцип:

`marker = extent_core ∩ low_boundary ∩ high_distance`

### Что запрещено

- брать markers из сырого extent-threshold без boundary-aware проверки;
- использовать недокументированные heuristic-константы;
- менять marker policy без отражения в manifest.

---

## Watershed / instance separation

### Общий принцип

Watershed или аналогичный алгоритм используется как controlled instance separation step, а не как магический чёрный ящик.

### Требования

- входные markers должны быть диагностируемыми;
- маска допустимой области должна быть явной;
- причина separation-политики должна быть читаема из config/manifest;
- при нестабильности нужно сохранять промежуточные debug artifacts.

### Что полезно сохранять как diagnostics

- marker raster
- thresholded extent mask
- repaired boundary mask
- watershed seeds
- pre-filter instance raster

Не всё это обязано идти в production export, но для forensic/debug mode должно быть доступно.

---

## Region filtering и merge policy

После instance extraction обычно нужны фильтры:

- минимальная площадь;
- минимальная уверенность;
- подавление явного мусора;
- ограниченное слияние подозрительно раздробленных регионов.

### Важное правило

Merge policy должна быть **constrained**.

Нельзя делать агрессивное слияние только ради красивой картинки, если оно скрыто уничтожает реальные внутренние границы.

---

## Polygonization

### Обязательные правила

- polygonization должна выполняться по `parcel_instance.tif` или эквивалентному instance raster;
- spatial metadata должна сохраняться корректно;
- формат baseline export — `GPKG`;
- `Shapefile` — только optional export.

### После polygonization

Допустимы только conservative cleanup-операции:

- удаление совсем микроскопических шумовых объектов;
- исправление очевидно невалидных геометрий;
- лёгкая топологическая чистка;
- заполнение технических атрибутов.

Недопустимы агрессивные shape-altering операции без отдельного архитектурного решения.

---

## polygon_confidence

Итоговый векторный экспорт должен поддерживать `polygon_confidence` как rule-based QC score.

Этот score должен быть:

- объяснимым;
- детерминированным;
- вычислимым из доступных raster/vector diagnostics;
- описанным в config/manifest.

Нельзя делать `polygon_confidence` как “непонятную внутреннюю эвристику”.

---

## Требования к коду

### Разделять pipeline на стадии

Не писать всё в одной функции `run_postprocess()`.

Минимально разделять на слои:

- `io.py`
- `validators.py`
- `thresholds.py`
- `markers.py`
- `separation.py`
- `filters.py`
- `polygonize.py`
- `cleanup.py`
- `manifest.py`
- `cli.py`

### Чистые функции важнее monolithic pipeline

Предпочтительны функции вида:

- `apply_valid_and_aoi_mask(...)`
- `build_extent_core_mask(...)`
- `repair_boundary_map(...)`
- `generate_markers(...)`
- `run_watershed(...)`
- `filter_instances(...)`
- `polygonize_instances(...)`
- `compute_polygon_confidence(...)`

### Debug-mode должен быть предусмотрен заранее

Нужна возможность сохранить intermediate artifacts без переписывания кода.

---

## Spatial safety

Любая гео-операция должна быть spatially explicit.

Обязательно проверять:

- CRS
- transform
- shape
- alignment
- pixel meaning

Запрещено:

- silently reproject / resample входы;
- silently crop без фиксации;
- терять georeferencing при промежуточной записи.

---

## Threshold handling

Thresholds должны жить в config и manifest, а не быть зашиты внутри функций.

### Для каждого threshold важно фиксировать

- имя
- значение
- роль
- источник
- fixed / tuned / calibrated / overridden status

### Примеры threshold-групп

- extent binarization threshold
- boundary suppression threshold
- distance core threshold
- minimum instance area
- polygon confidence bins

---

## Diagnostics

Минимальный полезный diagnostics-набор для debug-capable режима:

- valid-suppressed extent map
- thresholded extent mask
- repaired boundary map
- marker raster
- parcel_instance pre-filter
- parcel_instance final
- polygon count summary
- size distribution summary

Diagnostics должны помогать понять, где именно ломается separation:

- extent слишком рыхлый;
- boundary слишком слабый;
- markers редкие/шумные;
- watershed пере-дробит;
- cleanup сливает лишнее.

---

## Manifest requirements

`postprocess_manifest.json` обязан сохранять минимум:

- source predict run id/path;
- input artifact descriptions;
- resolved valid/AOI policy;
- threshold policy;
- marker generation policy;
- watershed/separation policy;
- filtering and cleanup policy;
- output artifacts;
- warnings/errors.

Если thresholds или cleanup реально менялись, это должно быть видно в manifest.

---

## Что запрещено

Запрещено:

- делать финальную векторизацию прямо внутри predict-модуля;
- игнорировать `valid`;
- silently подбирать thresholds без записи provenance;
- считать boundary map необязательной для separation logic;
- делать агрессивный polygon smoothing по умолчанию;
- прятать merge/filter rules внутри кода без config/manifest;
- сравнивать два postprocess run-а без учёта threshold policy и AOI/valid policy.

---

## Как использовать эти инструкции с ИИ-ассистентами

Когда ассистент пишет код для postprocess, просить его:

1. сначала описать stage decomposition;
2. затем описать входные/выходные contracts функций;
3. затем написать validators;
4. затем реализовать marker/separation logic;
5. затем добавить manifest writing;
6. затем покрыть это unit + integration tests;
7. явно перечислить thresholds и debug artifacts.

Не просить: “сделай красивую постобработку”.

---

## Чеклист перед merge

Перед merge проверять:

- входные prediction rasters spatially aligned;
- `valid` реально используется;
- AOI-policy явно определена;
- thresholds вынесены в config;
- markers и separation диагностируемы;
- `parcel_instance.tif` сохраняется;
- `GPKG` экспортируется корректно;
- `polygon_confidence` объясним и воспроизводим;
- manifest и summary сохраняются;
- integration tests покрывают минимум один end-to-end postprocess path.
