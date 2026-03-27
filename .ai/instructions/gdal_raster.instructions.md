# gdal_raster.instructions.md

## Назначение

Этот файл задаёт **правила работы с растровыми и пространственными данными** в проекте **«ИИ для полей»**.

Он нужен для кода, который читает, валидирует, преобразует, режет, нормализует и записывает GeoTIFF/геоданные в:

- `module_prep_data`
- `module_target_predict`
- `module_postprocess_vectorize`
- частично `module_eval`

---

## 1. Главные принципы

### 1.1. Пространственный контракт важнее удобства

Нельзя жертвовать CRS, transform, alignment и valid/NoData semantics ради упрощения кода.

Если операция меняет spatial semantics, это должно быть:

- осознанно;
- явно зафиксировано;
- отражено в manifest;
- проверено валидатором.

### 1.2. Raster — это не просто ndarray

Любой raster должен рассматриваться как связка:

- пиксельные данные;
- CRS;
- affine transform;
- width / height;
- band count;
- dtype;
- nodata / masks;
- channel semantics.

Передавать по проекту «просто массив» без spatial context допускается только во внутренних локальных вычислениях, а не на границах модульного API.

### 1.3. NoData не равно фон

NoData / invalid area не должна молча превращаться в обычный background. Это критично для:

- построения `valid`;
- ignore policy;
- patch extraction;
- train loss;
- predict suppression;
- eval fairness.

---

## 2. Чтение raster-данных

### 2.1. Что нужно извлекать при чтении

При чтении raster желательно сразу зафиксировать:

- `crs`
- `transform`
- `width`
- `height`
- `count`
- `dtype`
- `nodata`
- dataset mask / band masks, если доступны

### 2.2. Проверки на входе

До начала обработки нужно проверять:

- файл читается без ошибок;
- band count соответствует ожидаемому контракту;
- CRS определён или есть допустимое правило его восстановления;
- raster не пустой;
- nodata/valid interpretation разрешима.

### 2.3. Когда нужно падать

Нужно падать явно, если:

- входной raster не читается;
- band count не соответствует ожидаемому режиму;
- spatial metadata отсутствует и не может быть надёжно восстановлена;
- нет способа корректно определить `valid`.

---

## 3. CRS и reprojection

### 3.1. Reference grid policy

В большинстве операций базовым пространственным reference должен быть raster grid.

Векторы и AOI обычно приводятся к CRS растра, а не наоборот, если модуль не делает специально оговорённую пространственную нормализацию.

### 3.2. Reprojection должна быть явной

Если выполняется reprojection:

- нужно явно знать source CRS и target CRS;
- нужно фиксировать это в metadata/manifest;
- нельзя «предполагать правильный CRS» по косвенным признакам без явной политики.

### 3.3. Alignment matters

Для согласованных raster-слоёв нужно проверять:

- одинаковый CRS;
- совместимый transform;
- одинаковый размер grid, если слой должен совпадать попиксельно;
- отсутствие half-pixel drift.

---

## 4. `valid` и NoData

### 4.1. Приоритет интерпретации

`valid` следует вычислять по явной политике, например:

1. internal/sidecar mask;
2. dataset nodata metadata;
3. проектное правило из конфига;
4. явная ошибка.

### 4.2. Важнейшее правило

`valid` должен вычисляться **до** возможной замены NoData на числовой fill value.

### 4.3. Отдельный слой

`valid` должен существовать как самостоятельный слой с семантикой 0/1.

Даже если downstream-код дополнительно использует `valid` как канал модели, отдельный raster-слой всё равно не должен теряться.

### 4.4. Запрещённые упрощения

Запрещено:

- молча считать все нули invalid или valid без политики;
- вычислять `valid` по одной логике в `prep_data`, а по другой в `predict`;
- silently drop invalid zones из manifests и summary.

---

## 5. Работа с band order и channel semantics

### 5.1. Band order должен быть фиксированным

Если модуль ожидает 8-band stack, нужно явно знать порядок каналов и отражать его в metadata/manifest.

### 5.2. Derived features

При построении `raw8_idx3` derived indices должны:

- строиться из явно определённых базовых каналов;
- иметь стабильный порядок;
- иметь явные имена в `channel_semantics`.

### 5.3. Нельзя делать скрытый reordering

Если band order не совпадает с ожидаемым контрактом, нельзя молча переставлять каналы без явно разрешённого adapter-слоя.

---

## 6. Patch / tile extraction

### 6.1. Patch extraction должен быть spatially honest

Каждый patch/tile должен иметь восстановимую связь с исходным raster:

- offsets / window;
- CRS;
- transform;
- связь с sample meta.

### 6.2. Invalid-aware extraction

При извлечении patch/tile нужно явно учитывать:

- valid ratio;
- invalid borders;
- полностью invalid tiles;
- влияние AOI, если она используется.

### 6.3. Не терять provenance окна

Если окно получено из конкретного `Window`, эта информация должна быть восстановима хотя бы из `meta` или manifest.

---

## 7. Raster writing

### 7.1. Запись должна сохранять spatial meaning

При записи GeoTIFF нужно явно контролировать:

- CRS;
- transform;
- width / height;
- dtype;
- count;
- nodata, если релевантно.

### 7.2. Output semantics должны быть понятны downstream

Downstream-модуль должен без догадок понимать:

- что означает каждый band / файл;
- какие допустимые значения есть;
- является ли слой probability, class encoding, distance, mask или auxiliary artifact.

### 7.3. Геопривязка predict outputs обязательна

Для `extent_prob`, `boundary_prob`, `distance_pred`, `valid` геопривязка не является опциональной.

---

## 8. Работа с векторами

### 8.1. Vector input нужно валидировать

При чтении vector ground truth / AOI желательно проверять:

- читаемость;
- CRS;
- geometry validity;
- пустые geometry;
- поддерживаемые geometry types.

### 8.2. GT boundaries должны быть linework-faithful

Если строится `boundary` target, он должен происходить от реальной геометрии границ, а не от суррогатного упрощения, которое теряет внутренние границы.

### 8.3. Rasterization policy должна быть явной

Нужно явно понимать:

- что именно растеризуется;
- в каком CRS;
- на какой grid;
- по каким правилам получается final target.

---

## 9. Нормализация raster-данных

### 9.1. Нормализация — часть контракта

Нельзя считать, что нормализация — это «локальная деталь train-кода».

Если она влияет на assembled model input, она должна быть:

- одинаковой по смыслу в train и predict;
- описанной в metadata;
- воспроизводимой через checkpoint/train artifacts.

### 9.2. Valid-aware statistics

Статистики нормализации должны считаться по valid пикселям, если проектный контракт требует именно этого.

### 9.3. Не смешивать storage dtype и model dtype

Оригинальный raster может иметь один dtype, а model-ready tensor — другой. Это нормально, если переход:

- явный;
- воспроизводимый;
- зафиксирован в policy.

---

## 10. Manifest-aware spatial coding

Каждая значимая raster/vector операция должна оставлять после себя информацию, достаточную для forensic-анализа:

- что прочитали;
- как интерпретировали `valid`;
- были ли reprojection/resampling;
- какой feature stack собрали;
- какие outputs записали;
- какие spatial warnings были замечены.

---

## 11. Антипаттерны

Избегать:

- передачи numpy array без spatial metadata через модульные границы;
- тихой потери CRS/transform;
- смешивания valid и background;
- «подозрительно похожих» auto-fix по геометрии без явной фиксации;
- silent clipping/resampling;
- hidden fallback при mismatch raster size/grid.

---

## 12. Чеклист перед merge

Перед merge желательно проверить:

- raster/vector spatial contract не потерян;
- `valid` считается до NoData replacement;
- band order и `channel_semantics` явно зафиксированы;
- outputs геопривязаны и downstream-readable;
- manifests отражают ключевые spatial/runtime решения;
- negative tests покрывают mismatch cases.
