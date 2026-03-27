# GLOSSARY.md

## Назначение

Этот глоссарий фиксирует единый словарь терминов для проекта **«ИИ для полей»**.  
Его цель — убрать двусмысленность между модулями, конфигами, manifest-артефактами, README, кодом и ревью.

Термины ниже считаются **предпочтительными формулировками v1**.  
Если в старых документах встречаются более ранние названия, приоритет у этого файла.

---

## A

### AOI

**Area of Interest** — область интереса, в пределах которой система должна работать или на которой должна фокусироваться.

В проекте:

- AOI не является строго обязательным входом;
- AOI рекомендуется использовать, если он доступен;
- AOI может применяться в `module_prep_data`, `module_target_predict`, `module_postprocess_vectorize`, `module_eval`;
- baseline buffer вокруг AOI: **30 м**.

Не путать:

- `AOI` — географическое ограничение области обработки;
- `valid` — маска валидности данных.

---

### Assembled model input

Финальный входной тензор, который реально подается в модель после сборки всех необходимых каналов.

Примеры:

- `raw8_valid = raw8 + valid`
- `raw8_idx3_valid = raw8_idx3 + valid`

Важно:

- assembled model input — это не то же самое, что dataset-side feature stack;
- assembled input определяется не только `feature_mode`, но и ролью `valid`.

---

## B

### Baseline v1

Зафиксированный стартовый рабочий вариант архитектуры, контрактов и правил проекта, с которым синхронизированы все модули.

Baseline v1 нужен для:

- единообразной реализации;
- честного сравнения экспериментов;
- предотвращения скрытого drift-а между train/predict/eval.

---

### Boundary

Карта границ полей.

В multitask-постановке:

- `boundary` — отдельный предсказываемый target;
- используется вместе с `extent` и `distance`;
- играет ключевую роль в разделении соседних полей.

В `module_prep_data`:

- boundary должен строиться **linework-faithful** от реальных границ полигонов.

---

### Boundary encoding

Схема кодирования boundary-target в датасете.

Baseline v1:

- `background`
- `skeleton`
- `buffer`

Иными словами:

- фон
- центральная линия границы
- буферная зона вокруг границы

---

### Boundary repair

Этап постпроцессинга, на котором локально восстанавливаются или улучшаются boundary-структуры перед marker-controlled watershed.

Baseline v1:

- используется **hybrid boundary repair**;
- запрещены агрессивные глобальные “сшивки” без локальных ограничений.

---

### Boundary-support mask

Вспомогательная маска в postprocess, отражающая поддержку гипотезы о наличии границы.

Используется не как самостоятельный конечный результат, а как часть логики thresholding / repair / watershed.

---

## C

### Calibration

Настройка или выбор порогов и правил принятия решений на основе validation-stage наблюдений.

В проекте:

- thresholding должен быть **validation-calibrated**, а не “на глаз”;
- threshold provenance должен фиксироваться в manifest/reporting artifacts.

---

### Channel semantics

Смысл каждого канала входного тензора.

Примеры:

- band_1 ... band_8
- NDVI
- SAVI
- NDWI
- valid

Важно:

- channel semantics должны быть однозначно известны train и predict;
- несоответствие semantics и `in_channels` — это ошибка контракта.

---

### Checkpoint metadata

Набор metadata, сохраняемый вместе с обученной моделью.

Должен содержать как минимум:

- `feature_mode`
- `in_channels`
- `channel_semantics`
- использование `valid` как входного канала
- настройки, нужные для безопасного predict-time восстановления контракта

---

### Composite metric

Составная метрика выбора лучшего чекпоинта.

Baseline v1:

- `0.6 * boundary_F1 + 0.4 * extent_F1`

---

### Conservative topology cleanup

Консервативная топологическая очистка геометрий на этапе postprocess/vectorization.

Смысл:

- исправлять технические дефекты геометрии;
- не менять агрессивно картографический смысл объектов.

Включает:

- `make valid`
- удаление мусора
- ограниченную cleanup-логику
- осторожную simplification

Не включает по умолчанию:

- агрессивный smoothing
- массовый dissolve
- сильный snapping

---

## D

### Dataset-side feature mode

Режим feature stack, который формируется на стороне датасета до добавления `valid` как части assembled model input.

В проекте:

- `raw8`
- `raw8_idx3`

Важно:

- это не финальный input модели;
- это не то же самое, что `raw8_valid` / `raw8_idx3_valid`.

---

### Deep supervision

Архитектурный прием, при котором промежуточные уровни модели тоже получают supervised signal.

Используется в `module_net_train` baseline v1.

---

### Distance

Карта расстояния до ближайшей границы.

В проекте:

- это отдельный multitask target;
- используется на train;
- используется на predict;
- используется на postprocess, особенно для marker generation.

Baseline v1:

- `distance` = **unsigned distance-to-nearest-boundary**

---

### Distance_pred

Результат предсказания головы `distance` на этапе predict.

Файл:

- `distance_pred.tif`

---

## E

### Edge-aware

Свойство архитектуры, loss policy или postprocess-логики, при котором модель и pipeline специально учитывают границы объектов как ключевую часть задачи.

В этом проекте задача не просто сегментация полей, а именно **boundary-aware delineation**.

---

### Encoder-decoder

Базовый класс архитектур, где encoder сжимает и извлекает признаки, а decoder восстанавливает пространственную структуру выхода.

В проекте baseline v1:

- гибридный multitask encoder-decoder
- CNN-first residual encoder
- lightweight global-context bottleneck
- multi-scale fusion
- refine module

---

### Eval provenance

Информация о происхождении входов и правил eval-запуска.

Должна включать:

- source run identifiers
- source manifests/configs
- thresholds
- scene list
- valid/AOI policy

---

### Extent

Карта протяженности/маски поля.

Это не просто бинарная маска “интерес / не интерес”, а отдельный основной target, который задаёт область поля.

Файлы:

- train target: `extent`
- predict output: `extent_prob.tif`

---

### Extent-core

Консервативная внутренняя часть extent-области, используемая в postprocess для marker generation.

---

## F

### Feature contract

Общий контракт признаков, который должен быть одинаково понятен:

- `module_prep_data`
- `module_net_train`
- `module_target_predict`

Включает:

- состав каналов
- их порядок
- channel semantics
- правила normalisation
- связь feature stack и `valid`

---

### Feature mode

Краткое имя режима признаков.

В проекте есть два смысла этого термина, и их важно не путать:

1. **Dataset-side feature mode**  
   - `raw8`
   - `raw8_idx3`

2. **Assembled model input contract**  
   - `raw8_valid`
   - `raw8_idx3_valid`

Если слово `feature_mode` используется без уточнения, лучше явно писать, о каком уровне идёт речь.

---

### Float32

Рабочий числовой формат входа модели после preprocess.

Baseline v1:

- `valid` вычисляется до преобразований;
- далее данные приводятся к `float32`.

---

## G

### Gaussian blending

Метод склейки перекрывающихся tile-предсказаний на этапе tiled inference.

Baseline v1:

- tile size = `512`
- overlap = `25%`
- blending = `Gaussian`

---

### GeoTIFF

Основной raster-формат входов и промежуточных/итоговых растровых outputs в проекте.

---

### GOC / GUC / GTC

Object/structure-level метрики оценки.

Используются в `module_eval` как часть обязательного object/structure evaluation.

---

## H

### Hardware-adaptive runtime policy

Политика автоматической адаптации исполнения под доступную среду.

Baseline v1:

- приоритет устройств: `CUDA -> MPS -> CPU`
- AMP включается автоматически, если это безопасно
- при OOM деградация идет по runtime-параметрам, а не через ломание model contract

Нельзя ради памяти:

- убирать `valid` из assembled input
- менять число каналов скрыто
- ломать ignore/valid policy
- менять train/predict preprocessing logic

---

### Hybrid boundary repair

Baseline v1 подход для восстановления границ в postprocess.

Сочетает:

- локальную morphology-first repair-логику
- topology-aware endpoint connection

Но только с ограничениями:

- по дистанции
- по углу
- по согласованности с extent/boundary evidence

---

## I

### Ignore policy

Правила исключения пикселей или зон из loss/metrics/postprocess consideration.

Чаще всего связано с:

- `valid == 0`
- NoData
- special ignore labels

---

### In_channels

Число каналов, которое модель ожидает на входе.

Важно:

- должно совпадать с assembled model input, а не только с dataset-side feature stack.

Baseline v1:

- `raw8_valid` -> `9`
- `raw8_idx3_valid` -> `12`

---

### Instance raster

Растр, где каждому объекту соответствует свой integer id.

В postprocess baseline:

- это `parcel_instance.tif`

---

## M

### Manifest

Машиночитаемый артефакт, который фиксирует:

- resolved config
- contract
- runtime parameters
- outputs
- provenance
- fallback-режимы

Проект придерживается manifest-first подхода.

---

### Marker

Консервативное внутреннее ядро предполагаемого объекта, используемое для watershed-based parcel reconstruction.

---

### Marker-controlled watershed

Основной baseline-метод parcel reconstruction в `module_postprocess_vectorize`.

Использует:

- markers
- boundary/distance/extent-derived relief
- дополнительные ограничения и фильтрацию

---

### Meta / metadata

Сопровождающая информация о sample, run, checkpoint или output.

Не равна manifest:

- metadata может быть локальной и краткой;
- manifest — это формализованный run-level источник истины.

---

### Multi-run comparison

Режим `module_eval`, в котором сравниваются разные model/run variants по единому протоколу.

---

### Multitask

Постановка, где модель одновременно предсказывает несколько связанных targets.

В проекте baseline multitask heads:

- `extent`
- `boundary`
- `distance`

---

## N

### NDVI

Один из спектральных индексов.

Используется в `raw8_idx3`.

---

### NDWI

Один из спектральных индексов.

Используется в `raw8_idx3`.

---

### NoData

Специальное значение или состояние пикселей, которые не содержат валидных данных.

Важно:

- NoData не должен молча смешиваться с фоном;
- `valid` должен вычисляться до любых преобразований NoData;
- invalid / NoData зоны исключаются из loss и учитываются в downstream логике.

---

### Normalization stats

Статистики, используемые для приведения входных данных к model-ready диапазону.

Baseline v1:

- считаются только по valid train pixels
- per-band robust statistics
- clipping `p0.5 / p99.5`
- scaling в `[0, 1]`
- predict использует те же train-derived stats

---

## O

### OOM

Out Of Memory — ситуация нехватки памяти.

При OOM baseline v1 требует:

- сначала уменьшать batch/tile batch
- затем снижать вторичные runtime-параметры
- затем применять более медленные корректные режимы
- не ломать model contract

---

## P

### Parcel

Итоговая единица поля/участка, которую система пытается выделить и векторизовать.

---

### Parcel delineation

Задача восстановления отдельных полей и их границ по спутниковому изображению.

---

### Parcel instance

Отдельный объект поля после postprocess.

На raster-уровне представлен в `parcel_instance.tif`.

---

### parcel_instance.tif

Обязательный baseline output `module_postprocess_vectorize`.

Содержит:

- integer id каждого parcel instance
- фон
- правила кодирования, описанные в manifest

---

### Patch

Небольшое окно/фрагмент, используемый в `module_prep_data` и train-dataset.

Baseline patch sizes:

- `256`
- `384`
- `512`

Baseline default:

- `512`

Не путать:

- `patch` — train/prep единица
- `tile` — inference единица

---

### Pixel metrics

Класс метрик, считаемых по пикселям.

Примеры:

- IoU
- F1
- Precision
- Recall

---

### Polygon confidence

Rule-based QC score для итогового полигона.

Используется в `module_postprocess_vectorize`.

Назначение:

- качество и приоритет ручной проверки
- optional filtering
- downstream QC

Не должен по умолчанию быть жестким удаляющим фильтром.

---

### Polygonization

Преобразование instance raster или mask в векторные полигоны.

В проекте:

- выполняется после `parcel_instance.tif`
- не должна напрямую заменять полноценный postprocess

---

### Postprocess

Этап после predict, который преобразует model outputs в устойчивые parcel instances и финальные полигоны.

---

### Predict manifest

Manifest predict-run-а, в котором фиксируются:

- source checkpoint
- resolved device/runtime
- assembled input contract
- tile policy
- output paths
- fallbacks

---

### Provenance

Информация о происхождении артефакта или вычисления.

Ключевая идея:

- любой сравниваемый результат должен иметь прозрачное происхождение.

---

## Q

### QC

Quality Control — контроль качества.

В проекте часто связан с:

- diagnostics
- `polygon_confidence`
- manual review support
- forensic analysis

---

## R

### Raw8

Dataset-side feature mode:

- 8 спектральных каналов
- без добавления `valid` в сам feature stack

---

### Raw8_idx3

Dataset-side feature mode:

- 8 спектральных каналов
- плюс `NDVI`, `SAVI`, `NDWI`
- итого 11 feature-каналов до добавления `valid`

---

### Raw8_valid

Assembled model input:

- `raw8 + valid`
- итого `9` каналов

---

### Raw8_idx3_valid

Assembled model input:

- `raw8_idx3 + valid`
- итого `12` каналов

---

### Refine module

Архитектурный блок baseline модели, используемый для улучшения итогового предсказания после основной decode/fusion-части.

---

### Region filtering

Фильтрация регионов после watershed или других intermediate stages.

---

### Resolved contract

Полностью восстановленный и уже не двусмысленный контракт конкретного запуска.

Пример:

- какой `feature_mode` использовался
- какой assembled input был построен
- какая valid/AOI policy была применена
- какие thresholds были реально использованы

---

### Robust percentile normalization

Базовая стратегия нормализации входов.

Baseline v1:

- per-band
- по valid train pixels
- clipping `0.5 / 99.5`

---

## S

### Sample

Одна обучающая единица датасета.

Обычно включает:

- `img`
- `valid`
- `extent`
- `boundary`
- `distance`
- `meta`

---

### SAVI

Один из спектральных индексов.

Используется в `raw8_idx3`.

---

### Scene

Одна целостная область данных, используемая для predict/eval.

Может соответствовать:

- одному raster input
- AOI-ограниченной части большого raster-а
- отдельному evaluation case

---

### Skeleton

Центральная линия boundary-структуры в boundary encoding.

---

### Split

Разделение датасета:

- train
- val
- test

---

### Strategic sampling

Политика генерации patch-ей, при которой выбор окон идёт не только случайно, а с учетом полезных сценариев:

- boundary-rich
- center-like
- hard zones
- near-invalid
- негативные примеры

---

## T

### Threshold provenance

Явная фиксация того, откуда взялись threshold values.

Должно быть ясно:

- validation-calibrated они или нет
- в каком config/manifest они сохранены
- одинаковы ли они между сравниваемыми run-ами

---

### Tile

Окно, используемое на predict/inference этапе.

Baseline predict:

- tile size = `512`
- overlap = `25%`

Не путать:

- `tile` — predict
- `patch` — train/prep

---

### Tiled inference

Режим предсказания по большому raster-у через окна с overlap и последующей склейкой.

---

### Topology cleanup

Этап чистки векторных геометрий после polygonization.

См. также:

- `Conservative topology cleanup`

---

### Train-derived statistics

Статистики нормализации, вычисленные на train-data и используемые потом на predict.

Это обязательная часть честного контракта между train и predict.

---

## V

### Valid

Ключевой слой проекта, отражающий валидность данных.

У `valid` в этом проекте **двойная роль**:

1. **служебная valid-mask**  
   Используется для:
   - ignore policy
   - loss masking
   - metric masking
   - suppression invalid-зон
   - diagnostics

2. **дополнительный входной канал модели**  
   Используется как часть assembled model input:
   - `raw8_valid`
   - `raw8_idx3_valid`

Важно:

- это одна из центральных проектных договорённостей;
- её нельзя “случайно потерять” при реализации.

---

### valid-mask

Функциональная роль `valid` как маски корректных данных.

Лучше использовать этот термин, когда нужно подчеркнуть именно маскирующую семантику, а не его роль как входного канала.

---

### Validation-calibrated thresholding

Thresholding, чьи параметры выбраны по validation protocol, а не вручную “на глаз”.

Используется в baseline v1 постпроцессинге.

---

### Vector output

Итоговый полигональный результат pipeline.

Baseline default format:

- `GPKG`

Optional:

- `SHP`

---

## W

### Watershed

Алгоритм разделения областей по рельефу и markers.

В baseline v1 используется **marker-controlled watershed**, а не произвольный watershed без консервативных markers.

---

## Рекомендуемые правила именования в коде и документации

### Использовать предпочтительно

- `dataset-side feature mode`
- `assembled model input`
- `valid-mask`
- `predict manifest`
- `postprocess manifest`
- `resolved contract`
- `train-derived statistics`
- `validation-calibrated thresholding`

### Избегать без уточнения

- `features` — слишком общее слово
- `input` — непонятно, raw input это или assembled input
- `mask` — без указания, это `extent`, `boundary`, `valid` или что-то ещё
- `normalization` — без указания источника статистик
- `postprocess cleanup` — без указания, raster это cleanup или vector topology cleanup

---

## Короткие правила против путаницы

1. `raw8` и `raw8_idx3` — это **не финальный вход модели**, а dataset-side feature modes.
2. Финальный input модели — это `raw8_valid` или `raw8_idx3_valid`.
3. `valid` — это и mask, и входной канал.
4. `patch` и `tile` — не одно и то же.
5. `extent`, `boundary`, `distance` — три разных target/output-а, не взаимозаменяемые.
6. `postprocess` начинается после predict и не должен жить внутри predict-логики.
7. `eval` должен сравнивать только те run-ы, у которых понятен provenance и contract.

---

## Статус документа

Этот глоссарий относится к **architectural freeze v1** и должен обновляться только при явном изменении проектного контракта.
