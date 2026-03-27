# ТЗ `module_net_train` v1

## 1. Назначение модуля

`module_net_train` отвечает за обучение, валидацию, контроль экспериментов и экспорт обученных моделей для задачи выделения и разделения сельскохозяйственных полей по высокоразрешённым 8-бэндовым спутниковым снимкам.

Модуль должен принимать строго подготовленный датасет из `module_prep_data` и обучать нейросеть, оптимизированную не только под общую сегментацию поля, но и под точное восстановление границ, разделение соседних участков и устойчивость в сложных boundary/near-invalid сценариях.

## 2. Границы ответственности

### 2.1. Модуль обязан

* читать train-ready датасет по фиксированному контракту;
* строить модель по конфигу;
* обучать и валидировать модель;
* считать train/val/test метрики;
* сохранять checkpoints, manifests и историю обучения;
* экспортировать артефакты, необходимые для `module_target_predict`;
* экспортировать metadata, достаточную для однозначного восстановления downstream model input contract;
* обеспечивать совместимость checkpoint metadata, dataset contract и predict pipeline без скрытых fallback-догадок.
* поддерживать сравнение режимов `raw8` и `raw8_idx3` без изменения кода.

### 2.2. Модуль не обязан

* готовить датасет из сырых GeoTIFF/векторных данных;
* выполнять full-AOI production inference;
* делать финальную векторизацию полей;
* исправлять дефекты геометрии постпроцессингом.

## 3. Входной контракт

### 3.1. Источник данных

Единственным поддерживаемым источником обучающих данных является экспорт `module_prep_data`.

### 3.2. Обязательные входные слои на sample

* `img`
* `extent`
* `boundary`
* `distance`
* `valid`
* `meta`

### 3.3. Поддерживаемые feature modes

* `raw8`
* `raw8_idx3`
Здесь `raw8` и `raw8_idx3` понимаются как dataset-side feature modes, приходящие из `module_prep_data`.

Финальный model input внутри `module_net_train` собирается с обязательным учетом слоя `valid`, который используется не только для ignore/masking, но и как дополнительный входной канал модели.

### 3.4. Число входных каналов

* `raw8` → 8 каналов
* `raw8_idx3` → 11 каналов

Модуль должен строго сверять число каналов модели и датасета.
Указанные выше значения относятся к числу feature-каналов до добавления `valid`.

Финальный model input contract для baseline v1:

* `raw8_valid = raw8 + valid` → 9 каналов
* `raw8_idx3_valid = raw8_idx3 + valid` → 12 каналов

Соответственно, модуль должен строго сверять:

* число feature-каналов в датасете;
* наличие и семантику слоя `valid`;
* итоговое число каналов фактического model input tensor.

## 4. Основная ML-постановка

### 4.1. Базовая постановка v1

В первой сильной версии принимается multitask-постановка с тремя задачами:

1. `extent` — основная карта принадлежности пикселя полю;
2. `boundary` — boundary-aware карта границ;
3. `distance` — карта расстояния до ближайшей границы.

### 4.2. Мотивация

Модель не должна ограничиваться простой бинарной сегментацией `field / background`. Она должна явно учить геометрию границ и separation adjacent fields.

## 5. Целевая архитектура модели

### 5.1. Архитектурный выбор v1

В качестве базовой архитектуры принимается **hybrid edge-aware multitask encoder-decoder network**.

### 5.2. Почему именно этот класс архитектур

Архитектура должна одновременно:

* извлекать локальные текстурные и спектральные признаки;
* использовать многомасштабный контекст;
* усиливать edge-awareness;
* поддерживать multitask supervision;
* быть реалистичной по вычислительной стоимости для MVP.

### 5.3. Общая структура

1. **Stem / input adapter** под 8 или 11 входных каналов.
В baseline v1 input adapter должен быть согласован не только с dataset feature stack (`8` или `11` каналов), но и с финальным assembled model input (`9` или `12` каналов) после добавления `valid`.
2. **Encoder** CNN-first, residual-style, 4–5 уровней.
3. **Context module / bottleneck** с lightweight transformer-like или equivalent global-context block.
4. **Decoder** UNet/FPN-style с skip connections.
5. **Boundary-aware enhancement blocks** внутри сети.
6. **Deep supervision outputs** на нескольких уровнях.
7. **Три output heads**:

   * `extent head`
   * `boundary head`
   * `distance head`
8. **Refine module** для повышения точности границ на финальном уровне.

## 6. Архитектурные компоненты, обязательные для v1

### 6.1. CNN-first encoder

* encoder должен быть устойчивым и хорошо работать на ограниченном объёме данных;
* предпочтителен residual encoder;
* encoder не должен требовать гигантского объёма pretraining-only данных для базовой работоспособности.

### 6.2. Multi-scale context

* модель обязана использовать multi-scale features;
* допускается pyramid-like aggregation, dilated convolutions и/или lightweight transformer block;
* pure-transformer architecture как основной baseline в v1 не принимается.

### 6.3. Edge-aware enhancement

Модель должна содержать отдельные механизмы, усиливающие извлечение границ, например:

* edge detection / edge enhancement block;
* contour-aware branch;
* dual attention / spatial-channel attention;
* low-level + high-level feature fusion.

### 6.4. Deep supervision

* deep supervision является обязательной частью архитектуры v1;
* промежуточные выходы должны участвовать в loss;
* это используется для лучшего контроля multi-scale boundary learning.

### 6.5. Refine module

* архитектура должна иметь отдельный финальный refinement stage или equivalent head для доуточнения границ;
* refine stage может быть лёгким residual encoder-decoder либо компактным correction head.

## 7. Output heads

### 7.1. `extent head`

* задача: предсказание маски поля;
* тип выхода: logits/probabilities для бинарной сегментации;
* должна учитывать ignore policy по `valid`.

### 7.2. `boundary head`

* задача: предсказание boundary-aware target;
* head должен быть совместим с boundary encoding из `module_prep_data` (`skeleton / buffer / background`);
* head является критически важной частью модели, а не второстепенным auxiliary output.

### 7.3. `distance head`

* задача: регрессия unsigned distance-to-nearest-boundary;
* head должен быть согласован с нормализацией/клиппингом distance target.

## 8. Лоссы

### 8.1. Общие требования

Финальный loss должен быть многокомпонентным и учитывать все три задачи.

### 8.2. Extent loss

Рекомендуемая базовая форма:

* region-aware segmentation loss;
* комбинация `BCE/Focal BCE + Dice` или эквивалентный устойчивый вариант.

### 8.3. Boundary loss

Boundary loss должен:

* учитывать дисбаланс классов;
* быть совместимым с `skeleton / buffer / background`;
* усиливать точность локализации узких границ.

Для v1 принимается boundary-aware loss класса `weighted CE / focal CE / contour-aware loss` над boundary encoding.

### 8.4. Distance loss

Для v1 принимается регрессионный loss типа:

* `MSE` или `SmoothL1/Huber`.

Базовая рекомендация: `SmoothL1/Huber`, если distance target клиппируется и нормализуется.

### 8.5. Deep supervision loss

Промежуточные выходы должны вносить вклад в общий loss с пониженными весами.

### 8.6. Базовые веса loss-компонент v1

Стартовый baseline:

* `extent`: `1.0`
* `boundary`: `2.5`
* `distance`: `1.0`
* `auxiliary deep supervision`: `0.3–0.5` на промежуточный выход

Эти значения считаются starting baseline, а не окончательной истиной.

## 9. Политика ignore / valid

### 9.1. Общий принцип

Все потери и метрики должны вычисляться только по корректным пикселям.

### 9.2. Обязательные правила

* invalid пиксели по `valid` исключаются из loss;
* ignore-значения в `extent` и `boundary` не участвуют в loss;
* distance loss не должен учитывать invalid пиксели и ignore-зоны;
* train metrics должны явно документировать mask of evaluated pixels.

### 9.3. Двойная роль `valid`

Слой `valid` в рамках `module_net_train` имеет две обязательные роли одновременно:

1. служебная valid-mask для ignore policy в loss, metrics и диагностике;
2. дополнительный входной канал модели в составе финального model input tensor.

Модуль не должен терять эту двойную семантику ни на уровне dataset reader, ни на уровне runtime sample assembly, ни при экспорте checkpoint metadata.

## 10. Dataset interface внутри модуля

### 10.1. Обязательные возможности

* чтение GeoTIFF-слоёв и metadata;
* сбор sample в единый tensor dict;
* поддержка `raw8` и `raw8_idx3`;
* сбор финального model input tensor с обязательным добавлением `valid` как отдельного входного канала;
* валидация согласованности между feature_mode, channel semantics и ожидаемым `in_channels`;
* строгая проверка shapes и dtypes;
* воспроизводимая работа train/val/test loaders.

### 10.2. Внутренний формат sample в runtime

Минимальный runtime sample dict:

* `image`
* `extent`
* `boundary`
* `distance`
* `valid`
* `sample_id`
* `meta`
Поле `image` в runtime sample должно означать уже собранный model input tensor, совместимый с текущим `feature_mode` и включающий `valid` как дополнительный входной канал.

При этом `valid` должен сохраняться и отдельно, поскольку он требуется не только как часть входа, но и как источник ignore/masking semantics для loss, metrics и диагностики.

## 11. Аугментации

### 11.1. Принципы

Аугментации должны улучшать устойчивость без разрушения геометрической согласованности.

### 11.2. Разрешённые базовые аугментации v1

* horizontal flip;
* vertical flip;
* rotation by 90° multiples;
* mild brightness/contrast jitter, если он корректен для данного источника;
* лёгкий additive noise только если не нарушает физический смысл спектральных каналов.

### 11.3. Ограничения

* запрещены аугментации, искажающие соответствие между image и targets;
* запрещены агрессивные color transforms без доказанной пользы для multispectral input;
* любые spatial transforms применяются синхронно ко всем таргетам.

## 12. Стратегия обучения

### 12.1. Baseline training regime

Для первой сильной версии модуль должен поддерживать:

* один официальный baseline run;
* воспроизводимый seed;
* валидацию на каждом epoch;
* checkpointing лучшей модели;
* early stopping по выбранной ключевой метрике.

### 12.2. Оптимизатор

В качестве стартового baseline принимается `AdamW`.

### 12.3. LR scheduling

Модуль должен поддерживать современный scheduler, как минимум один из:

* cosine decay;
* one-cycle;
* plateau-based fallback.

Для baseline v1 предпочтителен `cosine schedule with warmup`.

### 12.4. Mixed precision

* mixed precision training должен поддерживаться;
* при доступности оборудования baseline рекомендуется запускать с AMP.

### 12.5. Gradient stability

Модуль должен поддерживать:

* gradient clipping;
* safe handling NaN/Inf;
* логирование нестабильности.

### 12.6. Hardware-adaptive runtime policy

Модуль должен автоматически адаптироваться к доступной среде выполнения.

Базовый приоритет устройств:

* `CUDA -> MPS -> CPU`

Обязательные правила:

* AMP включается автоматически там, где это безопасно и поддерживается средой;
* при нехватке памяти модуль должен сначала деградировать по runtime-параметрам, а не ломать модельный контракт;
* предпочтительный порядок реакции на OOM:
  1. уменьшение `batch_size`;
  2. включение или увеличение `gradient_accumulation`;
  3. снижение вторичных runtime-параметров;
  4. переход к более медленному, но корректному режиму исполнения.
* ради экономии памяти нельзя:
  * убирать `valid` из model input contract;
  * отключать ignore/valid handling;
  * молча менять число каналов;
  * ломать multitask-постановку `extent + boundary + distance`.

## 13. Контроль экспериментов

### 13.1. Run structure

Каждый run должен сохранять:

* `config_used.yaml`
* `train_manifest.json`
* `history.csv`
* `best.ckpt`
* `last.ckpt`
* `eval_val.json`
* `eval_test.json` (если test запускался)
* `visuals/`
* `sample_predictions/`

### 13.2. Что сохранять в manifest

* feature_mode;
* in_channels;
* model name/version;
* loss weights;
* optimizer/scheduler settings;
* train dataset id;
* split manifest reference;
* seed;
* patch_size;
* number of epochs;
* best epoch;
* best monitored metric.
* channel semantics;
* признак использования `valid` как входного канала модели;
* resolved model input channel count;
* device requested / device resolved;
* amp requested / amp used;
* accumulation steps;
* effective batch size;
* OOM fallbacks applied.

## 14. Метрики

### 14.1. Pixel metrics

Обязательные:

* IoU
* F1
* Precision
* Recall

### 14.2. Boundary metrics

Обязательные:

* boundary F1 / edge F1
* omission/commission style metrics или эквиваленты

### 14.3. Object/structure metrics

Желательные для v1 и обязательные для последующего eval-модуля:

* GOC
* GUC
* GTC

### 14.4. Training selection metric

Для checkpoint selection не рекомендуется использовать только extent IoU.

Базовая метрика выбора лучшего checkpoint v1:

* composite metric с приоритетом boundary quality, например `0.6 * boundary_F1 + 0.4 * extent_F1`,
  либо equivalent metric, зафиксированная в конфиге.

## 15. Визуальный контроль

Модуль должен автоматически сохранять фиксированный набор preview-артефактов:

* input RGB preview;
* extent prediction vs GT;
* boundary prediction vs GT;
* distance prediction vs GT;
* сложные boundary-кейсы;
* near-invalid кейсы.

Этот набор должен быть фиксирован между run-ами для честного сравнения.

## 16. Baseline эксперименты v1

До расширения scope модуль должен официально поддерживать и сравнивать только следующие линии:

1. `raw8 + baseline_architecture`
2. `raw8_idx3 + baseline_architecture`
3. `raw8 + ablation_no_distance`
4. `raw8 + ablation_no_refine`

Все остальные эксперименты считаются вторичными до завершения baseline comparison.
В обозначениях baseline-экспериментов `raw8` и `raw8_idx3` используются как краткие имена dataset-side feature modes; фактический model input baseline v1 собирается как `raw8_valid` и `raw8_idx3_valid`.

## 17. Архитектурные ограничения первой версии

### 17.1. Что не принимается как основной baseline

* pure binary single-head segmentation model;
* pure-transformer heavy architecture;
* contour-first / polygon-sequence model как главный production path;
* foundation-model-dependent architecture как обязательный путь для MVP.

### 17.2. Что допускается как future branch

* stronger transformer bottleneck;
* self-supervised pretraining;
* domain adaptation branch;
* uncertainty modeling;
* vector-aware supervision.

## 18. Конфигурирование

### 18.1. Конфиг обязан задавать

* пути к датасету;
* feature_mode;
* in_channels;
* architecture parameters;
* loss configuration;
* optimizer/scheduler;
* augmentation policy;
* batch size;
* epochs;
* seed;
* AMP / device settings;
* channel semantics / expected model input contract;
* политику сборки финального input tensor из feature stack и `valid`;
* accumulation settings;
* runtime fallback policy under OOM.
* monitored metric;
* export policy.

### 18.2. Совместимость

Конфиг не должен содержать fallback-магии вида «если такого таргета нет, попробуй другой».
Также конфиг не должен допускать ситуацию, когда `feature_mode`, `in_channels` и фактическая семантика собранного model input расходятся, но модуль продолжает работу молча.

Любое несоответствие между dataset-side feature stack и final model input contract должно приводить к явной ошибке.

## 19. Ошибки и диагностика

Модуль обязан завершаться с явной ошибкой, если:

* число входных каналов не совпадает с feature_mode;
* dataset contract нарушен;
* shapes таргетов не согласованы;
* boundary encoding не соответствует конфигу модели;
* loss получает пустую valid область;
* checkpoint metadata несовместима с predict pipeline.
* отсутствует или некорректно описан слой `valid`;
* невозможно однозначно восстановить финальный model input contract;
* `in_channels` не согласован с assembled input tensor;
* checkpoint metadata не содержит достаточной информации для downstream predict.

## 20. Критерии приемки модуля

Модуль считается готовым для интеграции, если:

1. Способен обучить multitask-модель на датасете из `module_prep_data` без ручных правок кода.
2. Поддерживает `raw8` и `raw8_idx3` одним и тем же training pipeline.
3. Корректно использует `valid` и ignore policy во всех loss и метриках.
4. Сохраняет полный набор run-артефактов для forensic-сравнения.
5. Экспортирует checkpoint и metadata, достаточные для `module_target_predict`.
6. Позволяет честно сравнить baseline и ablation runs по фиксированным метрикам и визуальным кейсам.
7. Модуль корректно собирает финальный model input с учетом `valid` как дополнительного входного канала.
8. Checkpoint metadata и train manifest содержат достаточно информации для автоматического и безошибочного использования в `module_target_predict`.

## 21. Зафиксированные baseline-решения v1

1. Основной класс модели: `hybrid edge-aware multitask encoder-decoder`.
2. Encoder: CNN-first residual.
3. Context: lightweight global-context block / transformer-like bottleneck.
4. Heads: `extent + boundary + distance`.
5. Deep supervision: обязательно.
6. Refine module: обязательно.
7. Optimizer: `AdamW`.
8. Scheduler: `cosine with warmup`.
9. Loss weights baseline: `1.0 / 2.5 / 1.0`.
10. Checkpoint selection: boundary-aware composite metric.
11. Первый baseline comparison: `raw8` vs `raw8_idx3`.
12. `valid` является обязательной частью финального model input contract baseline v1.
13. Финальный model input contract baseline v1: `raw8_valid` или `raw8_idx3_valid`.
14. Hardware-adaptive execution является режимом по умолчанию при сохранении воспроизводимости и корректности.

## 22. Зафиксированные финальные решения v1

1. Конкретная стратегия boundary loss для baseline v1:

   * основа: `class-weighted focal cross-entropy` над boundary encoding `background / skeleton / buffer`;
   * дополнительный мягкий геометрический компонент: `soft Dice` только по `skeleton`-классу;
   * итоговая boundary loss = `focal_CE_boundary + lambda_skel_dice * soft_dice_skeleton`, где `lambda_skel_dice` задается конфигом.
2. Конкретная стратегия distance loss для baseline v1:

   * `SmoothL1 / Huber loss` по нормализованной unsigned distance map;
   * baseline-предпочтение отдается `SmoothL1`, а не `MSE`, как более устойчивому варианту к редким большим ошибкам на сложных границах.
3. Политика pretraining для baseline v1:

   * официальный baseline не зависит от generic ImageNet-only pretraining;
   * baseline считается `multispectral-first` и должен корректно обучаться без внешнего RGB-only encoder pretraining;
   * допускается optional remote-sensing-specific pretraining, если он совместим с числом и семантикой входных каналов без хрупких channel-hack адаптаций;
   * ImageNet-only pretraining не считается обязательным или предпочтительным default для 8/11-канального baseline.
   * ImageNet-only pretraining не считается обязательным или предпочтительным default для multispectral baseline с обязательным учетом `valid` в финальном model input contract.

4. Формула baseline monitored metric для выбора лучшего checkpoint:

   * `best_metric = 0.6 * boundary_F1 + 0.4 * extent_F1`.
5. Batch size baseline определяется доступной видеопамятью, но при нехватке памяти предпочтение отдается `gradient accumulation`, а не уменьшению архитектурной полноты модели.
6. Визуальный benchmark-набор должен быть фиксирован до начала baseline-comparison и не меняться между `raw8` и `raw8_idx3` run-ами.
7. `raw8` и `raw8_idx3` рассматриваются как dataset-side feature modes, а финальный вход модели собирается как `raw8_valid` или `raw8_idx3_valid`.
8. В exported checkpoint metadata обязательно фиксируются `feature_mode`, `in_channels`, `channel_semantics` и факт использования `valid` как входного канала.
9. При нехватке памяти baseline должен предпочитать runtime adaptation (`batch size`, `accumulation`, AMP policy), а не скрытую смену model input contract.

## 23. Рекомендуемый порядок реализации

1. Dataset reader/runtime sample contract.
2. Loss interface и ignore handling.
3. Baseline model skeleton.
4. Heads + deep supervision.
5. Train/eval loops.
6. Metrics and visual logging.
7. Checkpoint/export metadata.
8. Baseline run.
9. Ablation runs.
10. Freeze для перехода к `module_target_predict`.
