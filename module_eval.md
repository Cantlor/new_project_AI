# ТЗ `module_eval` v1

## 1. Назначение модуля

`module_eval` отвечает за единую, воспроизводимую и сопоставимую оценку качества всей системы выделения сельскохозяйственных полей.

Модуль должен обеспечивать:

* оценку качества raster-предсказаний и итоговых vector-полигонов;
* сравнение экспериментов (`raw8`, `raw8_idx3`, ablations, разные postprocess-конфиги);
* раздельный анализ pixel-level, boundary-level и object/structure-level качества;
* разбор ошибок по типам сцен и сложности участков;
* выпуск стандартных отчётов для технического анализа и принятия решений.

## 2. Границы ответственности

### 2.1. Модуль обязан

* читать ground truth и outputs из `module_net_train`, `module_target_predict`, `module_postprocess_vectorize`;
* считать согласованный набор метрик;
* строить срезы по сложности/типам ошибок;
* сравнивать несколько run-ов между собой;
* сохранять отчёты, таблицы, diagnostics и manifests.
* сохранять provenance-данные, достаточные для полного воспроизведения eval-запуска;
* явно фиксировать, какие именно source run-ы, manifests, thresholds и scene lists участвовали в сравнении.

### 2.2. Модуль не обязан

* обучать модель;
* выполнять inference;
* выполнять postprocess/vectorization;
* автоматически выбирать «лучшую» модель без явно заданного правила.

## 3. Основной принцип оценки

### 3.1. Недостаточность одной метрики

Оценка качества не должна сводиться к одному IoU/F1.

### 3.2. Зафиксированная схема оценки v1

Для v1 оценка делится на три обязательные группы:

1. `global/pixel metrics`
2. `boundary metrics`
3. `object/structure metrics`

### 3.3. Дополнительный обязательный слой анализа

Помимо агрегированных средних значений, модуль обязан поддерживать разбиение по:

* сложности parcel-сцены;
* размеру объектов;
* качеству границ;
* близости к invalid/AOI границам;
* типу ошибки (over-segmentation / under-segmentation / boundary shift / spurious objects).

## 4. Источники данных для оценки

### 4.1. Ground truth

Модуль должен поддерживать:

* raster GT (`extent`, `boundary`, optional `distance`);
* vector GT (полигоны участков).

### 4.2. Предсказания

Модуль должен уметь читать:

* raster predictions (`extent_prob`, `boundary_prob`, `distance_pred`, optional masks/logits);
* vector outputs (`parcels.gpkg` и эквиваленты);
* manifests и summaries соответствующих run-ов.
* metadata, достаточную для восстановления eval-relevant contract: feature_mode / assembled input contract, valid/AOI policy, postprocess policy, export semantics и source run identifiers.

### 4.3. Пространственный контракт

* CRS должен быть согласован;
* spatial alignment raster↔GT должен быть проверяемым;
* для vector metrics должно быть определено правило пространственного сопоставления объектов.

### 4.4. Provenance contract

Для каждого eval-run должно быть однозначно известно:

* какие GT-артефакты использовались;
* какие prediction/vector outputs использовались;
* из каких source run-ов они получены;
* какие manifests/configs были прочитаны как источник правил;
* какие scenes вошли в расчёт, а какие были исключены и почему.

Eval не должен сравнивать артефакты с неясным происхождением как будто они сопоставимы по умолчанию.

## 5. Режимы оценки

### 5.1. Raster-only evaluation

Оценка только вероятностных/бинарных raster outputs.

### 5.2. Vector-only evaluation

Оценка итоговых polygon outputs против GT polygons.

### 5.3. End-to-end evaluation

Сравнение полного pipeline от raster inference до final vector parcels.

### 5.4. Multi-run comparison

Сравнение нескольких run-ов по единому набору метрик и фиксированным тестовым сценам.

## 6. Pixel / global metrics

### 6.1. Обязательные pixel metrics v1

Для `extent` как минимум:

* IoU
* F1-score
* Precision
* Recall
* Overall Accuracy (optional but recommended)

### 6.2. Boundary raster metrics

Для `boundary` raster как минимум:

* boundary F1 / edge F1
* boundary precision
* boundary recall

### 6.3. Distance auxiliary evaluation

Для `distance_pred` как минимум:

* MAE
* RMSE
* optional rank/correlation summary

### 6.4. Valid-aware evaluation

Все raster metrics должны считаться только на valid пикселях и с учётом ignore-policy.
Если в pipeline дополнительно использовалась AOI-policy, eval обязан явно фиксировать, применяется ли она только как reporting mask, как strict evaluation mask или не применяется вовсе.

Смена valid/AOI evaluation policy между run-ами без явной фиксации должна считаться нарушением fair-comparison protocol.

## 7. Boundary metrics

### 7.1. Основной принцип

Для задач delineation качество границы должно измеряться отдельно от общей площади маски.

### 7.2. Обязательные boundary metrics v1

* `Boundary F1`
* `Boundary Precision`
* `Boundary Recall`
* `Boundary Displacement Error (BDE)` или эквивалентная метрика физического смещения границы
Если метрика boundary displacement считается в физических единицах, baseline v1 должен предпочитать метры как основную единицу отчётности при корректной геопривязке.

Используемая единица измерения и способ перевода pixel -> meters должны фиксироваться в eval manifest.

### 7.3. Дополнительные boundary diagnostics

Рекомендуется поддерживать:

* average boundary offset in meters;
* omission / commission near boundaries;
* continuity / closure proxy.

## 8. Object / structure metrics

### 8.1. Основной принцип

Модуль должен измерять не только совпадение пикселей, но и качество разделения parcel instances.

### 8.2. Обязательные object/structure metrics v1

* `Global Over-segmentation / Over-Classification Error (GOC)`
* `Global Under-segmentation / Under-Classification Error (GUC)`
* `Global Total Classification Error (GTC)`

### 8.3. Желательные дополнительные object metrics

* object-level precision/recall after spatial matching;
* match rate between GT parcels and predicted parcels;
* count-based split / merge diagnostics.

### 8.4. Обязательный разбор ошибок структуры

Модуль должен отдельно оценивать:

* ложные слияния соседних полей;
* ложные splits одного поля;
* пропущенные поля;
* спонтанные мелкие ложные объекты.

## 9. Spatial matching policy для vector evaluation

### 9.1. Требование

Для vector/object evaluation должна быть определена воспроизводимая политика сопоставления GT и predicted polygons.

### 9.2. Baseline policy v1

В baseline принимается matching policy на основе overlap / IoU с правилами:

* один GT объект может быть сопоставлен одному основному predicted объекту;
* множественные predicted объекты, покрывающие один GT, трактуются как split-case;
* один predicted объект, покрывающий несколько GT, трактуется как merge-case.

### 9.3. Параметры matching

* min IoU / overlap threshold должны задаваться конфигом;
* все thresholds и tie-breaking rules должны фиксироваться в manifest.
* также в manifest должны фиксироваться правила обработки edge-cases:
  * partial overlaps;
  * tiny sliver polygons;
  * invalid geometries после cleanup;
  * parcels, пересекающих границы scene/AOI.
  
## 10. Bucketing по сложности сцен и объектов

### 10.1. Основной принцип

Модуль должен уметь разрезать результаты по сложности, а не только считать одно среднее число.

### 10.2. Обязательные buckets v1

* по размеру объекта (`small`, `medium`, `large`);
* по степени фрагментированности / shape complexity;
* по boundary difficulty (`weak boundary`, `strong boundary`);
* по proximity к invalid/AOI границам;
* по типу ландшафтного окружения, если это доступно из metadata/annotations.

### 10.3. Parcels complexity score

Модуль должен поддерживать вычисление или чтение `parcel complexity score` / equivalent grouping, если это доступно из dataset metadata.

## 11. Error taxonomy

### 11.1. Обязательная taxonomy v1

Модуль должен классифицировать ошибки минимум по следующим классам:

* `merge error`
* `split error`
* `boundary shift`
* `missed parcel`
* `spurious parcel`
* `invalid-area artifact`

### 11.2. Назначение

Эта taxonomy нужна не только для отчёта, но и для сравнения run-ов по профилю ошибок.

## 12. Evaluation protocols

### 12.1. Baseline protocol v1

Для каждого официального run должны считаться:

* aggregate metrics по всей test/val выборке;
* per-scene metrics;
* bucketed metrics;
* error taxonomy summary;
* visual diagnostics.
Baseline protocol должен дополнительно фиксировать:

* список реально оценённых scenes;
* список пропущенных scenes и причину пропуска;
* правила binarization / thresholding, если raster metrics считаются не по probability summaries, а по бинаризованным outputs;
* источник threshold values: config, validation calibration или inherited postprocess settings.

### 12.2. Fair comparison rule

Сравнение run-ов должно выполняться:

* на одном и том же split;
* на одном и том же списке сцен;
* при одной и той же evaluation config.

### 12.3. Что запрещено

* сравнивать run-ы на разных сценах как будто это одно и то же качество;
* менять matching thresholds между экспериментами без фиксации;
* скрыто переключать valid-mask или AOI-policy между eval-запусками.

## 13. Input contract checks

### 13.1. Модуль обязан валидировать

* наличие всех обязательных файлов;
* согласованность CRS;
* согласованность raster shapes/transforms;
* наличие manifests/configs;
* корректность geometry в vector GT/predictions.
* согласованность source run manifests с eval-конфигом;
* наличие достаточной информации о valid/AOI policy;
* согласованность scene identities между GT и predictions;
* согласованность units/scale для boundary-distance-like метрик.

### 13.2. Ошибки

При нарушении eval contract модуль обязан завершаться с явной ошибкой, а не продолжать «как получится».

## 14. Отчётность

### 14.1. Основные отчёты

Модуль обязан сохранять:

* `eval_manifest.json`
* `summary.json`
* `metrics_aggregate.csv/json`
* `metrics_per_scene.csv`
* `metrics_by_bucket.csv`
* `error_taxonomy.json`
* `comparison_report.*` (если это multi-run режим)
* `scenes_included.csv/json`
* `scenes_excluded.csv/json`
* `eval_config_resolved.json`
* `source_runs.json`

### 14.2. Визуальные diagnostics

Модуль должен сохранять:

* overlay GT vs prediction;
* boundary error heatmaps / boundary mismatch previews;
* merge/split case galleries;
* top-K worst scenes;
* top-K most improved scenes при multi-run comparison.
Желательно, чтобы visual diagnostics сохраняли не только изображения, но и machine-readable индексы сцен/кейсов, по которым эти изображения были выбраны.

## 15. Multi-run comparison

### 15.1. Поддерживаемые сценарии

Модуль должен уметь сравнивать как минимум:

* `raw8` vs `raw8_idx3`
* baseline vs ablation
* baseline postprocess vs modified postprocess
* run A vs run B по фиксированному сценарию eval
В comparison mode обозначения `raw8` и `raw8_idx3` могут использоваться как краткие имена экспериментальных линий, но eval manifest должен хранить полный resolved contract соответствующего run-а.

Для baseline v1 это особенно важно, если upstream-модули различают dataset-side feature modes и финальный assembled model input contract.

### 15.2. Выход comparison mode

* delta tables по всем основным метрикам;
* список выигрышей/проигрышей по сценам;
* список bucket’ов, где изменение особенно полезно или вредно.

## 16. Ranking policy

### 16.1. Основной принцип

Модуль не должен навязывать одну абсолютную «лучшую» модель без явного правила.

### 16.2. Baseline ranking summary v1

По умолчанию модуль должен формировать ranking summary на основе:

* `boundary-aware composite score`
* object/structure penalties
* optional custom business rule

### 16.3. Рекомендуемая composite summary metric

Для общего техничeского summary рекомендуется:

* `0.4 * extent_F1 + 0.3 * boundary_F1 + 0.3 * (1 - normalized_GTC)`

Эта метрика не заменяет полный отчёт, а служит только для краткого ranking summary.

## 17. Конфигурирование

### 17.1. Конфиг обязан задавать

* пути к GT и predictions;
* eval mode (`raster`, `vector`, `end-to-end`, `comparison`);
* valid/AOI policy;
* thresholds для matching;
* bucket definitions;
* metric selection;
* export/report settings.
* scene selection policy;
* reporting units для distance/boundary-shift metrics;
* threshold provenance policy для raster/vector evaluation;
* правила включения/исключения scenes при contract violations;
* comparison source manifests / run identifiers.

### 17.2. Manifest-first policy

Все параметры eval должны сохраняться в manifest для полной воспроизводимости.
В manifest также должны сохраняться:

* source run identifiers;
* hashes/paths исходных manifests и configs;
* resolved scene list;
* resolved thresholds;
* resolved valid/AOI policy;
* runtime mode (`cpu`, parallel/chunked mode, scene-by-scene fallback и т.д.).

## 18. Hardware and runtime policy

### 18.1. Общий принцип

Модуль должен быть hardware-adaptive, но приоритет — воспроизводимость и корректность.

### 18.2. Baseline rules

* CPU execution считается нормальным baseline для eval;
* параллелизм и chunking допустимы, если не меняют результат;
* при нехватке памяти допустим scene-by-scene evaluation.
* любые runtime-оптимизации допустимы только если они не меняют итоговый набор сцен, policy masking и численные результаты метрик сверх заранее допустимого машинного допуска;
* если применяется fallback runtime mode, это должно быть явно отражено в eval manifest.

## 19. Критерии приемки модуля

Модуль считается готовым, если:

1. Считает pixel, boundary и object/structure metrics на одном и том же наборе сцен.
2. Корректно использует valid/ignore-policy.
3. Может сравнивать минимум два run-а по фиксированному config.
4. Выпускает воспроизводимые отчёты и visual diagnostics.
5. Даёт отдельную информацию о merge/split/boundary-shift ошибках.
6. Один и тот же eval config на одном и том же наборе входов даёт воспроизводимый результат и одинаковый состав scene coverage.
7. Отчёты содержат достаточно provenance-информации, чтобы восстановить, что именно и по каким правилам сравнивалось.

## 20. Зафиксированные baseline-решения v1

1. Оценка делится на `global/pixel`, `boundary`, `object/structure` группы.
2. Boundary metrics обязательны и не заменяются общей IoU.
3. Object/structure metrics включают как минимум `GOC`, `GUC`, `GTC`.
4. `BDE` включается как baseline boundary-shift metric.
5. Bucketed evaluation обязательна.
6. Error taxonomy обязательна.
7. Multi-run comparison является частью baseline design, а не дополнительной опцией.
8. Ranking summary использует composite boundary-aware score, но не заменяет полный отчёт.
9. Provenance tracking является обязательной частью baseline eval design.
10. Valid/AOI evaluation policy должна быть явно зафиксирована и не может меняться между сравниваемыми run-ами скрыто.
11. Threshold provenance для matching и raster binarization должна фиксироваться в manifest/reporting artifacts.

## 21. Открытые решения перед кодингом

1. Нужна ли строгая поддержка статистической значимости / bootstrap CIs уже в v1.
2. Какие bucket thresholds по size/complexity зафиксировать по умолчанию.
3. Делать ли baseline HTML-report наряду с CSV/JSON.
4. Нужен ли официальный leaderboard-format для внутренних экспериментов.
5. Нужна ли официальная baseline policy для probability-threshold-free summaries (например, PR-style или calibration-aware diagnostics) уже в v1.

## 22. Рекомендуемый порядок реализации

1. Input contract validator.
2. Raster metrics.
3. Boundary metrics.
4. Vector matching + object/structure metrics.
5. Bucketing and error taxonomy.
6. Report writers.
7. Visual diagnostics.
8. Multi-run comparison.
9. Ranking summary.
10. Freeze eval protocol.
