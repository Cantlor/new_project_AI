# DATA_CONTRACT.md

## 1. Назначение документа

Этот документ фиксирует сквозной контракт данных проекта ИИ для полей.

Его задача — определить единые правила для данных, которые передаются между модулями:

- module_prep_data
- module_net_train
- module_target_predict
- module_postprocess_vectorize
- module_eval

Документ описывает:

- типы артефактов;
- обязательные слои и их семантику;
- feature contract;
- spatial contract;
- NoData / valid contract;
- dtype / normalization contract;
- manifest / summary contract;
- правила совместимости между модулями;
- случаи, когда система обязана завершаться с явной ошибкой.

Этот документ является сквозным operational contract и не заменяет детальные ТЗ модулей.  
Если возникает расхождение между реализацией и этим документом, реализация должна считаться некорректной, пока расхождение не зафиксировано отдельным архитектурным решением.

## 2. Общий pipeline и роль данных

Система работает как последовательный pipeline:

```text
module_prep_data
  -> module_net_train
  -> module_target_predict
  -> module_postprocess_vectorize
  -> module_eval
```

На каждом этапе модуль обязан:

- читать только явно определённые входные артефакты;
- выпускать явно определённые выходные артефакты;
- сохранять manifests / summaries / config_used;
- не изменять скрыто смысл данных downstream-модулям.

Контракт между модулями должен быть воспроизводимым, проверяемым и недвусмысленным.

## 3. Основные сущности данных

В проекте используются следующие основные типы данных:

### 3.1. Source raster

Исходный мультиспектральный GeoTIFF.

Базовый основной кейс v1:

- 8-band raster;
- пространственное разрешение около 3 м/пиксель;
- CRS и dtype могут быть произвольными;
- NoData может быть задан явно или требовать явной интерпретации.

### 3.2. Vector ground truth

Векторные полигоны полей, используемые для построения target-слоёв.

### 3.3. AOI

Опциональная область интереса, ограничивающая рабочую область.

### 3.4. Feature raster stack

Растровый стек признаков, используемый как основа для model input.

### 3.5. Target rasters

Растровые целевые слои для обучения:

- extent
- boundary
- distance

### 3.6. Valid raster

Отдельный слой valid, определяющий корректные пиксели и одновременно участвующий в assembled model input.

### 3.7. Predict rasters

Геопривязанные вероятностные / регрессионные предсказания модели:

- extent_prob
- boundary_prob
- distance_pred
- valid

### 3.8. Vectorized parcels

Итоговые полигональные объекты после постпроцессинга и векторизации.

## 4. Cross-module source of truth

Для разных этапов системы источником истины считаются разные артефакты:

### 4.1. Для module_net_train

Единственным поддерживаемым источником данных является экспорт module_prep_data.

### 4.2. Для module_target_predict

Основным источником истины являются:

- checkpoint metadata;
- config_used.yaml train-run;
- exported train manifest.

### 4.3. Для module_eval

Источником истины являются:

- prediction artifacts;
- vector outputs;
- manifests и summaries соответствующих source run-ов;
- GT artifacts;
- eval config.

Ни один модуль не должен опираться на «ручные догадки» там, где контракт должен быть восстановим автоматически.

## 5. Пространственный контракт

### 5.1. Raster grid contract

Растр считается эталоном пространственной сетки.

Обязательные правила:

- target rasters должны быть выровнены по grid исходного raster;
- векторные данные должны быть приведены к CRS растра;
- AOI должен быть приведён к CRS растра;
- не допускается скрытый half-pixel drift;
- любая смена transform / resolution / alignment должна быть либо запрещена, либо явно отражена в metadata.

### 5.2. CRS contract

Для любого набора согласованных артефактов должно быть однозначно понятно:

- какой CRS использовался;
- какие артефакты были reprojection targets;
- были ли reprojection / resampling выполнены;
- какой модуль и на каком этапе это сделал.

### 5.3. Spatial compatibility

Артефакты считаются spatially compatible, если:

- CRS либо совпадает, либо есть явно зафиксированное и применённое правило приведения;
- raster size / transform / resolution согласованы там, где это требуется контрактом;
- vector/raster сопоставимы в одной и той же системе координат;
- mask/target/output относятся к одной и той же пространственной области.

## 6. NoData и valid contract

### 6.1. Общий принцип

NoData не является обычным фоном.

### 6.2. Правила вычисления valid

valid должен вычисляться до любой замены NoData на fill value.

Приоритет интерпретации valid:

1. dataset/internal/sidecar mask;
2. явно заданный nodata в metadata;
3. пользовательское правило из конфигурации;
4. явная ошибка, если корректная интерпретация невозможна.

### 6.3. Представление valid

valid хранится как отдельный raster-слой в формате 0/1.

Смысл:

- 1 — пиксель допустим для использования;
- 0 — invalid / nodata / исключённая область.

### 6.4. Двойная роль valid

Слой valid имеет две обязательные роли одновременно:

1. **служебная valid-mask**  
   используется для ignore-policy в loss, metrics, diagnostics, postprocess и evaluation;
2. **дополнительный входной канал модели**  
   участвует в assembled model input contract.

Потеря любой из этих ролей считается нарушением контракта.

### 6.5. Запрещённые упрощения

Запрещено:

- молча интерпретировать invalid как обычный фон;
- скрыто вшивать valid в img и при этом терять отдельный слой;
- использовать разные правила valid в train и predict;
- менять valid-policy между run-ами без фиксации в manifest.

## 7. Feature contract

### 7.1. Dataset-side feature modes

Поддерживаются только два dataset-side режима:

- raw8
- raw8_idx3

Другие feature modes в v1 не поддерживаются.

### 7.2. Режим raw8

raw8 состоит из 8 исходных спектральных каналов в фиксированном порядке.

Обязательные свойства:

- порядок каналов должен быть явно зафиксирован;
- channel semantics должны быть записаны в config/manifest;
- hidden reordering запрещён.

### 7.3. Режим raw8_idx3

raw8_idx3 состоит из:

- 8 исходных спектральных каналов;
- 3 фиксированных derived indices.

Для v1 зафиксированы:

- NDVI
- SAVI
- NDWI

### 7.4. Assembled model input contract

Финальный вход модели не совпадает полностью с dataset-side feature mode.

Финальные baseline input contracts:

- raw8_valid = raw8 + valid
- raw8_idx3_valid = raw8_idx3 + valid

### 7.5. Число каналов

Официальные baseline значения:

- raw8_valid = 9 каналов
- raw8_idx3_valid = 12 каналов

### 7.6. Совместимость feature contract

Совместимым считается только такой runtime, где:

- feature_mode согласован с входными слоями;
- channel_semantics согласованы с фактической сборкой тензора;
- in_channels совпадает с assembled model input;
- наличие и роль valid однозначно известны.

Любое несоответствие должно приводить к явной ошибке.

## 8. Target contract

### 8.1. Обязательные target-слои train-ready sample

Каждый train-ready sample должен содержать:

- img
- extent
- boundary
- distance
- valid
- meta

### 8.2. extent

extent — raster target для общей области поля.

Семантика:

- foreground / background;
- поддержка ignore-policy обязательна;
- invalid pixels не должны участвовать в loss и метриках.

### 8.3. boundary

boundary — boundary-aware target.

Для v1 используется encoding:

- background
- skeleton
- buffer

Boundary должен строиться linework-faithful от реальных polygon boundaries.  
Boundary не должен быть суррогатно сводим к «краю extent-маски», если это ломает внутренние границы.

### 8.4. distance

distance — auxiliary distance-to-boundary target.

Для v1:

- unsigned distance-to-nearest-boundary.

### 8.5. boundary_raw

boundary_raw не является обязательным downstream input каждого модуля, но является обязательным промежуточным диагностическим артефактом подготовки данных.

## 9. Dtype и числовой контракт

### 9.1. Оригинальные входы

Оригинальные GeoTIFF не обязаны переписываться в единый disk dtype.

### 9.2. Внутреннее представление для модели

На этапе формирования model-ready данных базовое внутреннее представление:

- float32 для входных тензоров.

### 9.3. Ограничения

Запрещено:

- скрыто повреждать динамический диапазон;
- выполнять неявный кастинг без отражения этого в metadata;
- менять train-time numeric semantics в predict-time без фиксации.

### 9.4. Target dtype

Конкретные disk dtypes target-слоёв могут различаться по реализации, но их семантика должна быть однозначной и зафиксированной.

Обязательное требование:

- downstream-модуль должен понимать значения слоя без угадывания.

## 10. Нормализация и scaling contract

### 10.1. Общий принцип

Нормализация должна быть одинаковой по смыслу в train и predict.

### 10.2. Baseline v1

Базовая политика:

- valid вычисляется до преобразований;
- данные приводятся к float32;
- применяется train-derived robust normalization;
- predict должен использовать те же train-derived stats.

### 10.3. Совместимость

Нормализация считается совместимой, если:

- известен источник статистик;
- одинаковы правила clipping/scaling;
- одинаково трактуются valid/invalid пиксели;
- metadata checkpoint достаточно, чтобы predict восстановил ту же логику.

## 11. Train-ready sample contract

### 11.1. Минимальный состав sample

Каждый sample в train-ready dataset обязан иметь:

- img
- extent
- boundary
- distance
- valid
- meta

### 11.2. Семантика img

img содержит dataset-side feature stack:

- либо raw8,
- либо raw8_idx3.

img не обязан физически содержать valid внутри файла.

### 11.3. Семантика meta

meta должен содержать достаточно информации для:

- идентификации sample;
- восстановления spatial provenance;
- связи sample с patch extraction;
- диагностики sampling/quality issues.

## 12. Predict input/output contract

### 12.1. Predict input

module_target_predict принимает:

- checkpoint;
- checkpoint metadata / config_used / train manifest;
- новый GeoTIFF;
- опционально AOI и predict overrides.

### 12.2. Predict-time feature reconstruction

Predict обязан:

- восстановить feature_mode;
- при необходимости автоматически построить raw8_idx3 из 8-band raster;
- автоматически собрать assembled input contract;
- корректно добавить valid как входной канал, если checkpoint этого требует.

### 12.3. Predict outputs

Обязательные raster outputs:

- extent_prob.tif
- boundary_prob.tif
- distance_pred.tif
- valid.tif

Опционально допустимы:

- extent_logit.tif
- boundary_logit.tif
- preview artifacts

### 12.4. Predict compatibility

Predict-run считается корректным только если:

- feature contract восстановлен без ручной догадки;
- normalization соответствует train;
- invalid-only tiles не отправляются в обычный model forward;
- output rasters геопривязаны и пригодны для downstream postprocess.

## 13. Postprocess contract

### 13.1. Вход postprocess-модуля

Postprocess должен принимать:

- extent_prob
- boundary_prob
- distance_pred
- valid
- опционально AOI и related manifests/configs

### 13.2. Выход postprocess-модуля

Минимальные baseline outputs:

- parcel_instance.tif
- финальные polygon outputs в основном формате GPKG

### 13.3. Семантическая граница ответственности

Predict не должен выполнять final thresholding / watershed / polygonization.  
Postprocess не должен менять смысл upstream feature contract.

## 14. Eval contract

### 14.1. Eval inputs

module_eval должен уметь читать:

- GT raster/vector artifacts;
- prediction rasters;
- postprocess vector outputs;
- manifests и summaries source run-ов;
- metadata, достаточную для восстановления eval-relevant contract.

### 14.2. Valid-aware evaluation

Все raster metrics должны считаться только по valid пикселям с учётом ignore-policy.

Если применяется AOI-policy, её роль должна быть явно зафиксирована:

- reporting mask,
- strict evaluation mask,
- либо отсутствие AOI-policy.

### 14.3. Fair comparison contract

Нельзя сравнивать run-ы как эквивалентные, если скрыто различаются:

- valid-policy;
- AOI-policy;
- threshold provenance;
- postprocess policy;
- source GT coverage;
- scene coverage.

## 15. Manifest / summary contract

### 15.1. Каждый run обязан выпускать

Для каждого run любого модуля обязательно сохраняются:

- config_used.*
- manifest.*
- summary.*
- диагностические логи
- versioned outputs

### 15.2. Runtime-sensitive metadata

Если модуль runtime-sensitive, рекомендуется и/или обязательно сохранять:

- device_requested
- device_resolved
- amp_requested
- amp_used
- oom_fallbacks_applied

### 15.3. Manifest must be sufficient

Manifest считается корректным, если по нему можно восстановить:

- что читалось;
- что было собрано как input;
- что было записано как output;
- по каким правилам это было сделано;
- какие runtime решения были приняты;
- какие ограничения / fallback-решения реально применялись.

## 16. Naming contract

### 16.1. Основные canonical layer names

Во всех модулях должны использоваться единые canonical names:

- img
- extent
- boundary
- distance
- valid
- meta
- extent_prob
- boundary_prob
- distance_pred
- parcel_instance

### 16.2. Запрет на скрытые алиасы

Запрещено строить pipeline на неявных подстановках вида:

- «если нет extent, попробуй другой target»;
- «если нет описания valid, считаем его встроенным»;
- «если channel order не совпал, молча переставим».

Явные backward-compatibility adapters допускаются только как специально зафиксированный transitional layer, а не как скрытая магия.

## 17. Ошибки совместимости

Система обязана завершаться с явной ошибкой, если:

- невозможно однозначно интерпретировать valid;
- нарушен spatial alignment contract;
- feature_mode не поддерживается;
- in_channels не совпадает с assembled model input;
- отсутствует channel_semantics;
- target semantics не совпадают с ожидаемыми;
- train и predict используют разные normalization rules без явного переходника;
- checkpoint metadata недостаточна для predict;
- eval пытается сравнить run-ы с несовместимой valid/AOI/postprocess policy;
- входные/выходные артефакты не дают восстановить provenance.

## 18. Версионирование контракта

### 18.1. Contract version

Сквозной data contract должен иметь явную версию, например:

```text
data_contract_version = "v1"
```

### 18.2. Изменения версии

Любое изменение следующих вещей должно рассматриваться как изменение контракта:

- состав feature modes;
- assembled input contract;
- semantics valid;
- target encoding;
- normalization rules;
- обязательный состав manifests;
- canonical layer names.

## 19. Что не входит в v1 contract

В первую стабильную версию не входят:

- поддержка произвольного числа сенсоров без явного контракта;
- скрытая generalization policy между источниками данных;
- автоматическое расширение feature modes beyond raw8 / raw8_idx3;
- смешение parcel delineation и crop-type classification в одном training contract;
- foundation-model-specific input contract как обязательная часть baseline pipeline.

## 20. Краткая таблица baseline v1

| Сущность | Baseline v1 |
|---|---|
| Pipeline | prep_data -> train -> predict -> postprocess -> eval |
| Dataset-side feature modes | raw8, raw8_idx3 |
| Derived indices | NDVI, SAVI, NDWI |
| Assembled input contracts | raw8_valid, raw8_idx3_valid |
| Input channels | 9 или 12 |
| Mandatory train sample layers | img, extent, boundary, distance, valid, meta |
| Mandatory predict outputs | extent_prob, boundary_prob, distance_pred, valid |
| Boundary encoding | background / skeleton / buffer |
| Distance target | unsigned distance-to-nearest-boundary |
| Valid semantics | mask + model input channel |
| Main raster output format | GeoTIFF |
| Main vector output format | GPKG |
| Run artifacts | config_used, manifest, summary, logs, versioned outputs |

## 21. Роль документа в разработке

DATA_CONTRACT.md используется как обязательная опора для:

- реализации readers/writers;
- validators;
- dataset assembly;
- checkpoint metadata export;
- predict-time feature reconstruction;
- postprocess input validation;
- eval provenance validation;
- unit/integration/e2e tests.

Новый код не должен нарушать этот контракт молча.  
Если реализация требует изменения контракта, сначала обновляется архитектурное решение и документация, и только потом код.
