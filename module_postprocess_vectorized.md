ТЗ module_postprocess_vectorize v1

1. Назначение модуля

module_postprocess_vectorize отвечает за преобразование растровых предсказаний модели в итоговые parcel instances и векторный слой полей, пригодный для анализа, QC и дальнейшей оценки.

Модуль должен быть production-like: он не просто polygonize-ит бинарную маску, а воспроизводимо использует extent, boundary, distance, valid и при наличии AOI, чтобы получить устойчивое разделение соседних полей, минимизировать ложные слияния/разделения и сохранить геопривязку и диагностируемость результата.

1. Границы ответственности
2.1. Модуль обязан
читать outputs из module_target_predict;
проверять входной контракт и совместимость raster artifacts;
учитывать valid и при наличии AOI;
выполнять thresholding и boundary-aware parcel reconstruction;
строить parcel_instance.tif;
выполнять polygonization;
выполнять консервативную топологическую очистку;
вычислять polygon_confidence;
экспортировать итоговый векторный слой и manifest постпроцессинга.
2.2. Модуль не обязан
обучать модель;
выполнять inference по исходному GeoTIFF;
пересчитывать train/predict normalization;
подменять собой module_eval;
агрессивно “дорисовывать” границы через heavy geometry editing.
2. Входные данные
3.1. Обязательные входы
extent_prob.tif
boundary_prob.tif
distance_pred.tif
valid.tif
3.2. Опциональные входы
AOI
predict_manifest.json
config_used.yaml
threshold / calibration overrides
optional diagnostics from predict stage
3. Основной принцип работы

Постпроцессинг должен не переписывать геометрию “с нуля”, а доводить boundary-derived parcel candidates до замкнутых и пригодных полигонов.

Baseline v1 строится вокруг следующей идеи:

подавить заведомо нерелевантные зоны (valid, AOI);
извлечь поддержанную extent-область;
усилить/починить границы;
получить консервативные markers;
выполнить marker-controlled watershed;
отфильтровать/слить только допустимые регионы;
polygonize-ить instance raster;
выполнить консервативную topology cleanup;
посчитать polygon_confidence.
5. Входной контракт
5.1. Обязательная согласованность

Все входные растры должны быть согласованы по:

CRS;
affine transform;
width / height;
pixel grid;
nodata / valid semantics.
5.2. Поведение при несовместимости

Если входные артефакты пространственно несовместимы, модуль должен завершаться с явной ошибкой, а не выполнять скрытый resample.

5.3. Источник истины

При наличии predict_manifest.json и config_used.yaml модуль должен использовать их как источник истины для provenance и для восстановление ожидаемой semantics входов.

1. valid / AOI policy
6.1. valid

valid.tif является обязательным входом baseline v1 и должен использоваться:

для suppression invalid-зон;
для запрета генерации parcel instances в invalid-областях;
для penalization в polygon_confidence;
для диагностики покрытия рабочего результата.
6.2. AOI

Если AOI предоставлен:

модуль должен уметь ограничивать обработку областью AOI + buffer, если это включено конфигом;
должен явно отражать выбранную AOI policy в manifest.

Если AOI отсутствует, модуль обязан оставаться полностью работоспособным.

1. Thresholding policy
7.1. Общий принцип

Thresholding в postprocess — это не отдельное “прибить всё одним порогом”, а калиброванная стадия формирования extent-support, boundary-support и marker-support масок.

7.2. Baseline v1

Для baseline v1 принимается validation-calibrated thresholding.

То есть пороги должны:

быть явно сохранены в конфиге/manifest;
происходить из validation-stage calibration либо из утверждённого baseline config;
не меняться скрыто между сравниваемыми run-ами.
7.3. Запрещено
silently брать “подходящий на глаз” порог без фиксации;
использовать разные threshold rules для сцен одного и того же run без явной причины и логирования.
8. Boundary repair
8.1. Принятое решение для v1

Для baseline v1 принимается hybrid boundary repair, а не чисто morphology-first и не чисто graph-first.

8.2. Состав baseline

Этап 1: локальная morphology-first очистка и ограниченный closing/bridging мелких разрывов внутри extent-supported области.

Этап 2: topology-aware endpoint connection только для оставшихся локальных разрывов.

Этап 3: connection разрешается только при выполнении ограничений:

максимальная дистанция;
угловая согласованность;
отсутствие запрещённых пересечений;
согласие с extent_prob.
8.3. Запрещено
глобально “сшивать” удалённые разрывы;
выполнять агрессивное graph completion без локальных ограничений;
допускать repair, который заметно меняет картографический смысл сцены.
9. Marker generation
9.1. Принятое baseline-правило

Markers должны быть консервативными внутренними ядрами полей, а не простым threshold на одной карте.

9.2. Формула baseline v1

marker generation = extent-core ∩ low-boundary ∩ high-distance

Где по смыслу:

extent-core — достаточно уверенная внутренняя часть extent;
low-boundary — зоны, где вероятность boundary невысока;
high-distance — пиксели, достаточно удалённые от вероятной границы.
9.3. Требование к markers

Markers должны быть:

устойчивыми;
консервативными;
пригодными для marker-controlled watershed;
не должны массово касаться друг друга по слабым границам.
10. Parcel reconstruction
10.1. Основной baseline

После marker generation baseline v1 использует marker-controlled watershed.

10.2. Источник “рельефа”

Рельеф для watershed должен собираться из комбинации extent/boundary/distance-derived signals, а не из одной карты без проверки.

10.3. После watershed

После получения первичных regions должны выполняться:

region filtering;
constrained merge;
suppression очевидного мусора;
контроль against invalid/AOI mask.
11. parcel_instance.tif
11.1. Обязательность

parcel_instance.tif является обязательным артефактом baseline v1.

11.2. Назначение

Он служит:

основным raster-instance output postprocess stage;
источником для polygonization;
forensic-артефактом для разборов ошибок;
опорой для object-level evaluation.
11.3. Требования
один parcel instance = один целочисленный id;
фон и invalid должны быть явно различимы;
схема кодирования должна быть задокументирована в manifest.
12. Polygonization
12.1. Основной принцип

Polygonization выполняется после получения финального parcel_instance.tif, а не напрямую по необработанной extent-mask.

12.2. Требования
polygonization должна сохранять геопривязку;
должна быть совместима с mask/NoData semantics;
должна быть воспроизводимой и детерминированной для одинаковых входов.
13. Topology cleanup
13.1. Принятое baseline-решение

Для v1 принимается только консервативный topology cleanup.

13.2. Default topology-fix для v1
удаление empty/invalid geometries;
make valid / исправление self-intersections;
удаление микродыр до порога площади;
удаление слишком маленьких островков и спайков;
нормализация multipart/singlepart policy;
консервативная topology-preserving simplification в метрах.
13.3. Что не включаем по умолчанию
агрессивный smoothing;
массовый dissolve соседних полигонов;
сильный snapping;
тяжёлую генерализацию, которая заметно двигает границы.
14. polygon_confidence
14.1. Статус

polygon_confidence входит в baseline v1 и хранится как атрибут полигона. Он нужен для QC, сортировки, optional filtering и downstream manual review.

14.2. Базовые компоненты score

Базовый score должен учитывать:

mean_extent_prob внутри полигона;
mean_boundary_prob_on_border;
mean_boundary_prob_inside_core с инверсией;
overlap_ratio с extent-support mask;
штраф за контакт с invalid;
штраф за сильную shape anomaly.
14.3. Базовая форма

confidence = w1*mean_extent_inside + w2*mean_boundary_on_border - w3*mean_boundary_inside_core + w4*overlap_ratio - penalties

Весовые коэффициенты должны храниться в конфиге.

14.4. Важное правило

По умолчанию polygon_confidence не должен использоваться как жёсткий фильтр удаления. Это QC-поле baseline v1, а не автоматический “палач” геометрий.

1. Выходные данные
15.1. Обязательные outputs
parcel_instance.tif
parcels.gpkg
postprocess_manifest.json
summary.json
config_used.yaml
15.2. Рекомендуемые diagnostics
extent_mask.tif
boundary_mask.tif
boundary_repaired.tif
marker_map.tif
parcels_preview.*
2. Форматы экспорта
16.1. Default format

Основной рабочий и архивный формат baseline v1: GPKG. SHP — только optional legacy export.

16.2. Почему не SHP по умолчанию

Shapefile неудобен как default для проекта из-за ограничений по полям, типам и общей архаичности контейнера, тогда как GPKG лучше подходит для стабильного рабочего и архивного формата. Это решение у тебя уже зафиксировано как итоговое.

16.3. Допустимые optional exports
parcels.shp — только по флагу;
GeoJSON — только для лёгкого обмена, не как archive default.
17. Manifest постпроцессинга
17.1. Модуль обязан сохранять
postprocess_manifest.json
config_used.yaml
summary.json
лог выполнения
17.2. Что должно быть в postprocess_manifest
source raster paths;
source predict run id / manifest reference;
CRS / transform summary;
AOI usage;
threshold settings;
boundary repair settings;
marker generation settings;
watershed settings;
cleanup settings;
polygon_confidence settings;
output paths;
geometry counts / area summaries;
invalid-contact summary;
runtime summary;
fallbacks applied.
18. Hardware-adaptive runtime policy
18.1. Общий принцип

Модуль должен автоматически адаптироваться к доступной среде выполнения, но без ломки логики результата.

18.2. Базовый приоритет устройств
CUDA -> MPS -> CPU
18.3. Baseline strategy under OOM

При нехватке памяти baseline должен:

сначала уменьшать chunk / batch size там, где это применимо;
затем переходить к более медленному scene-by-scene или tile-by-tile режиму;
не менять hidden образом thresholding, repair logic, marker logic или topology policy.
19. Ошибки и диагностика

Модуль обязан завершаться с явной ошибкой, если:

отсутствует хотя бы один обязательный raster input;
входные растры пространственно несовместимы;
valid отсутствует или неинтерпретируем;
thresholds / calibration settings не заданы при обязательном использовании;
невозможно построить parcel_instance.tif без нарушения входного контракта;
polygonization или cleanup дают некорректный результат без возможности безопасного fallback.
20. Критерии приемки модуля

Модуль считается готовым, если:

Корректно принимает extent_prob, boundary_prob, distance_pred, valid и optional AOI.
Воспроизводимо строит parcel_instance.tif.
Выполняет vectorization без скрытой смены правил между сценами.
Сохраняет polygon_confidence и postprocess manifest.
По умолчанию экспортирует GPKG, а не SHP.
Использует только консервативный topology cleanup.
Остаётся согласованным с upstream predict contract и downstream eval contract.
21. Зафиксированные baseline-решения v1
Входы: extent_prob.tif, boundary_prob.tif, distance_pred.tif, valid.tif, optional AOI.
Pipeline: input contract check -> valid/AOI suppression -> validation-calibrated thresholding -> hybrid boundary repair -> marker generation -> marker-controlled watershed -> region filtering / constrained merge -> parcel_instance.tif -> polygonization -> conservative topology cleanup -> polygon_confidence -> export to GPKG.
boundary repair = hybrid.
marker generation = extent-core ∩ low-boundary ∩ high-distance.
polygon_confidence обязателен как rule-based QC score.
Topology cleanup — только консервативный.
Default vector format = GPKG.
SHP — только optional.
22. Открытые решения перед кодингом
Нужен ли baseline optional export GeoJSON как явная часть v1 или только как later convenience feature.
Какой именно минимальный набор shape-anomaly penalties зафиксировать уже в первом кодинге.
Нужен ли официальный режим ablation без boundary repair для сравнительных экспериментов.
Нужна ли отдельная scene-level confidence summary поверх polygon-level confidence.
23. Рекомендуемый порядок реализации
Input contract checker.
valid / AOI suppression stage.
Thresholding utilities.
Hybrid boundary repair.
Marker generation.
Marker-controlled watershed.
Region filtering / constrained merge.
parcel_instance.tif writer.
Polygonization.
Conservative topology cleanup.
polygon_confidence.
GPKG writer + manifest.
Smoke tests and forensic diagnostics.
