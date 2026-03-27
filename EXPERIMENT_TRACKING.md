# EXPERIMENT_TRACKING.md

## 1. Назначение документа

Этот документ фиксирует правила **учёта, хранения, идентификации и сравнения экспериментов** проекта **«ИИ для полей»**.

Его задача:

- обеспечить воспроизводимость run-ов;
- задать единые правила именования и структуры run-артефактов;
- определить, что считается экспериментом, baseline-ом, candidate-ом и comparison run;
- зафиксировать обязательный минимум metadata для честного сравнения результатов;
- исключить ситуацию, когда метрики сравниваются между фактически несовместимыми запусками.

Этот документ дополняет:

- `DATA_CONTRACT.md` — что значат данные и какие между ними действуют контракты;
- `MANIFEST_SCHEMAS.md` — как устроены manifests и summaries;
- `REPO_CONVENTIONS.md` — где и как хранить артефакты в репозитории;
- `TESTING_STRATEGY.md` — как проверять корректность кода и пайплайна.

`EXPERIMENT_TRACKING.md` отвечает на другой вопрос:
**как документировать run-ы так, чтобы их можно было потом честно понять, воспроизвести и сравнить.**

---

## 2. Базовые определения

### 2.1. Experiment

**Experiment** — любой осмысленный запуск модуля или сквозного пайплайна, выполняемый для получения, проверки или сравнения результата.

Экспериментом считаются не только train-run-ы, но и:

- `module_prep_data` run;
- `module_net_train` run;
- `module_target_predict` run;
- `module_postprocess_vectorize` run;
- `module_eval` run;
- сквозной multi-module candidate flow;
- ablation / comparison run.

### 2.2. Run

**Run** — конкретный выполненный запуск одной конфигурации одного модуля или цепочки модулей.

### 2.3. Baseline

**Baseline** — зафиксированный run, который в текущий момент считается опорной точкой сравнения.

Baseline не обязан быть идеальным. Он обязан быть:

- воспроизводимым;
- достаточно документированным;
- пригодным для честного сравнения следующих кандидатов.

### 2.4. Candidate

**Candidate** — новый run, который сравнивается с baseline или другим reference run.

### 2.5. Comparison

**Comparison** — формализованное сопоставление двух или более run-ов по метрикам, policy, provenance и выходным артефактам.

### 2.6. Promotion

**Promotion** — явное решение, что candidate становится новым baseline или reference run.

Promotion никогда не должен происходить «молча по ощущениям».

---

## 3. Общие принципы tracking-а

### 3.1. Любой значимый run должен быть восстанавливаемым

Если run нельзя воспроизвести или хотя бы точно понять постфактум, он не должен считаться полноценным reference experiment.

### 3.2. Метрики сами по себе недостаточны

Числа без provenance, contract metadata и runtime context не считаются достаточным описанием результата.

### 3.3. Сравнение допустимо только между совместимыми run-ами

Нельзя делать вывод «модель стала лучше», если между run-ами скрыто различаются:

- feature contract;
- valid-policy;
- normalization policy;
- AOI-policy;
- postprocess policy;
- scene coverage;
- evaluation mask;
- threshold provenance;
- dataset split или GT coverage.

### 3.4. Tracking обязателен не только для train

В проекте требуется track-ить не только обучение, но и:

- подготовку данных;
- инференс;
- постпроцессинг;
- оценку;
- сквозные comparison flows.

### 3.5. Run tracking должен быть machine-readable

Главный источник истины — не заметки «вручную в чате», а versioned files:

- `config_used.*`
- `manifest.*`
- `summary.*`
- comparison artifacts
- selected previews / diagnostics

---

## 4. Что считается обязательным run artifact set

Для любого значимого run минимально обязательны:

- `config_used.*`
- `manifest.*`
- `summary.*`
- logs / diagnostics
- outputs текущего модуля

Если модуль multi-stage, допускаются stage-level manifests и summaries.

Если run участвует в сравнении, дополнительно должны сохраняться:

- comparison inputs;
- comparison summary;
- ссылка на reference run(s);
- явно зафиксированная причина сравнения.

---

## 5. Run ID policy

### 5.1. Общий формат

Каждый run должен иметь уникальный `run_id`.

Рекомендуемый baseline-формат:

```text
YYYYMMDD_HHMMSS
```

Пример:

```text
20260327_184512
```

### 5.2. Расширенный формат

При необходимости допускается расширение:

```text
YYYYMMDD_HHMMSS_<short_tag>
```

Пример:

```text
20260327_184512_raw8_valid_baseline
```

### 5.3. Требования

`run_id` должен быть:

- уникальным в пределах модуля;
- стабильным внутри всех артефактов данного run-а;
- пригодным для использования в путях, manifests и comparison reports.

### 5.4. Запрещено

Запрещено использовать неявные имена вроде:

- `new_run`
- `test_final`
- `better_model`
- `tmp`
- `last_try`

---

## 6. Структура хранения run-ов

### 6.1. Общее правило

Каждый run должен жить в собственной изолированной директории.

### 6.2. Рекомендуемый паттерн

```text
artifacts/
  module_prep_data/
    runs/
      <run_id>/
  module_net_train/
    runs/
      <run_id>/
  module_target_predict/
    runs/
      <run_id>/
  module_postprocess_vectorize/
    runs/
      <run_id>/
  module_eval/
    runs/
      <run_id>/
```

### 6.3. Внутренняя структура run directory

Минимально рекомендуемый паттерн:

```text
<run_id>/
  config_used.yaml
  manifest.json
  summary.json
  logs/
  diagnostics/
  outputs/
  previews/
```

Допускается module-specific детализация, но не должна теряться читаемость и единообразие.

---

## 7. Что именно нужно track-ить по модулям

## 7.1. `module_prep_data`

Для `module_prep_data` обязательно track-ить:

- source raster path / identity;
- vector GT path / identity;
- AOI path / identity, если AOI использовался;
- resolved CRS / spatial policy;
- feature mode;
- channel semantics;
- valid / nodata policy;
- patch size;
- sampling policy;
- split policy;
- counts по train/val/test;
- статистику отбраковок;
- exported dataset structure.

Также полезно сохранять:

- краткие preview fragments;
- label sanity previews;
- диагностические summaries по coverage / valid ratio / class balance.

## 7.2. `module_net_train`

Для `module_net_train` обязательно track-ить:

- source dataset run id;
- dataset manifest path;
- feature mode;
- assembled model input contract;
- final input channel count;
- channel semantics;
- роль `valid`;
- normalization policy и source stats;
- architecture name;
- loss composition и loss weights;
- optimizer / scheduler;
- batch size / accumulation / AMP;
- best checkpoint metric;
- history artifacts;
- best / last checkpoint paths.

Также полезно сохранять:

- curves;
- representative predictions на validation fragments;
- train diagnostics по invalid-border behaviour;
- краткие qualitative panels.

## 7.3. `module_target_predict`

Для `module_target_predict` обязательно track-ить:

- checkpoint path;
- checkpoint metadata path;
- source train run id;
- input raster identity;
- resolved feature contract;
- predict-time normalization;
- tiling policy;
- blending policy;
- invalid-only tiles skip stats;
- output rasters;
- AOI policy, если использовалась.

Также полезно сохранять:

- tile coverage diagnostics;
- preview rasters;
- seams / anti-seam sanity previews;
- processing stats по времени и памяти.

## 7.4. `module_postprocess_vectorize`

Для `module_postprocess_vectorize` обязательно track-ить:

- source predict run id;
- входные raster outputs;
- threshold policy;
- boundary repair policy;
- marker generation policy;
- watershed policy;
- filtering / merge policy;
- topology cleanup policy;
- `parcel_instance.tif`;
- final vector export path;
- output format.

Также полезно сохранять:

- previews до и после watershed;
- overlays with GT на ключевых фрагментах;
- parcel count stats;
- rule-based QC summaries.

## 7.5. `module_eval`

Для `module_eval` обязательно track-ить:

- source run ids;
- source manifest paths;
- GT sources;
- scene selection policy;
- resolved scene list;
- valid/AOI evaluation policy;
- thresholds и их provenance;
- enabled metrics;
- comparison pairs / groups;
- output reports;
- ranking summaries.

Также полезно сохранять:

- per-scene breakdown;
- bucketed metrics;
- error taxonomy summaries;
- qualitative error cases.

---

## 8. Что делает run сравнимым

Run считается **comparison-ready**, если для него можно однозначно восстановить:

- источник данных;
- feature contract;
- spatial contract;
- valid-policy;
- normalization policy;
- runtime-sensitive settings;
- source model/dataset lineage;
- полный путь от inputs до outputs.

Если хотя бы одна из этих вещей утрачена, run может оставаться полезным как exploratory, но не должен использоваться как строгий reference для выводов.

---

## 9. Обязательный comparison context

При сравнении двух или более run-ов обязательно фиксировать:

- `reference_run_id`
- `candidate_run_id`
- тип сравнения;
- цель сравнения;
- что именно изменилось между run-ами;
- что гарантированно осталось неизменным;
- какие метрики сравниваются;
- какая qualitative проверка была выполнена;
- итоговый verdict.

### 9.1. Типичные comparison types

- `data_ablation`
- `feature_ablation`
- `architecture_ablation`
- `loss_ablation`
- `train_policy_ablation`
- `predict_policy_ablation`
- `postprocess_ablation`
- `eval_policy_check`
- `full_pipeline_candidate_vs_baseline`

---

## 10. Change log для экспериментов

Каждый осмысленный candidate run должен иметь краткий **change summary**.

Минимальный формат:

```yaml
change_summary:
  purpose: string
  changed:
    - string
  unchanged:
    - string
  expected_effect: string | null
```

Пример:

```yaml
change_summary:
  purpose: compare raw8_valid vs raw8_idx3_valid on identical split
  changed:
    - feature_mode switched from raw8 to raw8_idx3
    - final input channels switched from 9 to 12
  unchanged:
    - same prep_data split
    - same architecture family
    - same training schedule
    - same eval policy
  expected_effect: test whether derived indices improve boundary-aware delineation
```

Это нужно, чтобы потом не гадать, почему run-ы вообще сравнивались.

---

## 11. Baseline promotion rules

Новый run может стать baseline только если:

- у него полный обязательный artifact set;
- он comparison-ready;
- выполнено сравнение с текущим baseline;
- зафиксировано, по каким критериям он признаётся лучше или предпочтительнее;
- promotion записан явно.

### 11.1. Promotion summary

Рекомендуется сохранять краткий promotion record:

```yaml
promotion_decision:
  previous_baseline_run_id: string | null
  new_baseline_run_id: string
  decision_date_utc: string
  rationale: []
  tradeoffs: []
```

### 11.2. Допустимые причины promotion

- лучшее качество при той же policy;
- то же качество при более простой и устойчивой конфигурации;
- тот же quality level при лучшей воспроизводимости;
- исправление критичной ошибки контракта или provenance.

---

## 12. Runtime tracking

Для runtime-sensitive модулей обязательно track-ить:

- `device_requested`
- `device_resolved`
- `amp_requested`
- `amp_used`
- `oom_fallbacks_applied`
- ключевые batch/tile parameters
- при возможности — время выполнения и memory-related notes

### 12.1. Почему это обязательно

В проекте принята hardware-adaptive policy, поэтому два run-а могут давать различное поведение не только из-за логики модели, но и из-за реально применившихся runtime решений.

Если это не записано, comparison становится слабым.

---

## 13. Data lineage

Для любого downstream run-а должна быть восстановима lineage-цепочка.

### 13.1. Минимально требуемая линия происхождения

#### Для train-run

- `prep_data_run_id`
- `prep_data_manifest_path`

#### Для predict-run

- `train_run_id`
- `train_manifest_path`
- `checkpoint_metadata_path`

#### Для postprocess-run

- `predict_run_id`
- `predict_manifest_path`

#### Для eval-run

- `source_run_ids`
- `source_manifest_paths`
- `gt_source_ids` или эквивалентные идентификаторы GT artifacts

### 13.2. Запрещено

Запрещено использовать downstream run как reference, если lineage восстановима только «по памяти» или «по переписке в чате».

---

## 14. Qualitative tracking

Кроме численных метрик, для project-level значимых run-ов рекомендуется сохранять qualitative evidence.

### 14.1. Что полезно сохранять

- overlay previews;
- boundary-focused fragments;
- cases with invalid borders;
- good cases;
- hard cases;
- known failure cases;
- before/after panels для candidate vs baseline.

### 14.2. Для чего это нужно

Это особенно важно для boundary-aware parcel delineation, где одна только aggregate metric может не показать реальные изменения в качестве внутренних границ, шумов, рваных контуров и postprocess artefacts.

---

## 15. Report levels

В проекте полезно различать три уровня отчётности.

### 15.1. Run summary

Короткое техническое резюме конкретного запуска.

### 15.2. Comparison report

Отчёт о сравнении нескольких run-ов.

### 15.3. Promotion note

Краткая фиксация, какой run теперь считается baseline и почему.

---

## 16. Recommended experiment registry

Кроме manifests по run-ам, рекомендуется иметь единый registry-файл, например:

```text
artifacts/experiment_registry.jsonl
```

или

```text
artifacts/experiment_registry.csv
```

Каждая запись registry должна содержать минимум:

- `run_id`
- `module_name`
- `created_at_utc`
- `status`
- `short_tag`
- `purpose`
- `reference_run_ids`
- `summary_path`
- `manifest_path`
- `is_baseline`

### 16.1. Назначение registry

Registry нужен не вместо manifests, а как быстрый индекс по run-ам.

---

## 17. Что считается плохой практикой

Плохой практикой считаются:

- запуск без сохранения `config_used`;
- сравнение только по одному числу без проверки policy и provenance;
- переиспользование пути старого run-а с перезаписью артефактов;
- отсутствие `run_id` в ключевых файлах;
- ручное переименование checkpoint-ов без обновления metadata;
- qualitative выводы без сохранённых preview artifacts;
- promotion baseline без formal comparison note;
- использование exploratory run как reference без доведения его до comparison-ready состояния.

---

## 18. Минимальный checklist для честного run-а

Перед тем как считать run полноценным, должно быть можно ответить на следующие вопросы:

1. Есть ли у него уникальный `run_id`?
2. Сохранён ли `config_used`?
3. Есть ли `manifest` и `summary`?
4. Понятно ли, какие входные артефакты использовались?
5. Восстановимы ли feature contract и valid-policy?
6. Зафиксированы ли runtime-sensitive решения?
7. Есть ли выходные артефакты и diagnostics?
8. Понятно ли, с чем и зачем этот run сравнивать?
9. Можно ли восстановить lineage до upstream run-ов?
10. Не противоречит ли comparison policy условиям честного сравнения?

Если ответ хотя бы на один критичный пункт отрицательный, run должен маркироваться как `exploratory_only` или аналогично.

---

## 19. Minimal status taxonomy

Рекомендуемые статусы run-а:

- `success`
- `partial`
- `failed`
- `invalidated`
- `exploratory_only`
- `comparison_ready`
- `baseline`
- `archived_reference`

Статусы могут существовать на разных уровнях:

- runtime outcome;
- comparison suitability;
- baseline status.

Главное — не смешивать их неявно.

---

## 20. Версионирование tracking policy

Tracking policy должна иметь свою версию:

```text
actionable_tracking_policy_version = "v1"
```

Изменение tracking policy требуется, если меняются:

- обязательные run artifacts;
- правила comparison readiness;
- baseline promotion rules;
- required lineage fields;
- registry schema.

---

## 21. Роль документа в разработке

Этот документ используется как опора для:

- run directory layout;
- manifest writers;
- experiment registry tooling;
- comparison scripts;
- baseline promotion decisions;
- forensic audit;
- README / handoff summaries;
- автоматизированных проверок воспроизводимости.

Код не должен создавать «удобные, но безымянные» результаты.
Любой важный результат должен входить в tracking system проекта.

---

## 22. Краткая практическая сводка

Для проекта **«ИИ для полей»** experiment tracking считается достаточным, если:

- у каждого важного run-а есть собственная директория;
- сохранены `config_used`, `manifest`, `summary`, outputs и diagnostics;
- восстановимы feature contract, valid-policy и lineage;
- comparison проводится только между совместимыми run-ами;
- baseline promotion фиксируется явно;
- qualitative evidence хранится вместе с численными результатами.

Иначе результаты могут быть полезны как исследовательские заметки, но не как надёжная инженерная основа проекта.
