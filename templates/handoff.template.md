# HANDOFF

## 1. Контекст

### Проект

- **Название:** `<project_name>`
- **Модуль / зона работы:** `<module_name_or_scope>`
- **Дата:** `<YYYY-MM-DD>`
- **Автор handoff:** `<name_or_agent>`
- **Run ID / рабочий идентификатор:** `<run_id_or_work_id>`

### Связанные документы

- `main_tech.md`
- `GLOSSARY.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`

### Связанные ТЗ / спецификации модуля

- `<module_spec_1>`
- `<module_spec_2>`

---

## 2. Что было задачей

Кратко описать, что именно нужно было сделать в этой итерации.

Пример:

- реализовать `<specific_scope>`;
- не ломать `<important_constraints>`;
- сохранить совместимость с `<data_contract_or_existing_outputs>`.

---

## 3. Что сделано

Перечислить только реально завершённые изменения.

### Реализовано

- `<completed_item_1>`
- `<completed_item_2>`
- `<completed_item_3>`

### Обновлены файлы

- `<path/to/file_1>`
- `<path/to/file_2>`
- `<path/to/file_3>`

### Не менялось принципиально

- `<what_was_explicitly_left_unchanged>`

---

## 4. Что важно понимать следующему человеку / агенту

### Текущий статус

- **Состояние:** `<draft / partial / stable / ready_for_review / ready_for_next_step>`
- **Степень уверенности:** `<high / medium / low>`
- **Главный риск:** `<main_risk>`

### Ключевые проектные инварианты

- `data_contract_version = <version>`
- `feature_mode = <raw8 | raw8_idx3 | n/a>`
- `assembled_model_input = <raw8_valid | raw8_idx3_valid | n/a>`
- `valid` используется как:
  - valid-mask для ignore/loss/postprocess/eval;
  - дополнительный входной канал модели.
- скрытые fallback-логики не допускаются там, где контракт должен быть явным;
- manifests / summaries / config_used должны сохраняться обязательно.

### Что особенно нельзя сломать

- `<critical_invariant_1>`
- `<critical_invariant_2>`
- `<critical_invariant_3>`

---

## 5. Принятые решения в этой итерации

Зафиксировать только решения, которые реально были приняты.

- `<decision_1>`
- `<decision_2>`
- `<decision_3>`

Если решение временное, явно отметить:

- `<temporary_decision>`  
  **Статус:** временно  
  **Почему:** `<reason>`  
  **Когда пересмотреть:** `<condition_or_next_phase>`

---

## 6. Что осталось незавершённым

### Не доделано

- `<unfinished_item_1>`
- `<unfinished_item_2>`

### Сознательно отложено

- `<deferred_item_1>`
- `<deferred_item_2>`

### Что требует отдельного решения

- `<open_question_1>`
- `<open_question_2>`

---

## 7. Артефакты и результаты

### Конфиги

- `<path/to/config_used.yaml>`
- `<path/to/other_config_or_override>`

### Manifest / summary

- `<path/to/manifest.json>`
- `<path/to/summary.json>`

### Логи / отчёты

- `<path/to/log_file>`
- `<path/to/run_report.md>`

### Основные outputs

- `<path/to/output_1>`
- `<path/to/output_2>`
- `<path/to/output_3>`

---

## 8. Проверки и валидация

### Что проверено

- `<unit_test_or_check_1>`
- `<integration_check_2>`
- `<manual_validation_3>`

### Результат

- `<passed / partially_passed / failed>`
- Краткий комментарий: `<short_validation_comment>`

### Что не проверено

- `<not_checked_item_1>`
- `<not_checked_item_2>`

---

## 9. Известные проблемы / риски

- `<issue_or_risk_1>`
- `<issue_or_risk_2>`
- `<issue_or_risk_3>`

Для каждой критичной проблемы желательно указывать:

- **Симптом**
- **Вероятная причина**
- **Влияние**
- **Что делать дальше**

---

## 10. Что делать следующим шагом

### Рекомендуемый ближайший шаг

1. `<next_step_1>`
2. `<next_step_2>`
3. `<next_step_3>`

### Нежелательные действия сейчас

- `<dont_do_now_1>`
- `<dont_do_now_2>`

---

## 11. Если работу продолжит другой агент / разработчик

Перед продолжением он должен:

1. прочитать:
   - `DATA_CONTRACT.md`
   - `MANIFEST_SCHEMAS.md`
   - ТЗ соответствующего модуля
   - этот handoff;

2. проверить:
   - совместимость с текущим `data_contract_version`;
   - не ломается ли `feature_mode / assembled_model_input`;
   - сохраняются ли manifests / summary / config_used;
   - не появилась ли скрытая fallback-магия;

3. только после этого:
   - продолжать код;
   - менять manifests;
   - трогать runtime или compatibility logic.

---

## 12. Короткое резюме в 5–10 строк

Краткий абзац, который можно быстро прочитать без всего документа.

Пример:
> В этой итерации был(а) `<short_scope>`.  
> Основной результат: `<main_result>`.  
> Контракт данных не менялся / менялся в части `<...>`.  
> Критично сохранить `<important_invariant>`.  
> Следующий шаг — `<next_step>`.  
> Главный риск — `<risk>`.

---

## 13. Приложения (опционально)

### Команды

```bash
<command_1>
<command_2>
<command_3>
