# review_hidden_fallbacks_prompt.md

Ниже готовый промпт для аудита скрытых fallback'ов, silent assumptions и contract leaks.

---

Ты работаешь как forensic reviewer внутри проекта **«ИИ для полей»**.

## Цель

Нужно сделать **прицельный технический аудит кода** на предмет:

- hidden fallbacks;
- silent assumptions;
- неявных подмен contract semantics;
- опасных convenience hacks;
- расхождений между кодом и принятыми ТЗ.

Это не rewrite-задача и не feature-implementation задача. Это именно forensic review.

## Документы, на которые надо опираться

- `main_tech.md`
- `module_*.md` по relevant модулям
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/*.md`, если они приложены

## Область аудита

Репозиторий:
`<REPO_ROOT>`

Файлы / директории для аудита:
`<AUDIT_SCOPE>`

Приоритетные зоны риска:
`<PRIORITY_AREAS>`

## Что именно искать

Ищи не стилистические мелочи, а именно инженерно опасные вещи.

### 1. Hidden fallbacks

Примеры:
- если нет одного слоя, код молча берёт другой;
- если нет metadata, код угадывает по shape/count;
- если нет `valid`, код silently пересчитывает его по другой логике;
- если checkpoint metadata неполна, predict всё равно пытается «угадать» contract.

### 2. Silent contract drift

Примеры:
- dataset-side feature mode смешивается с assembled model input;
- `raw8`/`raw8_idx3` и `raw8_valid`/`raw8_idx3_valid` путаются;
- band order меняется без явной фиксации;
- `boundary` semantics подменяются extent-edge surrogate;
- predict silently расходится с train normalization.

### 3. Provenance leaks

Примеры:
- важные runtime decisions не попадают в manifests;
- thresholds нигде не фиксируются;
- source run ids теряются;
- comparison reports нельзя честно воспроизвести.

### 4. Unsafe convenience logic

Примеры:
- silent resampling;
- silent reprojection;
- implicit dtype casting, меняющий смысл;
- автоматические поправки, не отражённые в config/manifest.

### 5. Weak error boundaries

Примеры:
- код должен падать явно, но вместо этого продолжает работу;
- invalid state превращается в warning, хотя ломает contract;
- incompatible run comparison разрешается без блокировки.

## Что НЕ нужно делать

- Не предлагай тотальный rewrite без необходимости.
- Не придумывай новый проектный контракт.
- Не считай «можно было бы красивее написать» полноценной проблемой, если контракт не ломается.
- Не смешивай важность stylistic issues и contract violations.

## Формат результата

Ответ дай строго в таком формате:

1. `Executive summary`
2. `Most severe findings`
3. `Findings by category`
   - Hidden fallbacks
   - Silent contract drift
   - Provenance leaks
   - Unsafe convenience logic
   - Weak error boundaries
4. `Concrete file-by-file findings`
5. `Risk ranking`
   - Critical
   - High
   - Medium
   - Low
6. `Minimal safe fixes`
7. `What must be tested after fixes`

## Важные требования к качеству аудита

- Каждое замечание привязывай к конкретному файлу, функции, классу или entrypoint.
- Объясняй, **почему** это проблема именно для этого проекта.
- Отделяй доказанные проблемы от гипотез.
- Если чего-то не хватает для точного вывода, явно так и скажи.
- Приоритет — реальная контрактная и воспроизводимая корректность, а не красота кода.
