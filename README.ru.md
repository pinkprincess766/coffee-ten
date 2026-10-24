[English](README.md) | **Русский**

# coffee-ten

Крошечный тензорный компилятор и проверка того, может ли аналитическая модель стоимости
заменить бо́льшую часть случайного поиска расписаний.

## Аннотация

coffee-ten компилирует небольшой тензорный граф на Python через граф-IR и loop-IR с
текстовым расписанием (split, reorder, vectorize, unroll) в C, который clang превращает в
ядро для CPU. Мы проверяем, ранжирует ли аналитическая модель стоимости (трафик кэша по
footprint циклов плюс roofline с потолками по load/store и латентности FMA) расписания
достаточно хорошо, чтобы в её top 10 было расписание, дающее не менее 90% от лучшего из 1000
случайных замеров (гипотеза A). На одном ядре Apple M5 при N = 256 первый прогон не прошёл
(отношение 0.622); после исправлений харнесса и модели прогон 2 выполнил критерий
(отношение 1.079), а прогон 3 на новой случайной выборке выполнил его вне выборки
(отношение 1.362), со Spearman 0.853 и медианной ошибкой ×1.26 между моделью и замером на
1000 случайных расписаний
([прогон 1](BENCHMARKS.md#hypothesis-a-spec-71-run-1-not-met),
[прогон 2](BENCHMARKS.md#hypothesis-a-spec-71-run-2-met),
[прогон 3](BENCHMARKS.md#hypothesis-a-spec-71-run-3-met-out-of-sample)). Написанное
вручную расписание с регистровым тайлом 4×16 достигает 94–96% измеренного пика FMA при
N ≤ 512 ([Register tile](BENCHMARKS.md#register-tile-s3)). Проверены одна машина и один
размер задачи.

## 1. Введение

Тензорные компиляторы отделяют то, что вычислять (граф), от того, как вычислять
(расписание) [3], [5], и выбирают расписания поиском, часто под управлением обученной
модели стоимости [11], [12]. coffee-ten — намеренно маленький компилятор такого рода
(один поток CPU, статические формы, бэкенд на C), созданный для проверки одного вопроса:
может ли *аналитическая* модель, без обучающих данных, ранжировать расписания
однопоточного f32 matmul достаточно хорошо, чтобы заменить бо́льшую часть случайного
поиска? Критерий, зафиксированный заранее (spec §7.1), — гипотеза A: top 10 модели
содержит расписание, дающее не менее 90% от лучшего из 1000 случайных замеров.

Вклад: компилятор, ядра которого совпадают с эталонным интерпретатором бит в бит, если
расписание сохраняет порядок суммирования, а иначе — в пределах выведенной границы
ошибки [14] ([§2.1](#21-компилятор)); аналитическая модель стоимости
([§2.3](#23-модель-стоимости)); и три записанных прогона гипотезы A с закоммиченными
сырыми данными ([§3](#3-результаты)).

## 2. Метод

### 2.1 Компилятор

Python-объект `Tensor` записывает граф (трассировка, [8], [10]); граф — это SSA-DAG со
статическими формами, который понижается в loop-IR, где цепочка поэлементных операций
становится одним гнездом циклов, а `matmul` — гнездом инициализации и гнездом обновления
`i, k, j`. Расписание преобразует loop-IR, бэкенд выдаёт C, и clang компилирует его в
ядро, загружаемое через libloading. Эталонный интерпретатор (`ct.run`) служит оракулом
для каждого скомпилированного ядра. Подробности и код, проверяющий каждый шаг:
[Приложение B](#приложение-b-основания).

### 2.2 Расписания

Расписание — это текст: шаги, разделённые `;`. Шаг указывает гнездо по номеру и циклы по
имени.

- `split(nest, loop, outer, inner, factor)`: `loop` превращается в `outer` и `inner`,
  где `inner` имеет протяжённость `factor`; остаток покрывается guard-условием.
- `reorder(nest, loop, ...)`: новый порядок циклов гнезда, начиная с внешнего; каждый
  цикл должен быть указан ровно один раз.
- `vectorize(nest, loop)`: `loop` должен быть самым внутренним циклом, не циклом
  редукции, и иметь шаг 1 в выходе.
- `unroll(nest, loop)`: любой цикл, в том числе векторизованный, если его протяжённость не
  больше 64 и он ещё не развёрнут. Как и `vectorize`, это запрос к clang (`unroll(full)`).

Гнёзда нумеруются с 0 в порядке `ct.loop_ir`. В примере гнездо 0 заполняет нулями
результат matmul, гнездо 1 — обновление `i, k, j`, к которому применяется `split`, а
гнездо 2 — relu. Недопустимый шаг вызывает `ValueError` с номером шага.

`Kernel.order_class` говорит, что расписанию разрешено менять:

- `preserving`: каждый элемент выхода получает свои слагаемые в том же порядке, поэтому
  результат совпадает с `ct.run` бит в бит.
- `reordering`: порядок суммирования float меняется, поэтому результат совпадает только в
  пределах границы из [14] (см. Основания).

### 2.3 Модель стоимости

Модель (`crates/coffee-ten-bench/src/model.rs`, v1.3) предсказывает GFLOP/s гнезда
обновления matmul, не запуская его. Трафик кэша вычисляется по footprint каждого уровня
циклов; вычислительный потолок — наименьший из пика FMA, потолка по load/store (обращения
к `C` идут в память, если не удерживаются в регистрах) и потолка латентности для цепочек
зависимых FMA; предсказание — минимум roofline [1] из вычислений и трафика. Параметры
машины — входные данные, читаются из строки `bench/peak`
([Ceilings](BENCHMARKS.md#ceilings-benchpeak-one-core)). Исправления после прогона 1,
замороженные до следующего прогона: короткие «векторизованные» циклы моделируются как
скалярные (#57), потолки латентности FMA (#58, #59, v1.2), циклы, которые clang
разворачивает полностью (#63, v1.3).
`coffee-ten-bench model compare --from <results.jsonl>` выводит модель рядом с замерами,
с ранговой корреляцией Spearman и медианной ошибкой.

### 2.4 Измерения

`bench/run.sh` снимает числа на простаивающем Mac, от сети, с чистого коммита; он
запускает `matmul` и `relu-matmul-bias` для N от 64 до 1024 и пишет
`bench/results/<date>-<commit>.jsonl`. Помимо наивного ядра он измеряет тайловое
расписание, его же с векторизацией и регистровый тайл 4x16 (`unroll` по `ii` и `ji`).
Текущие числа — в BENCHMARKS.md.
Каждый вариант перед замером проверяется против наивного ядра в пределах границы ошибки;
для каждой пары (ядро, N) делается прогрев и 30 чередующихся раундов, приводятся медиана и
p25–p75 ([Conditions](BENCHMARKS.md#conditions)).

Гипотеза A (N = 256, пространство из 7334 расписаний): измеряются 1000 случайных
кандидатов и top 10 модели; затем лучшие из каждой группы запускаются в одном
чередующемся решающем прогоне из 30 раундов, и отношение равно медиане GFLOP/s лучшего по
модели / лучшего случайного.

## 3. Результаты

**Скорость отдельных расписаний** ([Matmul](BENCHMARKS.md#matmul-a--b), [What the numbers say](BENCHMARKS.md#what-the-numbers-say)).
Векторизация тайлового matmul поднимает его примерно с 7 до 57–60 GFLOP/s, 47–50% измеренного пика FMA, при N ≤ 256; при
N = 512 это падает до 37%, при N = 1024 — до 20%; один тайлинг даёт не более 8%.
Регистровый тайл 4×16 достигает 94–96% пика для matmul при N ≤ 512 и 81% при N = 1024
(fused relu(A @ B + bias): 78–91%; одна сессия, S3).

![Fig. 1](docs/bench-gflops.png)

**Рис. 1.** GFLOP/s matmul в зависимости от N для четырёх расписаний (naive, tiled,
tiled + vec, регистровый тайл 4×16), сессия S3, медиана с p25–p75; полые маркеры —
варианты с флагом `drift`. Сырые данные `bench/results/2026-10-18-4928ffa.jsonl`;
`python3 bench/plot.py`.

![Fig. 2](docs/bench-roofline.png)

**Рис. 2.** Roofline [1] для тех же точек S3. Потолки — собственные медианы `bench/peak`
сессии S3: FMA 129.2 GFLOP/s и triad памяти 112.1 GB/s (ridge 1.15 FLOP/byte).
Арифметическая интенсивность — N / 8 FLOP/byte по обязательному трафику DRAM, это верхняя
граница реальной AI каждого варианта ([подробности](BENCHMARKS.md#benchmarks));
`python3 bench/roofline.py`.

**Гипотеза A**, везде при N = 256:

- Прогон 1 (#56): отношение 0.622, не выполнено. Spearman 0.292 по случайным кандидатам —
  число, которое позже оказалось смещено харнессом
  ([прогон 1](BENCHMARKS.md#hypothesis-a-spec-71-run-1-not-met)).
- Исправления модели и харнесса после прогона 1: короткие «векторизованные» циклы
  моделируются как скалярные (#57), потолки латентности FMA (#58, #59, модель v1.2),
  прогрев по времени 300 ms (#61), циклы, которые clang разворачивает полностью (#63,
  модель v1.3).
- Прогон 2 (#62): отношение 1.079, выполнено (модель v1.2, Spearman 0.571)
  ([прогон 2](BENCHMARKS.md#hypothesis-a-spec-71-run-2-met)).
- Прогон 3 (#64): отношение 1.362, выполнено вне выборки на новом seed (модель v1.3,
  Spearman 0.853, медианная ошибка ×1.26)
  ([прогон 3](BENCHMARKS.md#hypothesis-a-spec-71-run-3-met-out-of-sample)).
- Локальный поиск под управлением модели по соседним расписаниям на один шаг (#65): только
  код, измеренного прогона пока нет.

![Fig. 4](docs/hypothesis-a-ratio.png)

**Рис. 4.** Отношение гипотезы A по прогонам: медиана GFLOP/s лучшего из top 10 модели,
делённая на медиану лучшего из 1000 случайных кандидатов, обе из чередующегося решающего
прогона (30 раундов); полосы объединяют p25–p75 обоих. Прогон 1 0.622, прогон 2 1.079,
прогон 3 1.362, как в [BENCHMARKS.md](BENCHMARKS.md#hypothesis-a-spec-71-run-1-not-met).
Сплошная линия 1.0 (лучшее по модели = лучшее случайное), пунктир 0.9 (критерий). Сырые
данные `bench/results/*hypothesis-a*.jsonl`; `python3 bench/hypothesis_ratio.py`.

Известные ошибки модели, стабильные в прогонах 2 и 3 (#64): векторизованный `ji` = 32/64
переоценён в ×1.49, скалярный `ii` как самый внутренний — в ×1.42, короткие
«векторизованные» циклы (скалярный код) недооценены в ×0.73
([прогон 3](BENCHMARKS.md#hypothesis-a-spec-71-run-3-met-out-of-sample)).

![Fig. 3](docs/model-vs-measured.png)

**Рис. 3.** Предсказание модели v1.3 против замера для 1000 случайных кандидатов
гипотезы A, прогон 2 (синие) и прогон 3 (оранжевые), matmul N = 256, log-log, линия
y = x. Прогон 2: Spearman 0.854, медианная ошибка ×1.24 (в выборке для v1.3); прогон 3:
Spearman 0.853, медианная ошибка ×1.26 (вне выборки). Параметры машины из строки
`bench/peak` P3 (`bench/results/2026-10-21-561ede3-peak.jsonl`), на которой сняты оба
прогона. Точки и числа — вывод `coffee-ten-bench model compare`, те же, что в
[BENCHMARKS.md](BENCHMARKS.md#hypothesis-a-spec-71-run-3-met-out-of-sample);
`python3 bench/model_scatter.py`.

## 4. Ограничения

Объём доказательств: одна машина (Apple M5, одно ядро), один размер (N = 256) для
гипотезы A, seeds 1–3. Пока не сделано: исправление трёх ошибок модели выше и прогон 4 при
N = 512. Угрозы валидности замеров: [Validity](BENCHMARKS.md#validity).

Ограничения компилятора:

- В Python только `f32`. У интерпретатора есть режим f64, который биндинг не открывает.
- Статические формы: `ct.input` принимает фиксированную форму; ничего динамического.
- Один поток: в сгенерированном C нет потоков.
- Только CPU: бэкенд на C собирается clang; пути для GPU нет.
- Операторы Python: `+ - * /`, унарный `-`, `ct.relu`, `ct.maximum`, `ct.exp`, а также `@`
  или `ct.matmul` для 2-D тензоров; `ct.softmax` собран из них.
- Редукции — только `ct.sum` и `ct.max` по последней оси, с сохранением оси размера 1.
- `exp` вызывает платформенную libm (в пределах 1 ulp, без корректного округления),
  поэтому графы с `exp` проверяются против float64 в пределах выведенной границы, а не бит
  в бит.
- clang должен быть в `PATH` при запуске `ct.compile`; без него `ct.compile` вызывает
  `RuntimeError`.
- Тестируется на Linux (`ubuntu-latest`) и macOS (`macos-latest`, только при push в
  `main` и ручных запусках); Windows в CI нет.
- Гнёзда loop-IR совершенные: одна запись под одной цепочкой циклов. relu или bias после
  matmul — отдельное гнездо; epilogue fusion пока нет.

### Не цели

Конкуренция с BLAS, TVM, XLA или Inductor; обучение и autograd; разреженные тензоры;
динамические формы; свёртки и attention; типы, отличные от `f32`; собственный язык;
импорт ONNX или torch.fx; MLIR и LLVM; потоки и GPU в MVP.

## Литература

DOI проверены по Crossref (api.crossref.org) 2026-10-24. Для записей без DOI (пост в
блоге, статьи OSDI, NeurIPS и SysML, ПО, курс) дана ссылка на источник.

- [1] S. Williams, A. Waterman, D. Patterson. Roofline: an insightful visual
  performance model for multicore architectures. *CACM* 52(4), 2009.
  https://doi.org/10.1145/1498765.1498785
- [2] S. Boehm. Fast Multidimensional Matrix Multiplication on CPU from Scratch.
  Blog post, 2022. https://siboehm.com/articles/22/Fast-MMM-on-CPU
- [3] J. Ragan-Kelley, C. Barnes, A. Adams, S. Paris, F. Durand, S. Amarasinghe.
  Halide: A Language and Compiler for Optimizing Parallelism, Locality, and
  Recomputation in Image Processing Pipelines. PLDI 2013.
  https://doi.org/10.1145/2491956.2462176 (CACM version:
  https://doi.org/10.1145/3150211)
- [4] T. Chen, T. Moreau, Z. Jiang, L. Zheng et al. TVM: An Automated End-to-End
  Optimizing Compiler for Deep Learning. OSDI 2018. https://arxiv.org/abs/1802.04799
- [5] S. Feng, B. Hou, H. Jin, W. Lin et al. TensorIR: An Abstraction for
  Automatic Tensorized Program Optimization. ASPLOS 2023.
  https://doi.org/10.1145/3575693.3576933 (preprint: https://arxiv.org/abs/2207.04296)
- [6] K. Goto, R. A. van de Geijn. Anatomy of High-Performance Matrix
  Multiplication. *ACM TOMS* 34(3), 2008. https://doi.org/10.1145/1356052.1356053
- [7] J. Ansel, E. Yang, H. He, N. Gimelshein et al. PyTorch 2: Faster Machine
  Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.
  ASPLOS 2024. https://doi.org/10.1145/3620665.3640366
- [8] tinygrad. https://github.com/tinygrad/tinygrad
- [9] C. Lattner, M. Amini, U. Bondhugula, A. Cohen et al. MLIR: Scaling Compiler
  Infrastructure for Domain Specific Computation. CGO 2021.
  https://doi.org/10.1109/CGO51591.2021.9370308
- [10] R. Frostig, M. J. Johnson, C. Leary. Compiling machine learning programs
  via high-level tracing. SysML 2018.
  https://mlsys.org/Conferences/2019/doc/2018/146.pdf
- [11] T. Chen, L. Zheng, E. Yan, Z. Jiang et al. Learning to Optimize Tensor
  Programs. NeurIPS 2018. https://arxiv.org/abs/1805.08166 ; L. Zheng, C. Jia,
  M. Sun, Z. Wu et al. Ansor: Generating High-Performance Tensor Programs for Deep
  Learning. OSDI 2020. https://arxiv.org/abs/2006.06762
- [12] A. Adams, K. Ma, L. Anderson, R. Baghdadi, T.-M. Li et al. Learning to
  Optimize Halide with Tree Search and Random Programs. *ACM TOG* 38(4), 2019.
  https://doi.org/10.1145/3306346.3322967
- [13] U. Bondhugula, A. Hartono, J. Ramanujam, P. Sadayappan. A practical
  automatic polyhedral parallelizer and locality optimizer. PLDI 2008.
  https://doi.org/10.1145/1375581.1375595
- [14] N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
  SIAM, 2002. https://doi.org/10.1137/1.9780898718027
- [15] T. Chen, T. Dettmers. CMU 10-414/714: Deep Learning Systems (Needle).
  https://dlsyscourse.org/

## Приложение A. Быстрый старт

Нужны Rust (версия из `rust-toolchain.toml`; её ставит `rustup`), clang в `PATH` и
Python 3.9 или новее. pip подтягивает NumPy.

```sh
cargo test --workspace
pip install ./crates/coffee-ten-py
```

```python
import numpy as np
import coffee_ten as ct

A = ct.input("A", (4, 3))
B = ct.input("B", (3, 8))
bias = ct.input("bias", (8,))
y = ct.relu(A @ B + bias)

rng = np.random.default_rng(0)
inputs = {
    "A": rng.standard_normal((4, 3), dtype=np.float32),
    "B": rng.standard_normal((3, 8), dtype=np.float32),
    "bias": rng.standard_normal(8, dtype=np.float32),
}

print(ct.loop_ir(y))
k = ct.compile(y, schedule="split(1, j, jo, ji, 4); vectorize(1, ji)")
out = k(**inputs)
ref = ct.run(y, **inputs)
assert np.array_equal(out.view(np.uint32), ref.view(np.uint32))
print("match:", k.order_class)
```

Вывод, сокращённо:

```text
kernel kernel(A: f32[4, 3], B: f32[3, 8], bias: f32[8], t0: f32[4, 8], out: f32[4, 8])
for i in 0..4:
...
match: preserving
```

## Приложение B. Основания

Каждый раздел называет идею, на которой стоит код, где он находится и что его проверяет.
`planned` означает «ещё не реализовано». Числа в квадратных скобках — ссылки на
[Литературу](#литература).

### Алгоритм и расписание — [3], [5]

Что вычислять — это граф; как вычислять — расписание, список именованных преобразований
loop-IR (`split`, `reorder`, `vectorize`). Каждый шаг либо оставляет программу
допустимой, либо возвращает ошибку. Индексы циклов — аффинные выражения [9].
Код: `crates/coffee-ten-loop/src/schedule.rs` (текстовая форма),
`crates/coffee-ten-loop/src/apply.rs` (split, reorder, vectorize). Проверяется:
proptest-ами на равенство трасс в `crates/coffee-ten-loop/src/apply.rs`;
`crates/coffee-ten-codegen-c/tests/reorder.rs` и
`crates/coffee-ten-codegen-c/tests/vectorize.rs`.

### Граф-IR и слияние — [4], [7]

SSA-DAG со статическими формами. Цепочка поэлементных операций понижается в одно гнездо
циклов, так что сливается бесплатно; `matmul` — отдельный узел. Бэкенд выдаёт исходник на
C, а не LLVM IR. Код: `crates/coffee-ten-ir/src/graph.rs`;
`crates/coffee-ten-loop/src/lower.rs` (поэлементная цепочка становится одним гнездом;
matmul — гнёздами инициализации и обновления); `crates/coffee-ten-codegen-c` (исходник C,
clang, libloading). Проверяется: тестами `lower.rs`;
`crates/coffee-ten-codegen-c/tests/end_to_end.rs` (скомпилированные ядра совпадают с
интерпретатором бит в бит).

### Тайлинг циклов для matmul — [2], [6]

`split` и `reorder` дают кэш-тайлы, `vectorize` расширяет самый внутренний цикл `j`.
Packing не реализован. Отдельного микроядра тоже нет: регистровый тайл — обычное
расписание, `split` + `reorder` + `vectorize` + `unroll` (тайл 4×16 в `bench/run.sh`),
и clang держит этот тайл C в векторных регистрах на протяжении `k`. Код: гнездо
обновления matmul понижается в порядке `i, k, j`
(`crates/coffee-ten-loop/src/lower.rs`); тайлы получаются из `split` и `reorder`, а цикл
`j` векторизуется (`crates/coffee-ten-loop/src/apply.rs`). Проверяется:
`crates/coffee-ten-codegen-c/tests/vectorize.rs` (remarks clang `-Rpass=loop-vectorize`),
`crates/coffee-ten-codegen-c/tests/reorder.rs` и
`crates/coffee-ten-codegen-c/tests/unroll.rs` (регистровый тайл в оптимизированном IR).
Скорость: измеряется `crates/coffee-ten-bench` (варианты чередуются по раундам, медиана и
IQR), один прогон всех размеров через `bench/run.sh`; числа в BENCHMARKS.md.

### Допустимость перестановки циклов — [13]

`reorder` переставляет циклы одного гнезда. Это допустимо, когда цель записи инъективна
по циклам (достаточная проверка, а не анализ зависимостей из [13]) и тело либо не читает
других элементов целевого буфера, либо, при наличии циклов редукции, является суммой
`target + e` в тот же элемент, где `e` не читает целевой буфер. Расписание сохраняет
порядок, если каждый элемент получает свои слагаемые в том же порядке; такие результаты
f32 сравниваются бит в бит. Любое другое расписание — переупорядочивающее и сравнивается с
допуском. Код: `crates/coffee-ten-loop/src/apply.rs` (`reorder`, `is_injective`,
`OrderClass`). Проверяется: proptest-ами в `crates/coffee-ten-loop/src/apply.rs`
(порядок слагаемых для каждого элемента); `crates/coffee-ten-codegen-c/tests/reorder.rs`
(бит в бит при сохранении порядка, иначе в пределах границы).

### Границы ошибок с плавающей точкой — [14]

Допуски выводятся, а не подбираются: `|fl(xᵀy) − xᵀy| ≤ γ_n Σ|x_i y_i|` с
`γ_n = nu/(1 − nu)` и `u = 2⁻²⁴` для `f32`, плюс абсолютный член для субнормальных
произведений. Код: `crates/coffee-ten-interp` (f32 в каноническом порядке, режим f64;
broadcasting через нулевые strides, как в [15]). Проверяется:
`crates/coffee-ten-py/tests/test_diff_numpy.py` (Hypothesis против NumPy, граница с bias
как (K+1)-м членом); `crates/coffee-ten-codegen-c/tests/reorder.rs`.

### Roofline — [1]

Скорость приводится как доля от `min(peak, AI · BW)`, где пиковые FLOP/s и пропускная
способность памяти измерены на той же машине. Код: `bench/peak/peak.c` (однопоточный пик
FMA и STREAM triad из памяти и из L2). Проверяется: его режимом `--check`.
Числа: 122 GFLOP/s, 106 GB/s память, 120 GB/s L2 (BENCHMARKS.md).

### Трассирующий фронтенд — [8], [10]

Python-объект `Tensor` ничего не вычисляет; он записывает граф, который исполняется
эталонным интерпретатором (`ct.run`) или компилируется с расписанием (`ct.compile`). Код:
`crates/coffee-ten-py` (`ct.input`, операторы, `@`, `ct.relu`, `ct.maximum`,
`ct.run`, `ct.compile`, `ct.loop_ir`, `ct.exp`, `ct.sum`, `ct.max`, `ct.softmax`). Проверяется:
`crates/coffee-ten-py/tests/test_api.py`, `crates/coffee-ten-py/tests/test_softmax.py`,
`crates/coffee-ten-py/tests/test_diff_numpy.py`, `crates/coffee-ten-py/tests/test_compile.py`,
`crates/coffee-ten-py/tests/test_loop_ir.py`.

### Автотюнинг по модели стоимости (исследование) — [11], [12]

Гипотеза: top 10 расписаний по модели содержит расписание, дающее не менее 90% от
лучшего, найденного случайным поиском с 1000 замерами. Результаты прогонов: BENCHMARKS.md.
Код: `crates/coffee-ten-bench/src/space.rs` (пространство расписаний matmul),
`crates/coffee-ten-bench/src/model.rs` (аналитическая модель и `model compare`),
`crates/coffee-ten-bench/src/search.rs` (случайный поиск, top 10 модели, чередующийся
решающий прогон), `crates/coffee-ten-bench/src/local.rs` (локальный поиск под управлением
модели). Проверяется: unit-тестами в этих файлах и
`crates/coffee-ten-bench/tests/quick_run.rs`.

## Лицензия

Apache-2.0, см. `LICENSE`.
