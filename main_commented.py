"""
Лабораторная работа: сравнение методов решения СЛАУ.
"""

import csv
import time
from pathlib import Path

import matplotlib

# Режим Agg нужен, чтобы графики можно было сохранять в файл
# даже без графического интерфейса.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Фиксированный seed нужен для воспроизводимости экспериментов.
SEED = 42

# Очень маленькое число для проверки, что элемент не слишком близок к нулю.
EPS = 1e-18


# =========================
# Генерация тестовых данных
# =========================

def generate_well_conditioned_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Генерирует случайную матрицу размера n x n.

    Сначала создаётся случайная матрица с элементами из [-1, 1],
    затем к ней прибавляется n * I
    """
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    a += n * np.eye(n)
    return a.astype(float)



def generate_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    """Генерирует случайный вектор правой части длины n."""
    return rng.uniform(-1.0, 1.0, size=n).astype(float)



def hilbert_matrix(n: int) -> np.ndarray:
    """
    Строит матрицу Гильберта размера n x n.
    Матрица Гильберта плохо обусловлена, поэтому на ней удобно
    проверять устойчивость методов.
    """
    h = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            h[i, j] = 1.0 / (i + j + 1)
    return h


# =====================================
# Решение треугольных систем подстановкой
# =====================================

def forward_substitution(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
методом прямой подстановки.
    """
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0

        for j in range(i):
            s += l[i, j] * y[j]

        if abs(l[i, i]) < EPS:
            raise ValueError('Zero diagonal in forward substitution')

        y[i] = (b[i] - s) / l[i, i]

    return y



def backward_substitution(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
     методом обратной подстановки.
    """
    n = len(y)
    x = np.zeros(n, dtype=float)


    for i in range(n - 1, -1, -1):
        s = 0.0


        for j in range(i + 1, n):
            s += u[i, j] * x[j]

        if abs(u[i, i]) < EPS:
            raise ValueError('Zero diagonal in backward substitution')

        x[i] = (y[i] - s) / u[i, i]

    return x


# ==============================
# Метод Гаусса без выбора pivot
# ==============================

def gauss_no_pivot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
     методом Гаусса без выбора ведущего элемента.

    Недостаток метода: если ведущий элемент очень мал,
    точность может резко ухудшиться или метод вообще остановится.
    """
    # Работаем с копиями, чтобы не портить исходные A и b.
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    # k — номер текущего столбца и ведущего элемента.
    for k in range(n - 1):
        pivot = a[k, k]

        if abs(pivot) < EPS:
            raise ValueError('Zero pivot in Gauss without pivoting')

        # Зануляем элементы ниже pivot в столбце k.
        for i in range(k + 1, n):
            factor = a[i, k] / pivot

            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    # После прямого хода матрица становится верхнетреугольной.
    return backward_substitution(a, b)


# =========================================
# Метод Гаусса с частичным выбором pivot
# =========================================

def gauss_partial_pivot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ методом Гаусса с частичным выбором
    главного элемента по столбцу.

    На каждом шаге выбирается максимальный по модулю элемент
    в текущем столбце среди строк ниже и включая текущую.
    Затем эта строка поднимается вверх.
    """
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    for k in range(n - 1):

        pivot_row = k + np.argmax(np.abs(a[k:, k]))

        if abs(a[pivot_row, k]) < EPS:
            raise ValueError('Singular matrix in Gauss with pivoting')

        # Если лучшая строка не текущая, меняем строки местами.
        if pivot_row != k:
            a[[k, pivot_row]] = a[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        pivot = a[k, k]

        # Зануляем элементы ниже диагонали в текущем столбце.
        for i in range(k + 1, n):
            factor = a[i, k] / pivot
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    return backward_substitution(a, b)


# =========================
# LU-разложение без pivot
# =========================

def lu_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    LU-разложение матрицы A без перестановок:
    """
    a = a.astype(float)
    n = a.shape[0]

    # L начинаем с единичной матрицы.
    l = np.eye(n, dtype=float)

    # U сначала заполняем нулями.
    u = np.zeros((n, n), dtype=float)

    for i in range(n):
        # Сначала вычисляем i-ю строку матрицы U.
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += l[i, k] * u[k, j]
            u[i, j] = a[i, j] - s

        # Если диагональный элемент U слишком мал, продолжать нельзя.
        if abs(u[i, i]) < EPS:
            raise ValueError('Zero pivot in LU decomposition')

        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += l[j, k] * u[k, i]
            l[j, i] = (a[j, i] - s) / u[i, i]

    return l, u



def solve_lu(l: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решает систему Ax = b, если уже известно разложение A = LU
    """
    y = forward_substitution(l, b)
    x = backward_substitution(u, y)
    return x


# =========================
# Метрики точности решения
# =========================

def relative_error(x_true: np.ndarray, x_calc: np.ndarray) -> float:
    """
    Относительная погрешность решения:
    """
    return float(np.linalg.norm(x_calc - x_true) / np.linalg.norm(x_true))



def residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ x - b))


# ======================
# Сохранение результатов
# ======================

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Сохраняет список словарей в CSV-файл."""
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ==========================
# Эксперимент 1: одна система
# ==========================

def experiment_one_system(out_dir: Path) -> list[dict]:
    """
    Сравнивает время решения одной СЛАУ для разных размеров матрицы.
    """
    rng = np.random.default_rng(SEED)
    sizes = [100, 200, 400, 600]
    rows = []

    for n in sizes:
        a = generate_well_conditioned_matrix(n, rng)
        b = generate_vector(n, rng)

        # Замер времени для Гаусса без выбора pivot.
        t0 = time.perf_counter()
        x1 = gauss_no_pivot(a, b)
        t1 = time.perf_counter()

        # Замер времени для Гаусса с частичным выбором pivot.
        t2 = time.perf_counter()
        x2 = gauss_partial_pivot(a, b)
        t3 = time.perf_counter()

        # Замер времени для LU: отдельно разложение и отдельно решение.
        t4 = time.perf_counter()
        l, u = lu_decomposition(a)
        t5 = time.perf_counter()
        x3 = solve_lu(l, u, b)
        t6 = time.perf_counter()

        rows.append({
            'n': n,
            'gauss_no_pivot_sec': round(t1 - t0, 6),
            'gauss_partial_pivot_sec': round(t3 - t2, 6),
            'lu_decomposition_sec': round(t5 - t4, 6),
            'lu_substitution_sec': round(t6 - t5, 6),
            'lu_total_sec': round(t6 - t4, 6),
            'residual_gauss_no_pivot': f'{residual_norm(a, x1, b):.6e}',
            'residual_gauss_partial_pivot': f'{residual_norm(a, x2, b):.6e}',
            'residual_lu': f'{residual_norm(a, x3, b):.6e}',
        })

    write_csv(out_dir / 'results_one_system.csv', rows, list(rows[0].keys()))
    return rows


# ==========================================
# Эксперимент 2: несколько правых частей b
# ==========================================

def experiment_many_rhs(out_dir: Path) -> list[dict]:
    """
    Показывает преимущество LU-разложения при одной и той же матрице A,
    но нескольких правых частях b.

    Сравниваем:
    - многократный запуск Гаусса с выбором pivot,
    - однократное LU-разложение + решение для каждого b.
    """
    rng = np.random.default_rng(SEED)
    n = 200
    a = generate_well_conditioned_matrix(n, rng)
    ks = [1, 5, 20]
    rows = []

    for k in ks:
        # Генерируем k разных правых частей.
        bs = [generate_vector(n, rng) for _ in range(k)]

        # Каждый раз решаем заново методом Гаусса с выбором элемента.
        t0 = time.perf_counter()
        for b in bs:
            gauss_partial_pivot(a, b)
        t1 = time.perf_counter()

        # Один раз строим LU-разложение.
        t2 = time.perf_counter()
        l, u = lu_decomposition(a)
        t3 = time.perf_counter()

        # Для каждой правой части решаем только треугольные системы.
        for b in bs:
            solve_lu(l, u, b)
        t4 = time.perf_counter()

        rows.append({
            'n': n,
            'k': k,
            'gauss_partial_pivot_total_sec': round(t1 - t0, 6),
            'lu_decomposition_sec': round(t3 - t2, 6),
            'lu_all_substitutions_sec': round(t4 - t3, 6),
            'lu_total_sec': round(t4 - t2, 6),
        })

    write_csv(out_dir / 'results_many_rhs.csv', rows, list(rows[0].keys()))
    return rows


# =========================================
# Эксперимент 3: точность на матрице Гильберта
# =========================================

def experiment_hilbert(out_dir: Path) -> list[dict]:
    """
    Проверяет точность методов на плохо обусловленной матрице Гильберта.
    """
    rows = []

    for n in [5, 10, 15]:
        h = hilbert_matrix(n)
        x_true = np.ones(n, dtype=float)
        b = h @ x_true

        x1 = gauss_no_pivot(h, b)
        x2 = gauss_partial_pivot(h, b)

        rows.append({
            'n': n,
            'relative_error_no_pivot': f'{relative_error(x_true, x1):.6e}',
            'relative_error_partial_pivot': f'{relative_error(x_true, x2):.6e}',
            'residual_no_pivot': f'{residual_norm(h, x1, b):.6e}',
            'residual_partial_pivot': f'{residual_norm(h, x2, b):.6e}',
        })

    write_csv(out_dir / 'results_hilbert.csv', rows, list(rows[0].keys()))
    return rows


# ==================
# Построение графиков
# ==================

def plot_one_system(rows: list[dict], out_dir: Path) -> None:
    """Строит график сравнения времени решения одной системы."""
    x = [r['n'] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, [r['gauss_no_pivot_sec'] for r in rows], marker='o', label='Gauss без выбора')
    plt.plot(x, [r['gauss_partial_pivot_sec'] for r in rows], marker='o', label='Gauss с выбором')
    plt.plot(x, [r['lu_total_sec'] for r in rows], marker='o', label='LU общее')

    plt.xlabel('Размер n')
    plt.ylabel('Время, с')
    plt.title('Сравнение времени решения одной системы')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'plot_one_system.png', dpi=150)
    plt.close()



def plot_many_rhs(rows: list[dict], out_dir: Path) -> None:
    """Строит график зависимости времени от числа правых частей k."""
    x = [r['k'] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, [r['gauss_partial_pivot_total_sec'] for r in rows], marker='o', label='Gauss с выбором')
    plt.plot(x, [r['lu_total_sec'] for r in rows], marker='o', label='LU')

    plt.xlabel('Количество правых частей k')
    plt.ylabel('Время, с')
    plt.title('Несколько правых частей')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'plot_many_rhs.png', dpi=150)
    plt.close()



def plot_hilbert(rows: list[dict], out_dir: Path) -> None:
    """Строит график относительной погрешности для матрицы Гильберта."""
    x = [r['n'] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, [float(r['relative_error_no_pivot']) for r in rows], marker='o', label='Без выбора')
    plt.plot(x, [float(r['relative_error_partial_pivot']) for r in rows], marker='o', label='С выбором')

    # Логарифмический масштаб удобен, потому что ошибки могут отличаться на порядки.
    plt.yscale('log')
    plt.xlabel('Размер n')
    plt.ylabel('Относительная погрешность')
    plt.title('Матрица Гильберта')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'plot_hilbert.png', dpi=150)
    plt.close()


# ==================
# Главная функция
# ==================

def main() -> None:
    """
    Запускает все эксперименты, строит графики и сохраняет результаты
    в папку, где лежит сам скрипт.
    """
    out_dir = Path(__file__).resolve().parent

    # Эксперимент 1: сравнение одной системы.
    one = experiment_one_system(out_dir)

    # Эксперимент 2: выгода LU при нескольких правых частях.
    many = experiment_many_rhs(out_dir)

    # Эксперимент 3: проверка точности на матрице Гильберта.
    hilb = experiment_hilbert(out_dir)

    # Построение графиков по полученным данным.
    plot_one_system(one, out_dir)
    plot_many_rhs(many, out_dir)
    plot_hilbert(hilb, out_dir)

    print('Готово. Результаты сохранены в папке проекта.')

if __name__ == '__main__':
    main()
