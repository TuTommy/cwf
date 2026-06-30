"""
低阶常微分方程 Clairaut 方程的奇解与数值实验。

方程：
    y = x y' + sqrt(1 + (y')^2)

运行：
    python ode_clairaut_experiment.py

生成：
    results/solutions.png
    results/taylor_compare.png
    results/error_compare.png
    results/ode_residual.png
    results/error_table.csv
"""

import csv
import math
import time
import warnings
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from tyjuliacall import JuliaEvaluator

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font.")


def symbolic_taylor_verify() -> str:
    """用 TySymbolicMath 验证 sqrt(1-x^2) 在 x=0 处的泰勒展开。"""
    JuliaEvaluator["using TySymbolicMath"]
    JuliaEvaluator["@variables x"]
    result = JuliaEvaluator["taylor_series(sqrt(1 - x^2), x, Order=13)"]
    JuliaEvaluator['println("TySymbolicMath Taylor 验证通过")']
    return str(result)


def _binom_half(k: int) -> float:
    """广义二项式系数 C(1/2, k)。"""
    if k == 0:
        return 1.0
    return np.prod([0.5 - j for j in range(k)]) / math.factorial(k)


def taylor_coeffs_numpy(max_degree: int = 12) -> np.ndarray:
    """用 numpy 计算 sqrt(1-x^2) 的泰勒系数。"""
    coeffs = np.zeros(max_degree + 1)
    for k in range(max_degree // 2 + 1):
        coeffs[2 * k] = (-1) ** k * _binom_half(k)
    return coeffs


def exact_singular(x: np.ndarray) -> np.ndarray:
    """奇解 y = sqrt(1 - x^2)。"""
    return np.sqrt(np.maximum(0.0, 1.0 - x * x))


def line_solution(c: float, x: np.ndarray) -> np.ndarray:
    """Clairaut 方程的通解直线 y = Cx + sqrt(1+C^2)。"""
    return c * x + np.sqrt(1.0 + c * c)


def ode_residual(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """计算 Clairaut 方程残差 y - x*y' - sqrt(1+(y')^2)。"""
    return y - x * yp - np.sqrt(1.0 + yp * yp)


def make_solution_plot(xs: np.ndarray) -> None:
    """绘制通解直线族与奇解包络线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3b82f6", "#16a34a", "#f97316", "#a855f7", "#ef4444", "#0891b2"]
    for i, c in enumerate([-3, -1, -0.3, 0.3, 1, 3]):
        ax.plot(xs, line_solution(c, xs), color=colors[i], lw=1.2, label=f"C={c:g} 通解")
    ax.plot(xs, exact_singular(xs), "#111827", lw=2.6, label="奇解")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 4.5)
    ax.set_title("通解直线族与奇解包络线", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "solutions.png", dpi=150)
    plt.close(fig)


def make_taylor_plots(xs: np.ndarray, coeffs: np.ndarray) -> list[dict]:
    """绘制泰勒近似与误差图，并返回误差统计。"""
    degrees = [0, 2, 4, 6, 8, 10, 12]
    colors = ["#737373", "#3b82f6", "#16a34a", "#f97316", "#a855f7", "#ef4444", "#0891b2"]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(xs, exact_singular(xs), "#111827", lw=2.5, label="精确奇解")
    for degree, color in zip(degrees, colors):
        y_taylor = np.polynomial.polynomial.polyval(xs, coeffs[: degree + 1])
        ax1.plot(xs, y_taylor, color=color, lw=1.2, label=f"{degree} 阶泰勒近似")
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title("泰勒多项式近似与精确奇解对比", fontsize=14)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="lower center", ncol=4, fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(RESULT_DIR / "taylor_compare.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    rows = []
    y_exact = exact_singular(xs)
    for degree, color in zip(degrees, colors):
        y_taylor = np.polynomial.polynomial.polyval(xs, coeffs[: degree + 1])
        errors = np.abs(y_taylor - y_exact)
        errors_plot = np.maximum(errors, 1e-16)
        ax2.semilogy(xs, errors_plot, color=color, lw=1.2, label=f"{degree} 阶误差")
        rows.append(
            {
                "泰勒阶数": degree,
                "最大绝对误差": float(np.max(errors)),
                "平均绝对误差": float(np.mean(errors)),
                "均方误差": float(np.mean(errors**2)),
                "x=0.5处误差": float(
                    np.abs(np.polynomial.polynomial.polyval(0.5, coeffs[: degree + 1]) - np.sqrt(0.75))
                ),
                "x=0.9处误差": float(
                    np.abs(np.polynomial.polynomial.polyval(0.9, coeffs[: degree + 1]) - np.sqrt(0.19))
                ),
                "x=1.0处误差": float(np.abs(np.polynomial.polynomial.polyval(1.0, coeffs[: degree + 1]))),
            }
        )
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_title("各阶泰勒近似的绝对误差", fontsize=14)
    ax2.set_xlabel("x")
    ax2.set_ylabel("绝对误差")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(RESULT_DIR / "error_compare.png", dpi=150)
    plt.close(fig2)
    return rows


def make_residual_plot() -> None:
    """绘制原方程残差验证图。"""
    x_fine = np.linspace(-0.99, 0.99, 1001)
    fig, ax = plt.subplots(figsize=(10, 6))

    y_s = exact_singular(x_fine)
    yp_s = -x_fine / np.sqrt(np.maximum(1e-12, 1.0 - x_fine * x_fine))
    res_s = ode_residual(x_fine, y_s, yp_s)
    ax.semilogy(x_fine, np.abs(res_s), "#111827", lw=2.0, label="奇解残差")

    for c in [-1, 1, 3]:
        y_l = line_solution(c, x_fine)
        yp_l = np.full_like(x_fine, c)
        res_l = ode_residual(x_fine, y_l, yp_l)
        ax.semilogy(x_fine, np.abs(res_l), lw=1.0, alpha=0.6, label=f"C={c} 通解残差")

    ax.set_title("原方程残差验证 |y - xy' - sqrt(1+(y')^2)|", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("残差绝对值")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ode_residual.png", dpi=150)
    plt.close(fig)


def save_error_table(rows: list[dict], total_elapsed: float) -> None:
    """保存误差统计 CSV。"""
    for row in rows:
        row["程序总耗时(s)"] = round(total_elapsed, 6)
    with (RESULT_DIR / "error_table.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "泰勒阶数",
                "最大绝对误差",
                "平均绝对误差",
                "均方误差",
                "x=0.5处误差",
                "x=0.9处误差",
                "x=1.0处误差",
                "程序总耗时(s)",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def ode_numeric_verify() -> None:
    """用 scipy solve_ivp 数值求解奇解路径，与解析解比对。"""

    def singular_ode(x: float, y: np.ndarray) -> np.ndarray:
        return np.array([-x / np.sqrt(max(1e-12, 1.0 - x * x))])

    sol = solve_ivp(singular_ode, [0.0, 0.95], [1.0], dense_output=True, rtol=1e-10, atol=1e-12)
    x_test = np.linspace(0.0, 0.95, 200)
    y_numeric = sol.sol(x_test)[0]
    y_exact = np.sqrt(1.0 - x_test * x_test)
    max_dev = np.max(np.abs(y_numeric - y_exact))
    print(f"solve_ivp 数值解与解析奇解的最大偏差：{max_dev:.2e}")


def main() -> None:
    start = time.perf_counter()
    RESULT_DIR.mkdir(exist_ok=True)

    print("=== TySymbolicMath 符号泰勒验证 ===")
    jl_expr = symbolic_taylor_verify()
    print("TySymbolicMath: sqrt(1-x^2) 在 x=0 处展开到 12 阶：")
    print(f"  {jl_expr}")
    print()

    print("=== scipy solve_ivp 数值 ODE 验证 ===")
    ode_numeric_verify()
    print()

    xs = np.linspace(-1.0, 1.0, 2001)
    coeffs = taylor_coeffs_numpy(12)

    print("=== 生成图像 ===")
    make_solution_plot(xs)
    rows = make_taylor_plots(xs, coeffs)
    make_residual_plot()

    elapsed = time.perf_counter() - start
    save_error_table(rows, elapsed)

    print("\nsqrt(1-x^2) 在 n=0..12 时的泰勒系数 a_n：")
    for n, a in enumerate(coeffs):
        print(f"a_{n:02d} = {a:.12g}")

    print("\n误差统计表：")
    for row in rows:
        print(
            f"{int(row['泰勒阶数']):02d} 阶: 最大误差={row['最大绝对误差']:.6e}, "
            f"平均误差={row['平均绝对误差']:.6e}, "
            f"均方误差={row['均方误差']:.6e}, "
            f"x=0.5 误差={row['x=0.5处误差']:.6e}, "
            f"x=0.9 误差={row['x=0.9处误差']:.6e}, "
            f"x=1.0 误差={row['x=1.0处误差']:.6e}, "
            f"程序总耗时={elapsed:.6f} s"
        )

    print(f"\n总耗时：{elapsed:.6f} s")
    print(f"结果已保存到：{RESULT_DIR}")


if __name__ == "__main__":
    main()
