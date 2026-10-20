# =============================================================================
# COMP 3105A Assignment 2 — 로컬 검증 스크립트 (제출하지 않음)
# =============================================================================
#
# 이 스크립트는 정답 코드를 담고 있지 않다. 대신 다음 세 종류의 성질로
# 네 A2codes.py 를 검사한다.
#
#   1. 명세 준수   : 함수 이름/인자/기본값, 반환 shape·타입, sign(0) → +1,
#                    top-level 코드 없음, 출력 없음, 시드, 금지 라이브러리
#   2. 최적성      : 명세 PDF 에 적힌 목적함수를 그대로 계산해서,
#                    네가 반환한 해 주변으로 조금씩 움직여도 목적값이
#                    더 작아지지 않는지 (= 실제로 최솟값인지) 확인
#   3. 형태 간 대조 : 선형 커널에서 Primal(Q1) / Adjoint(Q2) / Dual(Q3) 가
#                    같은 분류기를 내는지 (명세의 Debug hint 들)
#
# 사용법 (A2files 폴더에서, venv 활성화 후):
#
#   python A2check.py              # 정적 검사 + Q1~Q3 함수 검사 (수십 초)
#   python A2check.py q1           # Q1 함수만        (q2, q3, static 도 가능)
#   python A2check.py q1 q2        # 여러 개 조합
#   python A2check.py exp          # 실험 함수까지 (synExperiments*, cvMnist)
#                                  #   → 수 분 걸림. 표도 같이 출력해 준다
#   python A2check.py all          # 전부
#
# 아직 구현 안 한 함수는 SKIP 으로 넘어가니, 한 단계 끝낼 때마다 돌리면 된다.
#
# 결과 표시:
#   PASS  통과
#   FAIL  거의 확실히 틀림 — 메시지의 힌트를 보고 고칠 것
#   WARN  틀렸을 가능성이 있거나 감점 요인 — 확인해볼 것
#   SKIP  아직 구현 안 됨 (또는 앞 단계가 필요)
#   INFO  참고용 숫자
# =============================================================================

import ast
import contextlib
import inspect
import io
import os
import subprocess
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# 설정 — 여기만 고치면 된다
# ---------------------------------------------------------------------------
STUDENT_FOLDER = "JaeyoonLee"   # A2codes.py 가 들어 있는 폴더 이름
GROUP_ID = 105                 # Brightspace 그룹 번호 (A2-XX 의 XX). 예: 17
                                # 정하면 시드가 이 값인지 검사한다

# Windows 에서 출력을 파일로 돌리거나 cp949 콘솔일 때 ⪯, ‖ 같은 기호로 죽지 않게
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(HERE, STUDENT_FOLDER, "A2codes.py")
sys.path.insert(0, HERE)

from A2helpers import generateData, linearKernel, polyKernel, gaussKernel  # noqa: E402


# ---------------------------------------------------------------------------
# 결과 기록
# ---------------------------------------------------------------------------
class Report:
    COLORS = {"PASS": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m",
              "SKIP": "\033[90m", "INFO": "\033[36m"}
    RESET = "\033[0m"

    def __init__(self):
        self.counts = {k: 0 for k in self.COLORS}
        # Windows 기본 콘솔에서 색 코드가 깨지면 끈다
        self.use_color = sys.stdout.isatty() and (os.name != "nt" or "WT_SESSION" in os.environ
                                                  or "TERM" in os.environ)

    def _p(self, tag, name, msg=""):
        self.counts[tag] += 1
        t = f"{self.COLORS[tag]}{tag}{self.RESET}" if self.use_color else tag
        line = f"  [{t}] {name}"
        if msg:
            line += f"\n         {msg}".replace("\n", "\n         ") if "\n" in msg else f" — {msg}"
        print(line)

    def ok(self, n, m=""):   self._p("PASS", n, m)
    def fail(self, n, m=""): self._p("FAIL", n, m)
    def warn(self, n, m=""): self._p("WARN", n, m)
    def skip(self, n, m=""): self._p("SKIP", n, m)
    def info(self, n, m=""): self._p("INFO", n, m)

    def check(self, cond, name, fail_msg, ok_msg="", level="FAIL"):
        if cond:
            self.ok(name, ok_msg)
        elif level == "WARN":
            self.warn(name, fail_msg)
        else:
            self.fail(name, fail_msg)
        return bool(cond)

    def section(self, title):
        print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)

    def summary(self):
        c = self.counts
        print("\n" + "-" * 72)
        print(f"PASS {c['PASS']}   FAIL {c['FAIL']}   WARN {c['WARN']}   SKIP {c['SKIP']}")
        if c["SKIP"]:
            print("※ SKIP 은 '통과' 가 아니라 '아직 검사 못 함' 이다 (함수가 없거나 None 을 반환).")
        print("-" * 72)


R = Report()


class NotImplementedYet(Exception):
    pass


def load_module():
    """A2codes 를 import. 실패하면 이유를 보여주고 None."""
    try:
        import importlib
        return importlib.import_module(f"{STUDENT_FOLDER}.A2codes")
    except Exception:
        R.fail("A2codes import", "import 자체가 실패했다:\n" + traceback.format_exc(limit=3))
        return None


def get_fn(mod, name):
    return getattr(mod, name, None) if mod is not None else None


def call(fn, *args, **kwargs):
    """함수를 호출하고 (결과, 표준출력에 찍힌 내용, 걸린 시간) 을 돌려준다.
    아직 구현 안 된 것처럼 보이면 NotImplementedYet."""
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        try:
            out = fn(*args, **kwargs)
        except NotImplementedError:
            raise NotImplementedYet()
    dt = time.time() - t0
    if out is None:
        raise NotImplementedYet()
    if isinstance(out, tuple) and any(o is None for o in out):
        raise NotImplementedYet()
    return out, buf.getvalue(), dt


def is_scalar(v):
    """w0 / a0 / b 는 스칼라여야 한다 (float, np.float64, 0-차원 배열 OK)."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    return isinstance(v, np.ndarray) and v.ndim == 0


def check_printed(name, printed):
    R.check(printed.strip() == "", f"{name}: 표준출력 없음",
            "함수가 무언가를 출력했다 (명세: shouldn't print). 첫 줄: "
            + (printed.strip().splitlines()[0][:80] if printed.strip() else ""),
            level="WARN")


def check_pm1(name, yhat, m):
    good = True
    good &= R.check(isinstance(yhat, np.ndarray) and yhat.shape == (m, 1),
                    f"{name}: 반환 shape ({m},1)",
                    f"shape 이 {getattr(yhat, 'shape', type(yhat))} 이다. (m,1) 열벡터여야 한다.")
    if isinstance(yhat, np.ndarray):
        vals = set(np.unique(yhat).tolist())
        good &= R.check(vals <= {-1, 1}, f"{name}: 값이 ±1 뿐",
                        f"예측값 집합이 {sorted(vals)} 이다. "
                        + ("0 이 섞여 있다 → np.sign(0)=0 처리 누락." if 0 in vals else ""))
    return good


# ---------------------------------------------------------------------------
# 명세 PDF 에 적힌 목적함수들 — 해의 최적성 검사에 쓴다
# (구현이 아니라 '정의' 다. 네 코드가 이걸 최소화했는지를 확인할 뿐이다.)
# ---------------------------------------------------------------------------
def obj_sq_hinge(X, y, w, w0, lamb):
    m = y * (X @ w + w0)
    return float(np.sum(np.maximum(0, 1 - m) ** 2) + lamb / 2 * np.sum(w ** 2))


def obj_hinge(X, y, w, w0, lamb):
    m = y * (X @ w + w0)
    return float(np.sum(np.maximum(0, 1 - m)) + lamb / 2 * np.sum(w ** 2))


def obj_adj_sq_hinge(K, y, a, a0, lamb):
    m = y * (K @ a + a0)
    return float(np.sum(np.maximum(0, 1 - m) ** 2) + lamb / 2 * (a.T @ K @ a).item())


def obj_adj_hinge(K, y, a, a0, lamb):
    m = y * (K @ a + a0)
    return float(np.sum(np.maximum(0, 1 - m)) + lamb / 2 * (a.T @ K @ a).item())


def obj_dual(K, y, a, lamb):
    """명세 Eq.(1) 의 (최대화) 목적값."""
    v = y * a
    return float(np.sum(a) - (v.T @ K @ v).item() / (2 * lamb))


def numeric_grad(f, u, eps=1e-6):
    g = np.zeros_like(u)
    for i in range(u.size):
        e = np.zeros_like(u)
        e[i] = eps
        g[i] = (f(u + e) - f(u - e)) / (2 * eps)
    return g


def perturbation_test(f, u, n_trials=300, seed=0):
    """u 주변 랜덤 방향으로 움직였을 때 f 가 '의미 있게' 작아지는 방향이 있는지.
    반환: (가장 많이 줄어든 양, 기준 목적값)"""
    rng = np.random.RandomState(seed)
    f0 = f(u)
    worst = 0.0
    for t in range(n_trials):
        step = [1e-4, 1e-3, 1e-2][t % 3] * max(1.0, np.abs(u).max())
        d = rng.randn(u.size)
        d /= np.linalg.norm(d)
        worst = min(worst, f(u + step * d) - f0)
    return worst, f0


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(a)), np.max(np.abs(b))))


def data(gen_model, n=60, seed=1):
    """고정된 시드로 작은 데이터셋 생성 (매번 같은 데이터로 검사)."""
    state = np.random.get_state()
    X, y = generateData(n=n, gen_model=gen_model, random_seed=seed)
    np.random.set_state(state)
    return X, y


POLY3 = lambda X1, X2: polyKernel(X1, X2, 3)   # noqa: E731
GAUSS1 = lambda X1, X2: gaussKernel(X1, X2, 1.0)  # noqa: E731

REQUIRED = {
    # 이름: (인자 이름 목록, 기본값 dict)
    "minSqHinge": (["X", "y", "lamb"], {}),
    "minHinge": (["X", "y", "lamb", "stabilizer"], {"stabilizer": 1e-5}),
    "classify": (["Xtest", "w", "w0"], {}),
    "synExperimentsRegularize": ([], {}),
    "adjSqHinge": (["X", "y", "lamb", "kernel_func"], {}),
    "adjHinge": (["X", "y", "lamb", "kernel_func", "stabilizer"], {"stabilizer": 1e-5}),
    "adjClassify": (["Xtest", "a", "a0", "X", "kernel_func"], {}),
    "synExperimentsKernel": ([], {}),
    "dualHinge": (["X", "y", "lamb", "kernel_func", "stabilizer"], {"stabilizer": 1e-5}),
    "dualClassify": (["Xtest", "a", "b", "X", "y", "lamb", "kernel_func"], {}),
    "cvMnist": (["dataset_folder", "lamb_list", "kernel_list", "k"], {"k": 5}),
}
SEED_FUNCS = ["synExperimentsRegularize", "synExperimentsKernel", "cvMnist"]
ALLOWED_IMPORTS = {"numpy", "scipy", "cvxopt", "pandas", "matplotlib", "A2helpers",
                   "os", "math", "time"}


# =============================================================================
# 0. 정적 검사 — 파일 구조
# =============================================================================
def static_checks(mod):
    R.section("STATIC — 파일 구조 / 명세 준수")

    if not os.path.exists(CODE_PATH):
        R.fail("A2codes.py 위치", f"{CODE_PATH} 가 없다. STUDENT_FOLDER 설정을 확인.")
        return
    src = open(CODE_PATH, encoding="utf-8").read()
    tree = ast.parse(src)

    # --- import 시 아무것도 실행/출력하지 않는지 (채점 환경과 같은 방식) -----
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", f"import {STUDENT_FOLDER}.A2codes"],
                       cwd=HERE, capture_output=True, text=True)
    dt = time.time() - t0
    R.check(p.returncode == 0, "새 프로세스에서 import 성공",
            "import 에러:\n" + p.stderr.strip()[-500:])
    R.check(p.stdout.strip() == "", "import 시 출력 없음",
            "import 만 했는데 출력이 나온다 → top-level 코드가 실행되고 있다.")
    R.check(dt < 5, "import 가 즉시 끝남", f"{dt:.1f}초 걸렸다 → top-level 에서 계산이 돌고 있다.",
            ok_msg=f"{dt:.2f}s", level="WARN")

    # --- top-level 문장 종류 --------------------------------------------------
    bad = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue  # 상수
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue  # docstring
        bad.append(f"line {node.lineno}: {type(node).__name__}")
    R.check(not bad, "top-level 에는 import / 상수 / def 만",
            "허용되지 않는 top-level 문장: " + ", ".join(bad)
            + "\n(if __name__ 블록, 함수 호출, print 등은 제거)")

    # --- import 목록 ----------------------------------------------------------
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    banned = {m for m in mods if m in {"sklearn", "torch", "tensorflow", "keras", "statsmodels"}}
    R.check(not banned, "ML 라이브러리 import 없음", f"금지: {sorted(banned)}")
    other = mods - ALLOWED_IMPORTS - banned
    R.check(not other, "알 수 없는 모듈 import 없음",
            f"{sorted(other)} — 직접 만든 모듈이면 채점 환경에 없어서 import 에러가 난다 "
            "(A1 StudentID 사건). 표준 라이브러리면 무시해도 된다.", level="WARN")

    # --- 함수 시그니처 ---------------------------------------------------------
    for name, (params, defaults) in REQUIRED.items():
        fn = get_fn(mod, name)
        if fn is None:
            R.skip(f"시그니처 {name}", "아직 없음")
            continue
        sig = inspect.signature(fn)
        got = list(sig.parameters)
        ok = got == params
        for k, v in defaults.items():
            if k in sig.parameters and sig.parameters[k].default != v:
                ok = False
        R.check(ok, f"시그니처 {name}",
                f"명세: ({', '.join(params)})  기본값 {defaults}\n"
                f"현재: {sig}")

    # --- 함수 안의 print ------------------------------------------------------
    prints = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    R.check(not prints, "print 호출 없음", f"line {prints} 에 print 가 있다.", level="WARN")

    # --- cvxopt 출력 끄기 -----------------------------------------------------
    if "solvers.qp" in src or "qp(" in src:
        R.check("show_progress" in src, "cvxopt show_progress 설정",
                "solvers.options['show_progress'] = False 가 안 보인다.", level="WARN")

    # --- 시드 -----------------------------------------------------------------
    fdefs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fname in SEED_FUNCS:
        if fname not in fdefs:
            R.skip(f"시드 {fname}", "아직 없음")
            continue
        seeds = []
        for n in ast.walk(fdefs[fname]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "seed":
                seeds.append(ast.unparse(n.args[0]) if n.args else "?")
        if not seeds:
            R.fail(f"시드 {fname}", "np.random.seed(...) 호출이 없다.")
            continue
        s = seeds[0]
        if s in ("0", "42"):
            R.fail(f"시드 {fname}", f"아직 템플릿 값 seed({s}) 이다 → 그룹 ID 로 바꿀 것.")
        elif s == "101272513":
            R.warn(f"시드 {fname}", "학번이 들어가 있다. A2 명세는 그룹 ID (A2-XX 의 XX) 를 요구한다.")
        elif GROUP_ID is not None and s != str(GROUP_ID):
            # 상수 이름을 썼을 수도 있으니 모듈에서 값을 찾아본다
            val = getattr(mod, s, None) if mod is not None else None
            R.check(val == GROUP_ID, f"시드 {fname}", f"seed({s}) 인데 GROUP_ID={GROUP_ID} 와 다르다.")
        else:
            R.ok(f"시드 {fname}", f"seed({s})" + ("" if GROUP_ID else "  ← GROUP_ID 를 설정하면 값까지 확인"))


# =============================================================================
# Q1 — Primal
# =============================================================================
def q1_checks(mod):
    R.section("Q1 — Primal form")
    res = {}

    # ---------------- Q1(c) classify ----------------
    fn = get_fn(mod, "classify")
    if fn is None:
        R.skip("classify", "아직 없음")
    else:
        try:
            w = np.array([[1.0], [1.0]])
            Xt = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [2.0, -3.0]])
            yhat, pr, _ = call(fn, Xt, w, 0.0)
            if check_pm1("classify", yhat, 4):
                R.check(np.array_equal(yhat.ravel(), [1, 1, -1, -1]), "classify: 값 정확 (0 → +1 포함)",
                        f"기대 [+1,+1,-1,-1], 결과 {yhat.ravel()}. 첫 점은 예측값이 정확히 0 → +1 이어야 한다.")
            y1, _, _ = call(fn, Xt[:1], w, 0.5)
            R.check(isinstance(y1, np.ndarray) and y1.shape == (1, 1), "classify: m=1 처리", f"shape {getattr(y1,'shape',None)}")
            y3, _, _ = call(fn, np.random.randn(7, 5), np.random.randn(5, 1), -0.3)
            R.check(isinstance(y3, np.ndarray) and y3.shape == (7, 1), "classify: d=5 처리", f"shape {getattr(y3,'shape',None)}")
            check_printed("classify", pr)
            res["classify"] = fn
        except NotImplementedYet:
            R.skip("classify", "미구현")
        except Exception:
            R.fail("classify", "에러:\n" + traceback.format_exc(limit=2))

    # ---------------- Q1(a) minSqHinge ----------------
    fn = get_fn(mod, "minSqHinge")
    if fn is None:
        R.skip("minSqHinge", "아직 없음")
    else:
        try:
            ok_all = True
            for gm, lamb in [(1, 1.0), (2, 0.1), (3, 0.01)]:
                X, y = data(gm)
                (w, w0), pr, dt = call(fn, X, y, lamb)
                tag = f"minSqHinge gen{gm} λ={lamb}"
                if not R.check(isinstance(w, np.ndarray) and w.shape == (2, 1), f"{tag}: w shape (d,1)",
                               f"w shape {getattr(w,'shape',type(w))}"):
                    ok_all = False
                    continue
                R.check(is_scalar(w0), f"{tag}: w0 스칼라",
                        f"w0 가 {type(w0).__name__} {getattr(w0,'shape','')} — float(...) 로 반환할 것.")
                w0 = float(np.asarray(w0).ravel()[0])
                # 최적성: 명세 목적함수의 수치 그래디언트가 0 에 가까워야 한다
                f = lambda u: obj_sq_hinge(X, y, u[:2, None], u[2], lamb)  # noqa: E731
                u = np.concatenate([w.ravel(), [w0]])
                g = numeric_grad(f, u)
                gn = np.abs(g).max()
                scale = max(1.0, f(u))
                ok_all &= R.check(gn < 1e-3 * scale, f"{tag}: 최적해 (∇목적함수 ≈ 0)",
                                  f"명세 목적함수의 그래디언트 크기가 {gn:.2e} 이다 → 최솟값이 아니다.\n"
                                  "의심: (1) 그래디언트 식 오류  (2) w0 를 정규화함  (3) Σ 대신 평균(1/n)\n"
                                  "      (4) λ/2 vs λ 계수  (5) res['x'] 를 쪼개는 순서",
                                  ok_msg=f"|∇|∞={gn:.1e}, {dt*1000:.0f}ms")
                check_printed(tag, pr)
            # 여러 차원
            Xd = np.random.RandomState(5).randn(40, 6)
            yd = np.where(Xd[:, :1] - Xd[:, 1:2] + 0.3 * np.random.RandomState(6).randn(40, 1) >= 0, 1.0, -1.0)
            (w, w0), _, dt = call(fn, Xd, yd, 0.5)
            R.check(w.shape == (6, 1), "minSqHinge: d=6 처리", f"w shape {w.shape}")
            R.check(dt < 0.5, "minSqHinge: 속도", f"n=40 한 번에 {dt:.2f}s — jac 를 안 넘겼거나 루프가 있다. "
                    "synExperimentsRegularize 가 느려진다.", ok_msg=f"{dt*1000:.0f}ms", level="WARN")
            # λ 가 커질수록 ‖w‖ 는 줄어야 한다 (정규화의 기본 성질)
            X, y = data(3)
            norms = [np.linalg.norm(call(fn, X, y, l)[0][0]) for l in [0.01, 0.1, 1.0, 10.0]]
            R.check(all(norms[i] >= norms[i + 1] - 1e-4 for i in range(3)), "minSqHinge: λ↑ → ‖w‖↓",
                    f"‖w‖ = {np.round(norms, 4)} — 정규화 항 부호/계수 확인.")
            if ok_all:
                res["minSqHinge"] = fn
        except NotImplementedYet:
            R.skip("minSqHinge", "미구현")
        except Exception:
            R.fail("minSqHinge", "에러:\n" + traceback.format_exc(limit=3))

    # ---------------- Q1(b) minHinge ----------------
    fn = get_fn(mod, "minHinge")
    if fn is None:
        R.skip("minHinge", "아직 없음")
    else:
        try:
            ok_all = True
            for gm, lamb in [(1, 1.0), (2, 0.1), (3, 0.01)]:
                X, y = data(gm)   # gen3 의 y 는 정수(int) 배열 — cvxopt 에러가 나면 except 에서 힌트
                (w, w0), pr, dt = call(fn, X, y, lamb)
                tag = f"minHinge gen{gm} λ={lamb}"
                if not R.check(isinstance(w, np.ndarray) and w.shape == (2, 1), f"{tag}: w shape (d,1)",
                               f"w shape {getattr(w,'shape',type(w))}"):
                    ok_all = False
                    continue
                R.check(is_scalar(w0), f"{tag}: w0 스칼라",
                        f"w0 가 {type(w0).__name__} {getattr(w0,'shape','')} — float(...) 로 반환할 것.")
                w0 = float(np.asarray(w0).ravel()[0])
                f = lambda u: obj_hinge(X, y, u[:2, None], u[2], lamb)  # noqa: E731
                u = np.concatenate([w.ravel(), [w0]])
                drop, f0 = perturbation_test(f, u)
                ok_all &= R.check(drop > -1e-4 * max(1.0, f0), f"{tag}: 최적해 (주변에 더 작은 값 없음)",
                                  f"해를 살짝 움직였더니 목적값이 {-drop:.3e} 만큼 줄었다 → 최솟값이 아니다.\n"
                                  "의심: (1) u=[w;w0;ξ] 순서와 P/q/G 블록 순서 불일치  (2) P 의 λ 계수 (cvxopt 의 1/2!)\n"
                                  "      (3) 두 번째 제약의 부호  (4) sol['x'] 에서 w, w0 를 꺼내는 위치",
                                  ok_msg=f"obj={f0:.4f}, {dt*1000:.0f}ms")
                check_printed(tag, pr)
            Xd = np.random.RandomState(5).randn(40, 6)
            yd = np.where(Xd[:, :1] - Xd[:, 1:2] >= 0, 1.0, -1.0)
            (w, w0), _, _ = call(fn, Xd, yd, 0.5)
            R.check(w.shape == (6, 1), "minHinge: d=6 처리", f"w shape {w.shape}")
            if ok_all:
                res["minHinge"] = fn
        except NotImplementedYet:
            R.skip("minHinge", "미구현")
        except Exception:
            tb = traceback.format_exc(limit=3)
            hint = ""
            if "'d' matrix" in tb or "must be a 'd'" in tb:
                hint = "\n→ cvxopt 에 int 배열이 들어갔다. gen3 의 y 는 정수다: matrix(arr.astype(float))"
            R.fail("minHinge", "에러:\n" + tb + hint)

    return res


# =============================================================================
# Q2 — Adjoint
# =============================================================================
def q2_checks(mod, q1):
    R.section("Q2 — Adjoint form")
    res = {}

    # ---------------- Q2(c) adjClassify ----------------
    fn = get_fn(mod, "adjClassify")
    if fn is None:
        R.skip("adjClassify", "아직 없음")
    else:
        try:
            X, y = data(1, n=20)
            yhat, pr, _ = call(fn, X[:6], np.zeros((20, 1)), 0.0, X, linearKernel)
            if check_pm1("adjClassify", yhat, 6):
                R.check(np.all(yhat == 1), "adjClassify: 예측값 0 → +1",
                        f"a=0, a0=0 이면 모든 예측값이 0 → 전부 +1 이어야 하는데 {yhat.ravel()}")
            a = np.random.RandomState(2).randn(20, 1)
            y1, _, _ = call(fn, X[:1], a, 0.1, X, GAUSS1)
            R.check(y1.shape == (1, 1), "adjClassify: m=1 처리", f"shape {y1.shape}")
            # 직접 계산과 일치
            yh, _, _ = call(fn, X, a, 0.1, X, POLY3)
            z = POLY3(X, X) @ a + 0.1
            R.check(np.array_equal(yh.ravel(), np.where(z >= 0, 1, -1).ravel()),
                    "adjClassify: kernel_func(Xtest, X) @ a + a0 의 부호와 일치",
                    "커널 인자 순서 (Xtest, X) 또는 a 곱셈 확인.")
            check_printed("adjClassify", pr)
            res["adjClassify"] = fn
        except NotImplementedYet:
            R.skip("adjClassify", "미구현")
        except Exception:
            R.fail("adjClassify", "에러:\n" + traceback.format_exc(limit=2))

    # ---------------- Q2(a) adjSqHinge ----------------
    fn = get_fn(mod, "adjSqHinge")
    if fn is None:
        R.skip("adjSqHinge", "아직 없음")
    else:
        try:
            ok_all = True
            for gm, lamb, kname, kf in [(1, 1.0, "linear", linearKernel), (3, 0.1, "poly3", POLY3),
                                        (2, 0.1, "gauss1", GAUSS1)]:
                X, y = data(gm)
                (a, a0), pr, dt = call(fn, X, y, lamb, kf)
                tag = f"adjSqHinge gen{gm} {kname} λ={lamb}"
                if not R.check(isinstance(a, np.ndarray) and a.shape == (len(X), 1), f"{tag}: a shape (n,1)",
                               f"a shape {getattr(a,'shape',type(a))}"):
                    ok_all = False
                    continue
                R.check(is_scalar(a0), f"{tag}: a0 스칼라", f"a0 가 {type(a0).__name__} — float(...)")
                a0 = float(np.asarray(a0).ravel()[0])
                K = kf(X, X)
                # α 공간은 ill-conditioned 라 그래디언트 대신 '주변에 더 작은 값 없음' 으로 검사
                n = len(X)
                f = lambda u: obj_adj_sq_hinge(K, y, u[:n, None], u[n], lamb)  # noqa: E731
                drop, f0 = perturbation_test(f, np.concatenate([a.ravel(), [a0]]), n_trials=150)
                ok_all &= R.check(drop > -1e-3 * max(1.0, f0), f"{tag}: 최적해",
                                  f"주변에 목적값이 {-drop:.3e} 더 작은 점이 있다.\n"
                                  "의심: αᵀKα 의 그래디언트 계수, K 대신 X 사용, a0 처리",
                                  ok_msg=f"obj={f0:.4f}, {dt*1000:.0f}ms")
                check_printed(tag, pr)
                if kname == "linear" and "minSqHinge" in q1:
                    (w, w0), _, _ = call(q1["minSqHinge"], X, y, lamb)
                    w0 = float(np.asarray(w0).ravel()[0])
                    d = rel_diff(X @ w + w0, K @ a + a0)
                    R.check(d < 1e-2, f"{tag}: Q1(a) 와 마진 일치 (Xw+w0 ≈ Kα+α0)",
                            f"상대 차이 {d:.2e}. (α 자체가 아니라 Kα+α0 를 비교하는 것이다)",
                            ok_msg=f"상대 차이 {d:.1e}")
                    d2 = rel_diff(w, X.T @ a)
                    R.info(f"{tag}: w vs Xᵀα", f"상대 차이 {d2:.1e} (명세 debug hint: w* = Xᵀα*)")
            if ok_all:
                res["adjSqHinge"] = fn
        except NotImplementedYet:
            R.skip("adjSqHinge", "미구현")
        except Exception:
            R.fail("adjSqHinge", "에러:\n" + traceback.format_exc(limit=3))

    # ---------------- Q2(b) adjHinge ----------------
    fn = get_fn(mod, "adjHinge")
    if fn is None:
        R.skip("adjHinge", "아직 없음")
    else:
        try:
            ok_all = True
            for gm, lamb, kname, kf in [(1, 1.0, "linear", linearKernel), (3, 0.1, "linear", linearKernel),
                                        (3, 0.1, "poly3", POLY3), (2, 0.1, "gauss1", GAUSS1)]:
                X, y = data(gm)
                (a, a0), pr, dt = call(fn, X, y, lamb, kf)
                tag = f"adjHinge gen{gm} {kname} λ={lamb}"
                if not R.check(isinstance(a, np.ndarray) and a.shape == (len(X), 1), f"{tag}: a shape (n,1)",
                               f"a shape {getattr(a,'shape',type(a))}"):
                    ok_all = False
                    continue
                R.check(is_scalar(a0), f"{tag}: a0 스칼라", f"a0 가 {type(a0).__name__} — float(...)")
                a0 = float(np.asarray(a0).ravel()[0])
                K = kf(X, X)
                n = len(X)
                f = lambda u: obj_adj_hinge(K, y, u[:n, None], u[n], lamb)  # noqa: E731
                drop, f0 = perturbation_test(f, np.concatenate([a.ravel(), [a0]]), n_trials=150)
                ok_all &= R.check(drop > -1e-3 * max(1.0, f0), f"{tag}: 최적해",
                                  f"주변에 목적값이 {-drop:.3e} 더 작은 점이 있다.\n"
                                  "의심: u=[α;α0;ξ] 블록 순서, P 의 K 블록 계수, 제약의 Kα 부분",
                                  ok_msg=f"obj={f0:.4f}, {dt*1000:.0f}ms")
                check_printed(tag, pr)
                if kname == "linear" and "minHinge" in q1:
                    (w, w0), _, _ = call(q1["minHinge"], X, y, lamb)
                    w0 = float(np.asarray(w0).ravel()[0])
                    d = rel_diff(X @ w + w0, K @ a + a0)
                    R.check(d < 1e-2, f"{tag}: Q1(b) 와 마진 일치 (Xw+w0 ≈ Kα+α0)",
                            f"상대 차이 {d:.2e}", ok_msg=f"상대 차이 {d:.1e}")
            if ok_all:
                res["adjHinge"] = fn
        except NotImplementedYet:
            R.skip("adjHinge", "미구현")
        except Exception:
            R.fail("adjHinge", "에러:\n" + traceback.format_exc(limit=3))

    return res


# =============================================================================
# Q3 — Dual
# =============================================================================
def q3_checks(mod, q1, q2):
    R.section("Q3 — Dual form")
    res = {}

    # ---------------- Q3(a) dualHinge ----------------
    fn = get_fn(mod, "dualHinge")
    if fn is None:
        R.skip("dualHinge", "아직 없음")
    else:
        try:
            ok_all = True
            for gm, lamb, kname, kf in [(1, 1.0, "linear", linearKernel), (3, 0.1, "linear", linearKernel),
                                        (3, 0.1, "poly3", POLY3), (2, 0.1, "gauss1", GAUSS1)]:
                X, y = data(gm)
                yf = y.astype(float)
                (a, b), pr, dt = call(fn, X, y, lamb, kf)
                tag = f"dualHinge gen{gm} {kname} λ={lamb}"
                n = len(X)
                if not R.check(isinstance(a, np.ndarray) and a.shape == (n, 1), f"{tag}: a shape (n,1)",
                               f"a shape {getattr(a,'shape',type(a))}"):
                    ok_all = False
                    continue
                R.check(is_scalar(b), f"{tag}: b 스칼라", f"b 가 {type(b).__name__} {getattr(b,'shape','')} — float(...)")
                b = float(np.asarray(b).ravel()[0])
                K = kf(X, X)

                # (1) 제약 조건
                box = a.min() > -1e-6 and a.max() < 1 + 1e-6
                eq = abs((a.T @ yf).item()) < 1e-5
                ok_all &= R.check(box and eq, f"{tag}: 제약 0 ⪯ α ⪯ 1, αᵀy = 0",
                                  f"α 범위 [{a.min():.3g}, {a.max():.3g}], αᵀy = {(a.T @ yf).item():.3g}\n"
                                  "의심: G/h 블록, A/b (등식) 인자 누락, max→min 부호",
                                  ok_msg=f"{dt*1000:.0f}ms")

                # (2) 강쌍대성: dual 최댓값 = primal 최솟값
                #     primal 해는 dual 에서 복원한 결정함수로 계산
                f_dec = (K @ (yf * a)) / lamb + b
                primal_val = float(np.sum(np.maximum(0, 1 - yf * f_dec))
                                   + ((yf * a).T @ K @ (yf * a)).item() / (2 * lamb))
                dual_val = obj_dual(K, yf, a, lamb)
                gap = (primal_val - dual_val) / max(1.0, abs(primal_val))
                ok_all &= R.check(abs(gap) < 1e-2, f"{tag}: 쌍대 간극 ≈ 0 (primal = dual)",
                                  f"primal {primal_val:.4f} vs dual {dual_val:.4f} (상대 {gap:.2e}).\n"
                                  "α 가 최적이 아니거나 b 가 틀렸다. α 가 맞는데 이것만 실패하면 b 를 의심:\n"
                                  "0<αᵢ<1 인 점 선택 (0.5 에 가장 가까운 것), 1/λ 계수, Δ(y)α 계산",
                                  ok_msg=f"primal={primal_val:.4f}, dual={dual_val:.4f}")

                # (3) 마진 위의 support vector (0<αᵢ<1) 는 yᵢ f(xᵢ) ≈ 1
                free = (a.ravel() > 1e-3) & (a.ravel() < 1 - 1e-3)
                if free.any():
                    mfree = (yf * f_dec).ravel()[free]
                    R.check(np.max(np.abs(mfree - 1)) < 5e-2, f"{tag}: 0<αᵢ<1 인 점의 마진 ≈ 1",
                            f"마진 범위 [{mfree.min():.3f}, {mfree.max():.3f}] — b 계산 확인.",
                            ok_msg=f"{free.sum()}개 점", level="WARN")
                check_printed(tag, pr)

                # (4) 선형 커널: w = (1/λ) Xᵀ Δ(y) α  ≈  Q1(b) 의 w
                if kname == "linear" and "minHinge" in q1:
                    (w1, w01), _, _ = call(q1["minHinge"], X, y, lamb)
                    w01 = float(np.asarray(w01).ravel()[0])
                    wd = X.T @ (yf * a) / lamb
                    d = rel_diff(X @ w1 + w01, X @ wd + b)
                    R.check(d < 1e-2, f"{tag}: Q1(b) 와 일치 (w, b)",
                            f"Q1(b) w={w1.ravel().round(4)}, w0={w01:.4f}\n"
                            f"dual  w={wd.ravel().round(4)}, b ={b:.4f}  (마진 상대 차이 {d:.2e})",
                            ok_msg=f"w={wd.ravel().round(3)}, b={b:.3f}")
            if ok_all:
                res["dualHinge"] = fn
        except NotImplementedYet:
            R.skip("dualHinge", "미구현")
        except Exception:
            tb = traceback.format_exc(limit=3)
            hint = ""
            if "Rank(A)" in tb:
                hint = "\n→ A 의 shape 이 (1,n) 인지, stabilizer 를 더했는지 확인"
            elif "'d' matrix" in tb:
                hint = "\n→ cvxopt 에 int 배열 (y 로 만든 A 등) — astype(float)"
            R.fail("dualHinge", "에러:\n" + tb + hint)

    # ---------------- Q3(b) dualClassify ----------------
    fn = get_fn(mod, "dualClassify")
    if fn is None:
        R.skip("dualClassify", "아직 없음")
    else:
        try:
            X, y = data(3, n=30)
            yf = y.astype(float)
            yhat, pr, _ = call(fn, X[:5], np.zeros((30, 1)), 0.0, X, y, 0.1, linearKernel)
            if check_pm1("dualClassify", yhat, 5):
                R.check(np.all(yhat == 1), "dualClassify: 예측값 0 → +1",
                        f"a=0, b=0 이면 모두 +1 이어야 하는데 {yhat.ravel()}")
            a = np.random.RandomState(3).rand(30, 1)
            for lamb in [0.1, 2.0]:
                yh, _, _ = call(fn, X, a, 0.2, X, y, lamb, POLY3)
                z = POLY3(X, X) @ (yf * a) / lamb + 0.2
                R.check(np.array_equal(yh.ravel(), np.where(z >= 0, 1, -1).ravel()),
                        f"dualClassify: 명세 식과 일치 (λ={lamb})",
                        "sign((1/λ)·kernel_func(Xtest,X)·Δ(y)α + b) 와 다르다. 1/λ, Δ(y)α 확인.")
            check_printed("dualClassify", pr)
            res["dualClassify"] = fn

            # dual 과 adjoint 결정 경계 비교 (같은 λ, 커널 → 같은 hinge 분류기)
            if "dualHinge" in res and "adjHinge" in q2 and "adjClassify" in q2:
                X, y = data(3, n=80, seed=4)
                Xt, _ = data(3, n=500, seed=5)
                for kname, kf in [("poly3", POLY3), ("gauss1", GAUSS1)]:
                    a1, b1 = call(res["dualHinge"], X, y, 0.1, kf)[0]
                    a2, a02 = call(q2["adjHinge"], X, y, 0.1, kf)[0]
                    p1 = call(fn, Xt, a1, b1, X, y, 0.1, kf)[0]
                    p2 = call(q2["adjClassify"], Xt, a2, a02, X, kf)[0]
                    agree = np.mean(p1 == p2)
                    R.check(agree > 0.98, f"Dual vs Adjoint 예측 일치 ({kname}, 테스트 500점)",
                            f"일치율 {agree:.3f} — 같은 문제의 두 형태이므로 거의 같아야 한다.",
                            ok_msg=f"일치율 {agree:.3f}")
        except NotImplementedYet:
            R.skip("dualClassify", "미구현")
        except Exception:
            R.fail("dualClassify", "에러:\n" + traceback.format_exc(limit=2))

    return res


# =============================================================================
# 실험 함수 — 오래 걸린다 (python A2check.py exp)
# =============================================================================
def fmt_table(M, rows, title):
    head = f"{'':>14}" + "".join(f"{c:>9}" for c in ["SqH-M1", "SqH-M2", "SqH-M3", "H-M1", "H-M2", "H-M3"])
    lines = [title, head] + [f"{r:>14}" + "".join(f"{v:9.4f}" for v in row) for r, row in zip(rows, M)]
    return "\n".join(lines)


def exp_checks(mod):
    R.section("EXPERIMENTS — synExperimentsRegularize / synExperimentsKernel / cvMnist")
    reg = None

    fn = get_fn(mod, "synExperimentsRegularize")
    if fn is None:
        R.skip("synExperimentsRegularize", "아직 없음")
    else:
        try:
            print("  ... synExperimentsRegularize 실행 중 (명세 기준 ~20초)")
            (tr, te), pr, dt = call(fn)
            ok = R.check(np.shape(tr) == (4, 6) and np.shape(te) == (4, 6), "synExperimentsRegularize: (4,6) 두 개",
                         f"shape {np.shape(tr)}, {np.shape(te)}")
            if ok:
                tr, te = np.asarray(tr, float), np.asarray(te, float)
                R.check(np.all((tr >= 0) & (tr <= 1)) and np.all((te >= 0) & (te <= 1)),
                        "synExperimentsRegularize: 값이 [0,1]", "정확도가 아닌 값이 있다.")
                R.check(not np.allclose(tr, tr.flat[0]), "synExperimentsRegularize: 값이 전부 같지 않음",
                        "모든 셀이 같은 값 → 인덱스 [i,j,r] 를 잘못 채웠거나 평균 축 오류.")
                R.check(dt < 60, "synExperimentsRegularize: 실행 시간", f"{dt:.0f}s — 명세는 약 20초. 벡터화/jac 확인.",
                        ok_msg=f"{dt:.1f}s", level="WARN")
                check_printed("synExperimentsRegularize", pr)
                # 시드가 함수 안에서 설정되는지 → 두 번 돌려 같은지
                (tr2, _), _, _ = call(fn)
                R.check(np.allclose(tr, tr2), "synExperimentsRegularize: 재현성 (두 번 실행 결과 동일)",
                        "결과가 실행마다 다르다 → np.random.seed 가 함수 안 맨 앞에 있는지 확인.")
                lam = ["λ=0.001", "λ=0.01", "λ=0.1", "λ=1.0"]
                print("\n" + fmt_table(tr, lam, "  [Table 1] Training accuracy"))
                print(fmt_table(te, lam, "  [Table 2] Test accuracy") + "\n")
                reg = (tr, te)
        except NotImplementedYet:
            R.skip("synExperimentsRegularize", "미구현 (TODO 가 남아 있음)")
        except Exception:
            R.fail("synExperimentsRegularize", "에러:\n" + traceback.format_exc(limit=3))

    fn = get_fn(mod, "synExperimentsKernel")
    if fn is None:
        R.skip("synExperimentsKernel", "아직 없음")
    else:
        try:
            print("  ... synExperimentsKernel 실행 중 (명세 기준 ~5분)")
            (tr, te), pr, dt = call(fn)
            ok = R.check(np.shape(tr) == (5, 6) and np.shape(te) == (5, 6), "synExperimentsKernel: (5,6) 두 개",
                         f"shape {np.shape(tr)}, {np.shape(te)}")
            if ok:
                tr, te = np.asarray(tr, float), np.asarray(te, float)
                R.check(np.all((tr >= 0) & (tr <= 1)) and np.all((te >= 0) & (te <= 1)),
                        "synExperimentsKernel: 값이 [0,1]", "정확도가 아닌 값이 있다.")
                R.check(dt < 600, "synExperimentsKernel: 실행 시간", f"{dt:.0f}s — 명세는 약 5분.",
                        ok_msg=f"{dt:.0f}s", level="WARN")
                check_printed("synExperimentsKernel", pr)
                if reg is not None:
                    d = np.abs(te[0] - reg[1][0]).max()
                    R.check(d < 0.05, "Linear 커널 행 ≈ Q1 표의 λ=0.001 행 (테스트)",
                            f"최대 차이 {d:.3f}. 같은 문제를 다른 형태로 푼 것이라 비슷해야 한다 "
                            "(runs 수가 달라 완전히 같지는 않음).", ok_msg=f"최대 차이 {d:.3f}", level="WARN")
                ks = ["Linear", "Poly(d=2)", "Poly(d=3)", "Gauss(σ=1)", "Gauss(σ=0.5)"]
                print("\n" + fmt_table(tr, ks, "  [Table 3] Training accuracy"))
                print(fmt_table(te, ks, "  [Table 4] Test accuracy") + "\n")
        except NotImplementedYet:
            R.skip("synExperimentsKernel", "미구현 (TODO 가 남아 있음)")
        except Exception:
            R.fail("synExperimentsKernel", "에러:\n" + traceback.format_exc(limit=3))

    fn = get_fn(mod, "cvMnist")
    if fn is None:
        R.skip("cvMnist", "아직 없음")
    else:
        try:
            # 기능 검사용 임의 후보 (튜닝된 값 아님 — 네 후보는 직접 정할 것)
            lamb_list = [0.1, 1.0]
            kernel_list = [linearKernel,
                           lambda X1, X2: polyKernel(X1, X2, 2),
                           lambda X1, X2: gaussKernel(X1, X2, 1.0)]
            print("  ... cvMnist 실행 중 (2 λ × 3 커널 × 5 fold, 명세 기준 ~30초)")
            (cv, bl, bk), pr, dt = call(fn, HERE, lamb_list, kernel_list)
            ok = R.check(np.shape(cv) == (2, 3), "cvMnist: cv_acc shape (len(lamb_list), len(kernel_list))",
                         f"shape {np.shape(cv)} — k 축(axis=0) 평균을 했는지 확인.")
            if ok:
                cv = np.asarray(cv, float)
                R.check(np.all((cv >= 0) & (cv <= 1)), "cvMnist: 값이 [0,1]", f"{cv}")
                R.check(bl in lamb_list, "cvMnist: best_lamb 이 후보 중 하나", f"best_lamb = {bl}")
                R.check(any(bk is k for k in kernel_list), "cvMnist: best_kernel 이 후보 함수 객체 자체",
                        f"best_kernel = {bk!r} — 문자열/인덱스가 아니라 kernel_list 안의 함수를 그대로 반환.")
                i, j = np.unravel_index(np.argmax(cv), cv.shape)
                R.check(bl == lamb_list[i] and bk is kernel_list[j], "cvMnist: best 가 cv_acc 의 최댓값 위치",
                        f"cv_acc 최댓값은 (λ={lamb_list[i]}, 커널 #{j}) 인데 반환값이 다르다.")
                R.check(not np.allclose(cv, cv.flat[0]), "cvMnist: 조합마다 결과가 다름",
                        "모든 조합의 정확도가 같다 → 커널/λ 가 실제로 안 쓰이고 있거나 lambda late binding.",
                        level="WARN")
                R.check(dt < 120, "cvMnist: 실행 시간", f"{dt:.0f}s — 명세는 약 30초.",
                        ok_msg=f"{dt:.1f}s", level="WARN")
                check_printed("cvMnist", pr)
                R.info("cvMnist 결과 (기능 검사용 임의 후보)",
                       "행 λ=" + str(lamb_list) + ", 열 [linear, poly2, gauss(1.0)]\n" + np.array2string(cv, precision=4))
        except NotImplementedYet:
            R.skip("cvMnist", "미구현 (TODO 가 남아 있음)")
        except FileNotFoundError:
            R.fail("cvMnist", f"A2train.csv 를 못 찾았다. dataset_folder={HERE} 로 불렀다. 경로 조립 확인.")
        except Exception:
            R.fail("cvMnist", "에러:\n" + traceback.format_exc(limit=3))


# =============================================================================
def main():
    args = {a.lower() for a in sys.argv[1:]}
    if not args:
        args = {"static", "q1", "q2", "q3"}
    if "all" in args:
        args = {"static", "q1", "q2", "q3", "exp"}

    np.random.seed(12345)  # 검사 스크립트 자체의 난수 (네 시드와 무관)
    print(f"A2check — {CODE_PATH}")
    mod = load_module()
    if mod is None:
        R.summary()
        return

    if "static" in args:
        static_checks(mod)
    q1 = q2 = {}
    if args & {"q1", "q2", "q3"}:
        # Q2, Q3 의 대조 검사에는 Q1 결과가 필요하므로 Q1 은 조용히라도 돌린다
        q1 = q1_checks(mod) if "q1" in args else _silent(q1_checks, mod)
    if args & {"q2", "q3"}:
        q2 = q2_checks(mod, q1) if "q2" in args else _silent(q2_checks, mod, q1)
    if "q3" in args:
        q3_checks(mod, q1, q2)
    if "exp" in args:
        exp_checks(mod)
    R.summary()


def _silent(f, *a):
    """대조용으로만 실행 (결과는 출력/집계하지 않음)."""
    global R
    real = R
    R = Report()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return f(*a)
    finally:
        R = real


if __name__ == "__main__":
    main()
