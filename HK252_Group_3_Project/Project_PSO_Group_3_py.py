#!/usr/bin/env python3
"""
PSO runner for Cadence OCEAN using tb_opamp.ocn.

Mục tiêu:
- Python ghi input vào tb_core_param.txt
- gọi `ocean -nograph -restore tb_opamp.ocn`
- đọc output từ tb_core_results.txt
- dùng PSO để tối ưu bộ hệ số rời rạc

Phiên bản này đã chỉnh để bài toán analog dễ hội tụ hơn:
- fitness mềm, không fail cứng toàn bộ chỉ vì 1 metric lỗi
- bỏ phạt nặng với `iq_maxload` vì trong OCEAN hiện tại đó thực ra là dòng tải
- thêm velocity clamp
- thêm khởi tạo swarm quanh vài điểm guide tốt
- thêm reseed khi particle bị stagnant
- thêm timeout cho mỗi lần gọi OCEAN
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# =========================
# 1) CẤU HÌNH BIẾN THIẾT KẾ
# =========================
COEFF_ORDER: List[str] = [
    "ibn_coe",
    "ibp_coe",
    "rm_coe",
    "cm_coe",
    "fg_core_n1",
    "fg_core_n3",
    "fg_core_nd1",
    "fg_core_p3",
    "fg_core_mpp",
    "m_core_mpp",
    "stk_core_n1",
    "stk_core_n3",
    "stk_core_nd1",
    "stk_core_p3",
    "stk_core_mpp",
]

COEFF_BOUNDS: Dict[str, Tuple[int, int]] = {
    "ibn_coe": (2, 50),
    "ibp_coe": (2, 50),
    "rm_coe": (1, 20),
    "cm_coe": (1, 20),
    "fg_core_n1": (2, 100),
    "fg_core_n3": (2, 100),
    "fg_core_nd1": (2, 100),
    "fg_core_p3": (2, 100),
    "fg_core_mpp": (50, 200),
    "m_core_mpp": (10, 30),
    "stk_core_n1": (1, 10),
    "stk_core_n3": (1, 10),
    "stk_core_nd1": (1, 10),
    "stk_core_p3": (1, 10),
    "stk_core_mpp": (1, 10),
}

FIXED_PARAMS: Dict[str, float] = {
    "vddval": 1.2,
    "vrefval": 0.6,
    "offset": 0.12,
    "min_load": 8000,
    "max_load": 16,
}

PARAM_FILE_ORDER: List[str] = [
    "ibn",
    "ibp",
    "rm",
    "cm",
    "fg_core_n1",
    "fg_core_n3",
    "fg_core_nd1",
    "fg_core_p3",
    "fg_core_mpp",
    "m_core_mpp",
    "stk_core_n1",
    "stk_core_n3",
    "stk_core_nd1",
    "stk_core_p3",
    "stk_core_mpp",
    "vddval",
    "vrefval",
    "offset",
    "min_load",
    "max_load",
]

RESULT_ORDER: List[str] = [
    "dcop_check",
    "load_regulation",
    "line_regulation",
    "vout_err",
    "iq_maxload",
    "psr_1meg",
    "ugb_khz",
    "av_db",
    "pm_deg",
    "region_MND1",
    "region_MN1",
    "region_MN3",
    "region_MP3",
    "region_MPP",
]

# Các target này dùng cho fitness mềm.
# Chúng không còn là rào cứng; chỉ là mốc để chấm điểm.
SPEC_TARGETS = {
    "load_regulation_max": 40.0,
    "line_regulation_max": 8.0,
    "vout_err_max": 8.0,
    "psr_1meg_min": 30.0,
    "ugb_khz_min": 5000.0,
    "av_db_min": 50.0,
    "pm_deg_min": 60.0,
}

# Một vài điểm guide lấy từ chính history của bạn: có AC/regulation khá hơn mặt bằng.
# Swarm sẽ khởi tạo một phần quanh các điểm này để đỡ lang thang.
GUIDE_COEFFS: List[List[int]] = [
    [39, 21, 6, 14, 65, 64, 91, 20, 162, 27, 1, 1, 4, 3, 4],
    [13, 7, 6, 13, 38, 38, 23, 28, 190, 23, 6, 3, 8, 2, 4],
    [41, 37, 11, 19, 39, 56, 83, 63, 179, 22, 7, 1, 3, 4, 2],
]


# =========================
# 2) MAPPING HỆ SỐ -> GIÁ TRỊ THẬT CHO OCEAN
# =========================
def build_ibn_value(ibn_coe: int) -> float:
    return float(ibn_coe)


def build_ibp_value(ibp_coe: int) -> float:
    return float(ibp_coe)


def build_rm_value(rm_coe: int) -> float:
    c = float(rm_coe)
    return 500.0 * c + 50.0 * c**2 + 10.0 * c**3


def build_cm_value(cm_coe: int) -> float:
    return (1.0 + float(cm_coe)) ** 2


def coeff_vector_to_dict(vector: Sequence[int]) -> Dict[str, int]:
    if len(vector) != len(COEFF_ORDER):
        raise ValueError(f"Cần đúng {len(COEFF_ORDER)} hệ số, hiện tại nhận {len(vector)}")
    return {name: int(value) for name, value in zip(COEFF_ORDER, vector)}


def clamp_coeff_dict(coeffs: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in COEFF_ORDER:
        low, high = COEFF_BOUNDS[name]
        out[name] = max(low, min(high, int(round(coeffs[name]))))
    return out


def build_param_dict(coeffs: Dict[str, int], fixed_params: Dict[str, float] | None = None) -> Dict[str, float]:
    fixed = dict(FIXED_PARAMS)
    if fixed_params:
        fixed.update(fixed_params)

    coeffs = clamp_coeff_dict(coeffs)

    return {
        "ibn": build_ibn_value(coeffs["ibn_coe"]),
        "ibp": build_ibp_value(coeffs["ibp_coe"]),
        "rm": build_rm_value(coeffs["rm_coe"]),
        "cm": build_cm_value(coeffs["cm_coe"]),
        "fg_core_n1": float(coeffs["fg_core_n1"]),
        "fg_core_n3": float(coeffs["fg_core_n3"]),
        "fg_core_nd1": float(coeffs["fg_core_nd1"]),
        "fg_core_p3": float(coeffs["fg_core_p3"]),
        "fg_core_mpp": float(coeffs["fg_core_mpp"]),
        "m_core_mpp": float(coeffs["m_core_mpp"]),
        "stk_core_n1": float(coeffs["stk_core_n1"]),
        "stk_core_n3": float(coeffs["stk_core_n3"]),
        "stk_core_nd1": float(coeffs["stk_core_nd1"]),
        "stk_core_p3": float(coeffs["stk_core_p3"]),
        "stk_core_mpp": float(coeffs["stk_core_mpp"]),
        "vddval": float(fixed["vddval"]),
        "vrefval": float(fixed["vrefval"]),
        "offset": float(fixed["offset"]),
        "min_load": int(fixed["min_load"]),
        "max_load": int(fixed["max_load"]),
    }


# =========================
# 3) I/O FILE CHO OCEAN
# =========================
def write_param_file(param_path: Path, params: Dict[str, float]) -> None:
    lines = [str(params[name]) for name in PARAM_FILE_ORDER]
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_result_file(result_path: Path) -> Dict[str, float]:
    if not result_path.exists():
        raise FileNotFoundError(f"Không thấy file result: {result_path}")

    try:
        text = result_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = result_path.read_text(encoding="latin-1")

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) != len(RESULT_ORDER):
        raise ValueError(
            f"Số dòng output không khớp. Mong đợi {len(RESULT_ORDER)} dòng, nhưng nhận {len(raw_lines)} dòng."
        )

    values: List[float] = []
    for line in raw_lines:
        try:
            number = float(line)
        except ValueError as exc:
            raise ValueError(f"Output không phải số: {line}") from exc

        if float(number).is_integer():
            number = int(number)
        values.append(number)

    return {name: values[idx] for idx, name in enumerate(RESULT_ORDER)}


def run_ocean(
    ocn_file: Path,
    workdir: Path,
    ocean_bin: str = "ocean",
    timeout_sec: int = 300,
) -> subprocess.CompletedProcess:
    if not ocn_file.exists():
        raise FileNotFoundError(f"Không thấy file OCEAN: {ocn_file}")

    cmd = [ocean_bin, "-nograph", "-restore", ocn_file.name]
    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        text=False,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
    )
    return result


# =========================
# 4) FITNESS MỀM CHO ANALOG
# =========================
def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _metric_valid(value: float | int | None) -> bool:
    x = _safe_float(value)
    return x is not None and x >= 0


def _score_high_is_good(value: float | int | None, target: float, weight: float) -> float:
    x = _safe_float(value)
    if x is None or x < 0:
        return 0.0
    ratio = x / max(target, 1e-12)
    if ratio <= 1.0:
        return weight * ratio
    # đạt spec vẫn có thưởng nhẹ, nhưng không bùng quá mạnh
    return weight * (1.0 + 0.25 * min(ratio - 1.0, 1.0))


def _score_low_is_good(value: float | int | None, target: float, weight: float) -> float:
    x = _safe_float(value)
    if x is None or x < 0:
        return 0.0
    ratio = target / max(x, 1e-12)
    if ratio <= 1.0:
        return weight * ratio
    return weight * (1.0 + 0.25 * min(ratio - 1.0, 1.0))


def _region_quality(region: float | int | None) -> float:
    x = _safe_float(region)
    if x is None or x < 0:
        return 0.0
    r = int(round(x))
    if r == 2:
        return 1.0
    if r == 3:
        return 0.7
    if r == 1:
        return 0.25
    return 0.0


def compute_fitness(results: Dict[str, float]) -> float:
    score = 0.0

    dcop_ok = int(results.get("dcop_check", 0)) == 1
    if dcop_ok:
        score += 25.0
    else:
        # Không fail cứng; vẫn cho các metric khác kéo điểm lên
        score -= 18.0

    # Chấm từng transistor đại diện ở maxload.
    region_names = ["region_MND1", "region_MN1", "region_MN3", "region_MP3", "region_MPP"]
    region_score = sum(_region_quality(results.get(name)) for name in region_names) / len(region_names)
    score += 22.0 * region_score

    # Regulation/DC quality
    score += _score_low_is_good(results.get("load_regulation"), SPEC_TARGETS["load_regulation_max"], 18.0)
    score += _score_low_is_good(results.get("line_regulation"), SPEC_TARGETS["line_regulation_max"], 16.0)

    if _metric_valid(results.get("vout_err")):
        score += _score_low_is_good(results.get("vout_err"), SPEC_TARGETS["vout_err_max"], 18.0)
    else:
        score -= 6.0

    # AC metrics
    score += _score_high_is_good(results.get("psr_1meg"), SPEC_TARGETS["psr_1meg_min"], 14.0)
    # Giảm thưởng UGB để không bị hút quá mạnh vào vùng bandwidth cao nhưng PM thấp
    score += _score_high_is_good(results.get("ugb_khz"), SPEC_TARGETS["ugb_khz_min"], 8.0)
    score += _score_high_is_good(results.get("av_db"), SPEC_TARGETS["av_db_min"], 14.0)
    # Tăng trọng số PM
    score += _score_high_is_good(results.get("pm_deg"), SPEC_TARGETS["pm_deg_min"], 25.0)

    pm = _safe_float(results.get("pm_deg"))
    if pm is not None:
        if pm < 60.0:
            score -= 2.5 * (60.0 - pm)
        if pm < 50.0:
            score -= 20.0

    # Trong .ocn hiện tại `iq_maxload` chưa dùng làm phạt chính.
    if not _metric_valid(results.get("iq_maxload")):
        score -= 4.0

    # Bonus khi đạt combo quan trọng.
    if _metric_valid(results.get("av_db")) and float(results["av_db"]) >= SPEC_TARGETS["av_db_min"]:
        score += 4.0
    if _metric_valid(results.get("pm_deg")) and float(results["pm_deg"]) >= SPEC_TARGETS["pm_deg_min"]:
        score += 6.0
    if _metric_valid(results.get("psr_1meg")) and float(results["psr_1meg"]) >= 35.0:
        score += 3.0
    if _metric_valid(results.get("load_regulation")) and float(results["load_regulation"]) <= SPEC_TARGETS["load_regulation_max"]:
        score += 3.0
    if _metric_valid(results.get("line_regulation")) and float(results["line_regulation"]) <= SPEC_TARGETS["line_regulation_max"]:
        score += 3.0

    # Penalty nhẹ nếu nhiều metric chính cùng invalid.
    invalid_critical = 0
    for name in ["psr_1meg", "ugb_khz", "av_db", "pm_deg"]:
        if not _metric_valid(results.get(name)):
            invalid_critical += 1
    score -= 3.0 * invalid_critical

    return round(score, 6)


# =========================
# 5) EVALUATOR
# =========================
@dataclass
class EvaluationRecord:
    eval_id: int
    timestamp: str
    fitness: float
    coeffs: Dict[str, int]
    params: Dict[str, float]
    results: Dict[str, float]


class CadenceEvaluator:
    def __init__(
        self,
        workdir: Path,
        ocn_filename: str = "tb_opamp.ocn",
        param_filename: str = "tb_core_param.txt",
        result_filename: str = "tb_core_results.txt",
        ocean_bin: str = "ocean",
        run_dir: Path | None = None,
        dry_run: bool = False,
        timeout_sec: int = 300,
    ) -> None:
        self.workdir = workdir.resolve()
        self.ocn_file = self.workdir / ocn_filename
        self.param_file = self.workdir / param_filename
        self.result_file = self.workdir / result_filename
        self.ocean_bin = ocean_bin
        self.dry_run = dry_run
        self.timeout_sec = timeout_sec
        self.eval_counter = 0
        self.cache: Dict[Tuple[int, ...], EvaluationRecord] = {}

        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = (run_dir or (self.workdir / timestamp)).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, coeffs: Dict[str, int]) -> EvaluationRecord:
        coeffs = clamp_coeff_dict(coeffs)
        key = tuple(coeffs[name] for name in COEFF_ORDER)
        if key in self.cache:
            return self.cache[key]

        self.eval_counter += 1
        eval_id = self.eval_counter
        params = build_param_dict(coeffs)
        write_param_file(self.param_file, params)

        ocean_stdout = ""
        ocean_stderr = ""

        if not self.dry_run:
            try:
                proc = run_ocean(self.ocn_file, self.workdir, self.ocean_bin, self.timeout_sec)
                ocean_stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, (bytes, bytearray)) else str(proc.stdout)
                ocean_stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, (bytes, bytearray)) else str(proc.stderr)
                if proc.returncode != 0:
                    stdout_path = self.run_dir / f"eval_{eval_id:04d}_ocean_stdout.log"
                    stderr_path = self.run_dir / f"eval_{eval_id:04d}_ocean_stderr.log"
                    stdout_path.write_text(ocean_stdout, encoding="utf-8", errors="ignore")
                    stderr_path.write_text(ocean_stderr, encoding="utf-8", errors="ignore")
                    raise RuntimeError(
                        f"OCEAN chạy lỗi ở eval {eval_id}. Xem log:\n"
                        f"  {stdout_path}\n"
                        f"  {stderr_path}"
                    )
            except subprocess.TimeoutExpired as exc:
                stdout_text = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, (bytes, bytearray)) else str(exc.stdout)
                stderr_text = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, (bytes, bytearray)) else str(exc.stderr)
                (self.run_dir / f"eval_{eval_id:04d}_ocean_stdout.log").write_text(stdout_text, encoding="utf-8", errors="ignore")
                (self.run_dir / f"eval_{eval_id:04d}_ocean_stderr.log").write_text(stderr_text, encoding="utf-8", errors="ignore")
                raise RuntimeError(f"OCEAN timeout sau {self.timeout_sec}s ở eval {eval_id}") from exc

        results = read_result_file(self.result_file)
        fitness = compute_fitness(results)

        shutil.copy2(self.param_file, self.run_dir / f"eval_{eval_id:04d}_param.txt")
        shutil.copy2(self.result_file, self.run_dir / f"eval_{eval_id:04d}_result.txt")

        if ocean_stdout:
            (self.run_dir / f"eval_{eval_id:04d}_ocean_stdout.log").write_text(
                ocean_stdout, encoding="utf-8", errors="ignore"
            )
        if ocean_stderr:
            (self.run_dir / f"eval_{eval_id:04d}_ocean_stderr.log").write_text(
                ocean_stderr, encoding="utf-8", errors="ignore"
            )

        record = EvaluationRecord(
            eval_id=eval_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            fitness=fitness,
            coeffs=coeffs,
            params=params,
            results=results,
        )
        self.cache[key] = record
        return record

    def dump_cache_csv(self) -> Path:
        csv_path = self.run_dir / "all_evaluations.csv"
        fieldnames = ["eval_id", "timestamp", "fitness"] + COEFF_ORDER + PARAM_FILE_ORDER + RESULT_ORDER

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in sorted(self.cache.values(), key=lambda r: r.eval_id):
                row = {
                    "eval_id": record.eval_id,
                    "timestamp": record.timestamp,
                    "fitness": record.fitness,
                }
                row.update(record.coeffs)
                row.update(record.params)
                row.update(record.results)
                writer.writerow(row)
        return csv_path


# =========================
# 6) PSO
# =========================
@dataclass
class PSOConfig:
    population: int = 8
    iterations: int = 10
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.4
    c2: float = 1.4
    seed: int = 42
    vmax_frac: float = 0.15
    guide_frac: float = 0.4
    guide_jitter_frac: float = 0.15
    stagnation_limit: int = 4


class PSOOptimizer:
    def __init__(self, evaluator: CadenceEvaluator, cfg: PSOConfig) -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        random.seed(cfg.seed)

        self.dim = len(COEFF_ORDER)
        self.lower = [COEFF_BOUNDS[name][0] for name in COEFF_ORDER]
        self.upper = [COEFF_BOUNDS[name][1] for name in COEFF_ORDER]
        self.spans = [hi - lo for lo, hi in zip(self.lower, self.upper)]
        self.vmax = [max(1.0, cfg.vmax_frac * span) for span in self.spans]

    def _random_position(self) -> List[float]:
        return [random.uniform(lo, hi) for lo, hi in zip(self.lower, self.upper)]

    def _guide_position(self, guide: Sequence[int], jitter_frac: float) -> List[float]:
        pos: List[float] = []
        for i, center in enumerate(guide):
            span = self.spans[i]
            jitter = max(1.0, jitter_frac * span)
            pos.append(float(center) + random.uniform(-jitter, jitter))
        return self._clip_position(pos)

    def _position_to_coeffs(self, pos: Sequence[float]) -> Dict[str, int]:
        coeffs = {}
        for i, name in enumerate(COEFF_ORDER):
            lo, hi = COEFF_BOUNDS[name]
            coeffs[name] = max(lo, min(hi, int(round(pos[i]))))
        return coeffs

    def _clip_position(self, pos: Sequence[float]) -> List[float]:
        return [max(self.lower[i], min(self.upper[i], float(pos[i]))) for i in range(self.dim)]

    def _clip_velocity(self, vel: Sequence[float]) -> List[float]:
        return [max(-self.vmax[i], min(self.vmax[i], float(vel[i]))) for i in range(self.dim)]

    def _initial_population(self) -> List[List[float]]:
        guide_count = min(self.cfg.population, max(0, int(round(self.cfg.population * self.cfg.guide_frac))))
        pop: List[List[float]] = []

        for i in range(guide_count):
            guide = GUIDE_COEFFS[i % len(GUIDE_COEFFS)]
            pop.append(self._guide_position(guide, self.cfg.guide_jitter_frac))

        while len(pop) < self.cfg.population:
            pop.append(self._random_position())

        return pop

    def run(self) -> Tuple[EvaluationRecord, Path, Path]:
        pop = self._initial_population()
        vel = [[random.uniform(-self.vmax[d], self.vmax[d]) * 0.2 for d in range(self.dim)] for _ in range(self.cfg.population)]

        pbest_pos = [p[:] for p in pop]
        pbest_rec: List[EvaluationRecord | None] = [None for _ in range(self.cfg.population)]
        no_improve = [0 for _ in range(self.cfg.population)]

        gbest_pos: List[float] | None = None
        gbest_rec: EvaluationRecord | None = None

        history_path = self.evaluator.run_dir / "history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as f_hist:
            writer = csv.DictWriter(
                f_hist,
                fieldnames=["iteration", "particle", "fitness"] + COEFF_ORDER + RESULT_ORDER,
            )
            writer.writeheader()

            for iteration in range(self.cfg.iterations):
                if self.cfg.iterations == 1:
                    w = self.cfg.w_min
                else:
                    frac = iteration / (self.cfg.iterations - 1)
                    w = self.cfg.w_max - frac * (self.cfg.w_max - self.cfg.w_min)

                iter_best: EvaluationRecord | None = None
                iter_best_particle = -1

                for i in range(self.cfg.population):
                    coeffs = self._position_to_coeffs(pop[i])
                    record = self.evaluator.evaluate(coeffs)

                    row = {
                        "iteration": iteration,
                        "particle": i,
                        "fitness": record.fitness,
                    }
                    row.update(record.coeffs)
                    row.update(record.results)
                    writer.writerow(row)
                    f_hist.flush()

                    if (pbest_rec[i] is None) or (record.fitness > pbest_rec[i].fitness):
                        pbest_rec[i] = record
                        pbest_pos[i] = pop[i][:]
                        no_improve[i] = 0
                    else:
                        no_improve[i] += 1

                    if (gbest_rec is None) or (record.fitness > gbest_rec.fitness):
                        gbest_rec = record
                        gbest_pos = pop[i][:]

                    if (iter_best is None) or (record.fitness > iter_best.fitness):
                        iter_best = record
                        iter_best_particle = i

                assert gbest_pos is not None and gbest_rec is not None and iter_best is not None
                print(
                    f"[ITER {iteration+1}/{self.cfg.iterations}] best={iter_best.fitness:.3f} "
                    f"global={gbest_rec.fitness:.3f} particle={iter_best_particle}"
                )

                for i in range(self.cfg.population):
                    # reseed particle đứng im quá lâu
                    if no_improve[i] >= self.cfg.stagnation_limit:
                        if gbest_pos is not None:
                            pop[i] = self._guide_position(self._position_to_coeffs(gbest_pos).values(), 0.10)
                        else:
                            guide = GUIDE_COEFFS[i % len(GUIDE_COEFFS)]
                            pop[i] = self._guide_position(guide, 0.12)
                        vel[i] = [0.0 for _ in range(self.dim)]
                        no_improve[i] = 0
                        continue

                    for d in range(self.dim):
                        r1 = random.random()
                        r2 = random.random()
                        cognitive = self.cfg.c1 * r1 * (pbest_pos[i][d] - pop[i][d])
                        social = self.cfg.c2 * r2 * (gbest_pos[d] - pop[i][d])
                        vel[i][d] = w * vel[i][d] + cognitive + social
                    vel[i] = self._clip_velocity(vel[i])
                    pop[i] = self._clip_position([pop[i][d] + vel[i][d] for d in range(self.dim)])

        assert gbest_rec is not None
        all_eval_path = self.evaluator.dump_cache_csv()

        best_json_path = self.evaluator.run_dir / "best_result.json"
        with best_json_path.open("w", encoding="utf-8") as f_best:
            json.dump(
                {
                    "fitness": gbest_rec.fitness,
                    "coeffs": gbest_rec.coeffs,
                    "params": gbest_rec.params,
                    "results": gbest_rec.results,
                },
                f_best,
                indent=2,
                ensure_ascii=False,
            )

        return gbest_rec, history_path, all_eval_path


# =========================
# 7) CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSO + Cadence OCEAN runner cho tb_opamp.ocn")
    parser.add_argument("--mode", choices=["eval", "pso"], default="pso")
    parser.add_argument(
        "--coeffs",
        nargs=len(COEFF_ORDER),
        type=int,
        help="Dùng cho mode=eval. Thứ tự là: " + ", ".join(COEFF_ORDER),
    )
    parser.add_argument("--workdir", type=str, default=".", help="Thư mục chứa tb_opamp.ocn")
    parser.add_argument("--ocn", type=str, default="tb_opamp.ocn", help="Tên file OCEAN")
    parser.add_argument("--ocean-bin", type=str, default="ocean", help="Binary OCEAN")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=300, help="Timeout cho mỗi eval OCEAN")
    parser.add_argument("--dry-run", action="store_true", help="Không gọi OCEAN, chỉ đọc tb_core_results.txt hiện có")
    return parser.parse_args()


# =========================
# 8) MAIN
# =========================
def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()

    evaluator = CadenceEvaluator(
        workdir=workdir,
        ocn_filename=args.ocn,
        ocean_bin=args.ocean_bin,
        dry_run=args.dry_run,
        timeout_sec=args.timeout_sec,
    )

    print(f"[INFO] Workdir : {evaluator.workdir}")
    print(f"[INFO] Run dir : {evaluator.run_dir}")
    print(f"[INFO] OCN     : {evaluator.ocn_file}")
    print(f"[INFO] Dry-run : {evaluator.dry_run}")
    print(f"[INFO] Timeout : {evaluator.timeout_sec}s")

    try:
        if args.mode == "eval":
            if not args.coeffs:
                raise ValueError("mode=eval bắt buộc phải truyền --coeffs")
            coeffs = coeff_vector_to_dict(args.coeffs)
            rec = evaluator.evaluate(coeffs)
            evaluator.dump_cache_csv()

            print("\n=== KẾT QUẢ EVALUATE 1 ĐIỂM ===")
            print("Fitness:", rec.fitness)
            print("Coeffs :", json.dumps(rec.coeffs, indent=2, ensure_ascii=False))
            print("Params :", json.dumps(rec.params, indent=2, ensure_ascii=False))
            print("Results:", json.dumps(rec.results, indent=2, ensure_ascii=False))
            print(f"\n[OK] File tổng hợp: {evaluator.run_dir / 'all_evaluations.csv'}")
            return 0

        cfg = PSOConfig(
            population=args.population,
            iterations=args.iterations,
            seed=args.seed,
        )
        optimizer = PSOOptimizer(evaluator, cfg)
        best_rec, history_path, all_eval_path = optimizer.run()

        print("\n=== BEST RESULT ===")
        print("Fitness:", best_rec.fitness)
        print("Coeffs :", json.dumps(best_rec.coeffs, indent=2, ensure_ascii=False))
        print("Params :", json.dumps(best_rec.params, indent=2, ensure_ascii=False))
        print("Results:", json.dumps(best_rec.results, indent=2, ensure_ascii=False))
        print(f"\n[OK] History: {history_path}")
        print(f"[OK] All eval: {all_eval_path}")
        print(f"[OK] Best   : {evaluator.run_dir / 'best_result.json'}")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

