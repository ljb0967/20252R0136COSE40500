import math
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ExamStats:
    full: float
    observed_min: float
    observed_max: float
    mean: float
    median: float

# 7

@dataclass
class ModelConfig:
    n_students: int = 73
    n_sims: int = 20000
    seed: int = 42

    mid: ExamStats = field(default_factory=lambda: ExamStats(
        full=100, observed_min=23, observed_max=91, mean=55.1, median=55
    ))
    fin: ExamStats = field(default_factory=lambda: ExamStats(
        full=100, observed_min=0, observed_max=132, mean=70, median=70
    ))



def _estimate_sd_from_range(obs_min, obs_max):
    # rule-of-thumb: sd ≈ range/4
    sd = (obs_max - obs_min) / 4.0
    # 너무 작아지는 것 방지
    return max(sd, 1e-6)

def truncated_normal(mean, sd, low, high, size, rng):
    """
    Rejection sampling 기반 절단 정규.
    size: int 또는 tuple(shape) 모두 지원
    """
    # size가 튜플이면 전체 원소 개수로 펼쳐서 뽑은 뒤 reshape
    if isinstance(size, tuple):
        shape = size
        total = int(np.prod(shape))
    else:
        shape = (int(size),)
        total = int(size)

    out = np.empty(total, dtype=float)
    filled = 0

    while filled < total:
        n = (total - filled) * 2  # 넉넉히 뽑기
        samp = rng.normal(loc=mean, scale=sd, size=n)
        samp = samp[(samp >= low) & (samp <= high)]
        take = min(len(samp), total - filled)
        out[filled:filled + take] = samp[:take]
        filled += take

    return out.reshape(shape)


def _clamp(x, lo=1e-6, hi=1-1e-6):
    return max(lo, min(hi, x))

def sample_scores_beta(stats: ExamStats, rng, size):
    """
    (정규X) Beta 기반 비정규 분포 샘플링:
    - [min, max] 범위를 항상 만족
    - mean/median으로 왜도를 반영(가능한 범위 내에서)
    """
    a = float(stats.observed_min)
    b = float(stats.observed_max)
    if b <= a:
        # 범위가 이상하면 전부 a로
        return np.full(size, a, dtype=float)

    # 0~1로 스케일
    mean01 = _clamp((float(stats.mean) - a) / (b - a))
    med01  = _clamp((float(stats.median) - a) / (b - a))

    # 1) median을 이용해 alpha/beta를 "대충" 맞추는 근사(베타분포 median 근사식 사용)
    #    median ≈ (α - 1/3) / (α + β - 2/3)  (α,β>1에서 꽤 잘 맞음)
    #    mean = α/(α+β)
    # => mean, median으로 α를 닫힌형태로 근사해 볼 수 있음
    #    (불안정할 때는 fallback으로 mean만 사용)
    alpha = None
    beta = None

    # mean==median이면 대칭에 가깝다고 보고 mean 기반으로만 설정
    if abs(mean01 - med01) > 1e-4 and (1 - (med01 / mean01)) != 0:
        denom = (1.0 - (med01 / mean01))
        num = (1.0 - 2.0 * med01) / 3.0
        a_hat = num / denom  # 근사 alpha
        if a_hat > 0:
            b_hat = a_hat * (1.0 - mean01) / mean01
            if b_hat > 0:
                alpha, beta = a_hat, b_hat

    # 2) fallback: mean만으로 beta를 잡기 (농도(concentration)로 퍼짐 조절)
    #    너무 약하게 잡히면 이상한 모양이 나와서, 적당히 안정적인 k를 둠
    if alpha is None or beta is None:
        k = 6.0  # 농도: 클수록 평균 근처로 몰림(분산 감소)
        alpha = mean01 * k
        beta = (1.0 - mean01) * k

    # 3) 파라미터가 너무 작으면 U자/극단 분포가 과해질 수 있어서 하한을 둠
    alpha = max(alpha, 0.8)
    beta = max(beta, 0.8)

    x = rng.beta(alpha, beta, size=size)      # 0~1
    return a + x * (b - a)                    # [min,max]



def norm_cdf(x):
    """표준정규 CDF (scipy 없이)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ----------------------------
# 3) 핵심: 입력 -> 상위비율
# ----------------------------
def predict_top_ratio(mid_score, fin_score, config=ModelConfig(), method="mc"):
    """
    입력:
      mid_score: 나의 중간고사 점수
      fin_score: 나의 기말고사 점수
    출력:
      상위비율(상위 x%)에 해당하는 값(dict)  ※ 모델 입력/출력 준수
    """
    # 종합 점수는 만점 보정 후 50:50
    my_total = 0.5 * (mid_score / config.mid.full) + 0.5 * (fin_score / config.fin.full)

    # ---- (A) 정규근사 모델 ----
    # 중간/기말을 (만점 보정 비율)로 보고 정규 + 독립 가정
    if method.lower() in ["normal", "gaussian"]:
        # 중간 비율 분포 파라미터
        mid_mean_p = config.mid.mean / config.mid.full
        mid_sd_p = _estimate_sd_from_range(config.mid.observed_min, config.mid.observed_max) / config.mid.full

        # 기말 비율 분포는 "중간 비율 분포 형태"를 따른다고 보고 sd는 중간과 동일하게,
        # mean은 기말 중앙값(예상) 70점을 반영하도록 70/132로 둠
        fin_mean_p = config.fin.median / config.fin.full
        fin_sd_p = mid_sd_p

        # total = 0.5*mid_p + 0.5*fin_p
        total_mean = 0.5 * mid_mean_p + 0.5 * fin_mean_p
        total_sd = math.sqrt((0.5**2) * (mid_sd_p**2) + (0.5**2) * (fin_sd_p**2))

        z = (my_total - total_mean) / (total_sd + 1e-12)
        percentile = norm_cdf(z)  # 내가 이길 확률(= 아래쪽 비율)
        top_ratio = 1.0 - percentile  # 상위 비율(작을수록 좋음)
        return {
            "방법": "정규분포 근사",
            "상위 비율": float(top_ratio),          # 0.2 -> 상위 20%
            "상위 퍼센트": float(top_ratio * 100),  # 상위 xx%
        }

    # ---- (B) 몬테카를로 모델 ----
    elif method.lower() in ["mc", "montecarlo", "simulation"]:
        rng = np.random.default_rng(config.seed)

        # 중간 분포(원점수) 파라미터
        mid_sd = _estimate_sd_from_range(config.mid.observed_min, config.mid.observed_max)
        # 기말 분포는 "중간 비율 분포"를 따른다고 보고:
        # - 비율 sd 동일
        # - 평균은 중앙값 70을 반영하도록 mean=70으로 둠 (정확한 mean 정보 없어서 근사)
        mid_sd_p = mid_sd / config.mid.full
        fin_sd = mid_sd_p * config.fin.full
        fin_mean = config.fin.median  # 70점에 맞춤

        N = config.n_students
        sims = config.n_sims

        others_mid = sample_scores_beta(config.mid, rng, size=(sims, N-1))
        others_fin = sample_scores_beta(config.fin, rng, size=(sims, N-1))


        others_total = 0.5 * (others_mid / config.mid.full) + 0.5 * (others_fin / config.fin.full)

        # 내 등수 계산: 나보다 점수 큰 사람 수 + 1
        better = (others_total > my_total).sum(axis=1)
        my_rank = better + 1  # 1등이 최고

        top_ratio = my_rank / N  # 0.2면 상위 20%
        # 요약 통계
        mean_top = float(top_ratio.mean())
        lo, hi = np.quantile(top_ratio, [0.05, 0.95])  # 90% 구간

        return {
            "방법": "몬테카를로 시뮬레이션",
            "상위 비율": mean_top,
            "상위 퍼센트": mean_top * 100,
            "신뢰도 90%구간(상위5%~상위90%)": (float(lo * 100), float(hi * 100)),
        }

    else:
        raise ValueError("method must be 'normal' or 'mc'.")

if __name__ == "__main__":
    def read_int_or_keep(prompt: str, current: float) -> float:
        s = input(f"{prompt} (현재값: {current}) [모르면 엔터] : ").strip()
        if s == "":
            return current
        return float(s)

    print("-- 중간고사 기말고사 반영비율은 1:1로 가정 --")
    print("-- 점수/분포값은 모르면 엔터로 넘어가면 기본값이 유지됩니다. --\n")

    config = ModelConfig()

    my_mid = read_int_or_keep("내 중간고사 점수", 0)
    my_fin = read_int_or_keep("내 기말고사 점수", 0)
    config.mid.full = read_int_or_keep("중간고사 총점", 100)
    config.fin.full = read_int_or_keep("기말고사 총점", 100)

    print("\n-- 중간고사 분포 입력 (모르면 엔터) --")
    config.mid.mean = read_int_or_keep("중간고사 평균", config.mid.mean)
    config.mid.median = read_int_or_keep("중간고사 중앙값", config.mid.median)
    config.mid.observed_max = read_int_or_keep("중간고사 최대값", config.mid.observed_max)
    config.mid.observed_min = read_int_or_keep("중간고사 최소값", config.mid.observed_min)

    print("\n-- 기말고사 분포 입력 (모르면 엔터) --")
    config.fin.mean = read_int_or_keep("기말고사 평균", config.fin.mean)
    config.fin.median = read_int_or_keep("기말고사 중앙값", config.fin.median)
    config.fin.observed_max = read_int_or_keep("기말고사 최대값", config.fin.observed_max)
    config.fin.observed_min = read_int_or_keep("기말고사 최소값", config.fin.observed_min)

    print("\n[입력 반영된 설정]")
    print(f"- 중간: mean={config.mid.mean}, median={config.mid.median}, min={config.mid.observed_min}, max={config.mid.observed_max}, full={config.mid.full}")
    print(f"- 기말: mean={config.fin.mean}, median={config.fin.median}, min={config.fin.observed_min}, max={config.fin.observed_max}, full={config.fin.full}\n")

    print(predict_top_ratio(my_mid, my_fin, config=config, method="normal"))
    print(predict_top_ratio(my_mid, my_fin, config=config, method="mc"))

