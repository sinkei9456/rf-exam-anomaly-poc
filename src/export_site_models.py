"""두 번째 실험(src/run_sites.sh)의 모델을 장소 폴더 형식(src/site_train.py 출력과 같음)으로 내보낸다 (2026-11-19).

runs/site/<장소>/ → models/site/<장소>/
  site.json        배경 잡음 세기, (나) 픽셀 차이와 CAE 9개(크기 65k·21k·6k × 시드 0·1·2)의 커널·판정 임계값·학습 기록
  pixel_stats.npz  학습셋 정상 스냅샷의 픽셀별 평균·표준편차
  cae<크기>_s<시드>.pt
추론: python src/predict.py --site models/site/indoor --iq <IQ 파일> [--cae 21k_s1]
주의: 이 모델들은 장소 배경에 약한 와이파이·블루투스를 합성한 정상 스냅샷으로 학습했다(두 번째 실험 설정).
      배경 잡음 세기의 단위는 data/background/<장소>/ 파일 단위다. 실내는 .cs8 을 int8 ÷ 128 로 읽은 값이라 원본 .cs8 을 그대로 넣어도 맞고,
      공원은 이 저장소 릴리스의 Park02_*.npy 단위다.
사용: python src/export_site_models.py   (python src/site_setup.py 와 bash src/run_sites.sh 를 마친 뒤)
"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from site_setup import SITES, ORDER, BG, noise_power

SIZES = ["65k", "21k", "6k"]


def main():
    for site in ORDER:
        src = C.ROOT / "runs" / "site" / site; dst = C.MODELS / "site" / site; dst.mkdir(parents=True, exist_ok=True)
        nf, per = noise_power([BG / f for f in SITES[site]["files"]["train"]])
        shutil.copy2(src / "pixel_stats.npz", dst / "pixel_stats.npz")
        ps = json.load(open(src / "pixel_select.json", encoding="utf-8"))["B"]
        cae = {}
        for z in SIZES:
            sel = json.load(open(src / f"cae{z}_select.json", encoding="utf-8"))
            for s, v in sel.items():
                f = f"cae{z}_s{s}.pt"; shutil.copy2(src / f, dst / f)
                cae[f"{z}_s{s}"] = dict(size=z, seed=int(s), channels=v["channels"], n_params=v["n_params"], rows=v["rows"], cols=v["cols"],
                                        thr={"0.01": v["thr_fpr1"], "0.001": v["thr_fpr01"]}, best_epoch=v["best_epoch"],
                                        history=v["history"], train_seconds=v["train_seconds"], file=f)
        meta = dict(name=site, site_name=SITES[site]["name"], session=SITES[site]["session"],
                    trained_on="두 번째 실험: 장소 배경 IQ 데이터에 약한 와이파이·블루투스를 합성한 정상 스냅샷 10,000장",
                    input=dict(fs_hz=C.FS, snapshot_s=C.WIN, center_mhz=SITES[site]["fc"] / 1e6,
                               image="uint8 256 x 97, 0 = -6 dB, 255 = +34 dB, 0 dB = background noise level"),
                    noise_power=nf, noise_power_per_segment=per,
                    noise_units=("data/background/park/Park02_*.npy (이 저장소 릴리스) 단위" if site == "park"
                                 else ".cs8 을 int8 ÷ 128 로 읽은 단위(predict.py 의 open_iq 와 같음)"),
                    pixel=dict(rows=ps["rows"], cols=ps["cols"], thr={"0.01": ps["thr_fpr1"], "0.001": ps["thr_fpr01"]}),
                    cae=cae, default_cae="65k_s0")
        json.dump(meta, open(dst / "site.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(site, "→", dst.relative_to(C.ROOT), f"CAE {len(cae)}개", flush=True)
    print("EXPORT_SITE_MODELS_DONE")


if __name__ == "__main__":
    main()
