# Mô phỏng Monte Carlo xung đột đường bay drone

Ước lượng **xác suất xảy ra xung đột** giữa các drone trong một vùng trời
5 km × 5 km **trước khi** có bất kỳ thuật toán điều phối / xử lý xung đột nào.
Xác suất được tách theo bốn pha bay — **Cất cánh, Hạ cánh, Trên hành trình,
Bãi đỗ** — và cho **toàn kế hoạch bay**.

Đây là kịch bản "baseline" (chưa điều phối): các kế hoạch bay được đăng ký ngẫu
nhiên và độc lập, không nhường nhau. Con số thu được chính là mức rủi ro nền mà
một thuật toán tránh xung đột sẽ phải kéo về 0.

> 📐 **Mô hình giải tích & phân tích giới hạn** của xác suất xung đột (công thức
> đóng `P_p = 1 − e^(−(N−1)a_p/T)`, các giới hạn, đối chiếu với mô phỏng): xem
> [`docs/theory.md`](docs/theory.md).
>
> 🔀 **Đối chứng baseline vs thuật toán điều phối** (ghép bộ giải xung đột
> `scrp_simple` của repo time-departure: xoá sạch cất/hạ cánh & bãi đỗ hạ cánh,
> giảm >50% "toàn KHB" với trễ vài giây): xem
> [`docs/comparison.md`](docs/comparison.md).

---

## 1. Mô hình mô phỏng

### 1.1 Vùng trời (tĩnh, cố định cho mọi trial)

| Thành phần | Đặc tả |
|---|---|
| Khu vực | 5000 m × 5000 m |
| Đường bay | ~10 đường, **không giao nhau** |
| Mỗi đường bay | 1 điểm cất cánh, 1 điểm hạ cánh, **2–4 waypoint** |
| Mỗi điểm cất/hạ cánh | **3–5 bãi đỗ** |
| Mỗi bãi đỗ | thời gian dừng tối đa `pad_t` ∈ **2–4 phút** |
| Loại drone | **10 loại**, mỗi loại có dải vận tốc + bán kính an toàn |
| Khoảng cách an toàn | tối thiểu **50 m** (một số loại lớn hơn: 50…95 m) |

**Bảo đảm không giao nhau.** Khu vực được chia thành 10 dải ngang không chồng
nhau theo trục *y*. Mỗi đường bay bị giới hạn hoàn toàn trong dải của nó và đi
đơn điệu theo *x* (cất cánh bên trái → hạ cánh bên phải). Vì các dải cách nhau
≥ 200 m theo *y*, hai đường bay bất kỳ không thể cắt nhau và luôn cách nhau
theo phương ngang > 50 m. Xem `outputs/airspace_map.png`.

### 1.2 Kế hoạch bay (ngẫu nhiên, thay đổi theo trial)

Các kế hoạch bay được tạo **lần lượt**. Mỗi kế hoạch:

- chọn ngẫu nhiên: đường bay, loại drone, bãi đỗ đi/đến, vận tốc **từng chặng**;
- có **thời điểm tạo** (rải đều trên `horizon`) và **đăng ký thời gian xuất
  phát** = thời điểm tạo + độ trễ; drone cất cánh đúng thời gian đã đăng ký;
- gồm: điểm đi, điểm đến, waypoint, vận tốc từng chặng, **thời gian cất cánh**
  (mặt đất → waypoint đầu), **thời gian hạ cánh** (waypoint cuối → bãi đỗ),
  loại drone, thời gian dừng bãi đỗ (≤ `pad_t`).

Trục thời gian một chuyến bay:

```
 t_dep ──cất cánh──► WP1 ──hành trình (nhiều chặng)──► WPk ──hạ cánh──► t_arr
   │                                                                      │
 [chiếm bãi đỗ đi]                                            [chiếm bãi đỗ đến]
```

### 1.3 Định nghĩa xung đột theo pha

| Pha | Điều kiện xung đột |
|---|---|
| **Cất cánh** | Hai chuyến dùng **cùng một điểm cất cánh** có cửa sổ cất cánh **chồng lấn thời gian**. Điểm cất cánh hoạt động như "đường băng" dùng chung — trong lúc một phương tiện đang cất cánh, phương tiện khác không được chen vào. |
| **Hạ cánh** | Tương tự, tại **cùng một điểm hạ cánh**. |
| **Trên hành trình** | Hai drone tiến vào **khoảng cách < khoảng cách an toàn** (max hai bán kính an toàn, ≥ 50 m) trong khi cả hai đang bay hành trình và thời gian chồng lấn. |
| **Bãi đỗ** | Hai chuyến chiếm **cùng một bãi đỗ** với khoảng thời gian chiếm dụng **chồng lấn**. |
| **Toàn kế hoạch bay** | Dính xung đột ở **bất kỳ** pha nào ở trên. |

> Vì các đường bay tách biệt theo không gian, xung đột *hành trình* trên thực tế
> chỉ xảy ra giữa hai drone **cùng một đường bay** (một chiếc đuổi kịp chiếc kia).
> Bộ phát hiện vẫn kiểm tra tổng quát mọi cặp (có lọc broad-phase theo thời gian
> và hộp bao) nên kết luận này là hệ quả của hình học, không phải giả định cứng.

### 1.4 Cách tính xác suất

Đại lượng chính — **xác suất theo mỗi kế hoạch bay** — gộp toàn bộ chuyến bay
của mọi trial:

```
P_pha = (số kế hoạch bay dính xung đột ở pha đó) / (tổng số kế hoạch bay)
```

kèm khoảng tin cậy 95% (Wilson). Ngoài ra còn có **xác suất theo kịch bản** =
tỉ lệ trial có ≥ 1 xung đột ở pha đó.

---

## 2. Cách chạy

```bash
pip install -r requirements.txt

# Chạy mặc định: 3000 trial × 30 KHB, có quét mật độ, xuất hình vào outputs/
python run_experiment.py

# Tùy chỉnh
python run_experiment.py --trials 5000 --flights 40 --horizon 1800
python run_experiment.py --sweep 10 20 30 40 50 60   # quét mật độ giao thông
python run_experiment.py --no-plots                   # bỏ vẽ hình (chạy nhanh)

# Kiểm thử bất biến (không cần pytest)
python tests/test_sim.py
```

Tham số chính (`python run_experiment.py --help`): `--trials`, `--flights`,
`--horizon`, `--routes`, `--types`, `--seed`, `--dt`, `--sweep`.

---

## 3. Kết quả mẫu

3000 trial × 30 kế hoạch bay/trial, horizon 30 phút (90 000 KHB):

| Pha | P(theo KHB) | P(theo kịch bản) |
|---|---:|---:|
| Cất cánh | ~8.1 % | ~73 % |
| Hạ cánh | ~8.1 % | ~73 % |
| Trên hành trình | ~6.2 % | ~64 % |
| Bãi đỗ | ~14.8 % | ~91 % |
| **Toàn kế hoạch bay** | **~20.7 %** | **~97 %** |

Nhận xét:

- **Bãi đỗ** là pha rủi ro nhất (đặt bãi đỗ trùng nhau theo thời gian rất dễ xảy
  ra khi chọn ngẫu nhiên).
- Rủi ro tăng gần **tuyến tính theo mật độ giao thông** — xem
  `outputs/density_sweep.png`. Ở mức 30 KHB, ~1/5 mỗi kế hoạch bay dính ít nhất
  một xung đột; ở mức kịch bản, gần như chắc chắn có xung đột.

Các con số phụ thuộc tham số (mật độ, horizon, số bãi đỗ, dải vận tốc). Đổi seed
và tham số để khảo sát độ nhạy.

### Hình xuất ra `outputs/`
- `airspace_map.png` — 10 đường bay không giao nhau, điểm cất/hạ cánh, bãi đỗ.
- `phase_probabilities.png` — cột xác suất theo pha kèm KTC 95%.
- `density_sweep.png` — xác suất theo mật độ giao thông.
- `results.json` — số liệu đầy đủ.

---

## 4. Cấu trúc mã nguồn

```
drone_sim/
  airspace.py    # sinh vùng trời: đường bay không giao nhau, bãi đỗ, loại drone
  flight.py      # kế hoạch bay + quỹ đạo và mốc thời gian theo pha
  conflicts.py   # phát hiện xung đột 4 pha (sweep-line + broad-phase + lấy mẫu)
  montecarlo.py  # sinh KHB ngẫu nhiên, chạy trial, tổng hợp xác suất + KTC Wilson
run_experiment.py# CLI: chạy thí nghiệm, in bảng, vẽ hình, lưu JSON
tests/test_sim.py# kiểm thử: không giao nhau, đúng đặc tả, phát hiện xung đột
```

---

## 5. Giới hạn & hướng mở rộng

- Mô hình **chưa có điều phối**: đây là baseline để đo rủi ro nền.
- Bước tiếp theo tự nhiên: thêm thuật toán xử lý xung đột (xếp hàng bãi
  đỗ/đường băng, dịch thời gian xuất phát, giãn cách hành trình) rồi chạy lại
  cùng bộ Monte Carlo để đo mức giảm xác suất.
- Có thể mở rộng: đường bay giao nhau (bỏ ràng buộc dải), gió/độ bất định vận
  tốc, drone bay nhiều chặng, chia sẻ điểm cất/hạ cánh giữa nhiều đường bay.
