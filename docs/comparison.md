# Đối chứng: baseline vs thuật toán time-departure

So sánh **xác suất xung đột trước và sau khi điều phối**, dùng bộ giải xung đột
`scrp_simple.ConflictResolver` từ repo
[`hunglq-vht/time-departure`](https://github.com/hunglq-vht/time-departure).

Script: [`../compare_resolver.py`](../compare_resolver.py) · Hình:
[`baseline_vs_resolved.png`](../outputs/baseline_vs_resolved.png),
[`resolver_sweep.png`](../outputs/resolver_sweep.png),
[`resolver_maxwait.png`](../outputs/resolver_maxwait.png) · Số liệu:
[`compare_results.json`](../outputs/compare_results.json).

## 1. Khớp đúng mô hình scrp_simple (đã kiểm tra chéo 2 repo)

Theo `scrp_simple/DESIGN.md` (§4), thuật toán có **đúng 4 ràng buộc trên 3 loại
tài nguyên**:

| Ràng buộc | Tài nguyên | Nội dung |
|---|---|---|
| **C1** | Lane (chặng waypoint→waypoint) | Giữ ≥ `MIN_SEPARATION_M = 50m` |
| **C2** | **Không phận** thẳng đứng trên vertiport **đi** | 1 drone ascent/lúc |
| **C3** | **Không phận** thẳng đứng trên vertiport **đến** | 1 drone descent/lúc |
| **C4** | **Bề mặt pad hạ cánh** | Drone trước chưa rời pad thì drone sau chưa hạ được |

**Điểm mấu chốt (đính chính so với bản đối chứng trước):** bề mặt pad chỉ bị
ràng buộc **khi HẠ CÁNH (C4)**. Cất cánh chỉ là ascent chiếm **không phận**
(C2) — drone rời pad khi bay lên, **không có "đặt chỗ pad cất cánh" riêng**.
Nói cách khác, một pad là một pad: việc chiếm bề mặt pad được mô hình hoá ở pha
hạ cánh, còn cất cánh đã được C2 tuần tự hoá qua không phận. Vì vậy **không tồn
tại kênh xung đột "bãi đỗ cất cánh"** — bản đối chứng này bỏ kênh đó.

Hai điểm để khớp *đúng điều kiện* của scrp_simple (áp dụng đồng nhất cho cả
baseline lẫn resolved):

1. **Ràng buộc pad = HẠ CÁNH.** Pha "Bãi đỗ" = xung đột bề mặt pad hạ cánh (C4),
   occupancy = `pad.occupation_duration = pad.max_dwell`.
2. **Khoảng cách an toàn 50m thống nhất.** scrp_simple không có MSD theo loại
   (`drone_type` chỉ là chuỗi) và enforce `MIN_SEPARATION_M = 50` cố định.
3. **Dung sai biên** `EPS_T = 1ms`, `EPS_M = 1cm` để bỏ qua chồng lấn ~nano-giây
   khi resolver xếp các cửa sổ sát nhau đúng biên.

**Mệnh đề từ chối:** nếu độ trễ cần thiết > `max_wait` thì kế hoạch **không được
duyệt** (rejected) — chuyến đó **không bay** nên **không tính là vi phạm**; nó
chỉ tính vào *tỉ lệ từ chối*.

## 2. Phương pháp

Mỗi trial: sinh **cùng một** bộ N kế hoạch bay, rồi

- **BASELINE** — đo xung đột trực tiếp, không điều phối.
- **RESOLVED** — `resolve_batch` duyệt tuần tự theo thời gian; mỗi yêu cầu có
  thể bị dời giờ xuất phát (delay), được gán bãi đỗ hạ cánh, hoặc bị từ chối.
  Đo lại xung đột trên tập **đã duyệt** bằng đúng bộ phát hiện C1–C4.

Ánh xạ: điểm cất/hạ cánh → `Vertiport`; bãi đỗ → `Pad`; waypoint → `FlightPath`.

## 3. Kết quả chính (N = 30, 1500 trial, max_wait 600s)

| Pha | Baseline | Resolved | Giảm |
|---|---:|---:|---:|
| Cất cánh (C2) | 8.01% | **0.00%** | 100% |
| Hạ cánh (C3) | 8.04% | **0.00%** | 100% |
| Trên hành trình (C1) | 5.49% | **0.00%** | 100% |
| Bãi đỗ hạ cánh (C4) | 12.62% | **0.00%** | 100% |
| **Toàn kế hoạch bay** | **20.42%** | **0.00%** | **100%** |

→ **Xác suất xung đột sau điều phối = 0** ở cả bốn pha, đúng như kỳ vọng. Chi
phí: **từ chối 0%**, **trễ trung bình 1.9s** (p95 9–11s).

## 4. Xung đột = 0 ở mọi mật độ

| N | Baseline any | Resolved any | Từ chối | Trễ TB |
|--:|--:|--:|--:|--:|
| 10 | 6.73% | **0.00%** | 0% | 0.6s |
| 20 | 15.15% | **0.00%** | 0% | 1.4s |
| 30 | 20.82% | **0.00%** | 0% | 1.9s |
| 40 | 27.21% | **0.00%** | 0% | 2.6s |
| 50 | 32.44% | **0.00%** | 0% | 3.5s |
| 60 | 37.64% | **0.00%** | 0% | 4.1s |

Baseline tăng tới ~38% theo mật độ, còn **resolved luôn = 0**; ở `max_wait=600s`
không chuyến nào bị từ chối, chỉ mất vài giây trễ trung bình.

## 5. Đánh đổi từ chối ↔ độ trễ (mệnh đề max_wait)

Tại N = 60, quét `max_wait` (xung đột **luôn = 0**):

| max_wait | Từ chối | Trễ TB | P(xung đột) |
|--:|--:|--:|--:|
| 15s | 7.81% | 0.4s | 0.00% |
| 30s | 4.56% | 1.2s | 0.00% |
| 60s | 1.46% | 2.7s | 0.00% |
| 120s | 0.10% | 3.9s | 0.00% |
| 300s | 0.00% | 4.1s | 0.00% |
| 600s | 0.00% | 4.1s | 0.00% |

`max_wait` càng chặt → càng nhiều chuyến bị từ chối (đổi lấy độ trễ nhỏ hơn cho
các chuyến được duyệt), nhưng **xác suất xung đột luôn = 0** và chuyến bị từ
chối **không tính vi phạm**. Đây là đánh đổi **thông lượng ↔ độ trễ** ở mức an
toàn tuyệt đối. Xem `resolver_maxwait.png`.

## 6. Chạy lại

```bash
# Cần checkout repo time-departure và trỏ --resolver-path tới nó
python compare_resolver.py --resolver-path /path/to/time-departure \
    --trials 1500 --flights 30 --max-wait 600 \
    --sweep 10 20 30 40 50 60 --maxwait-sweep 15 30 60 120 300 600 --maxwait-N 60
```

> Script nạp `scrp_simple` theo đường dẫn (không sao chép mã của repo kia).

## 7. Kết luận

Với đúng mô hình của scrp_simple (ràng buộc bề mặt pad = hạ cánh; khoảng cách an
toàn 50m), thuật toán time-departure **triệt tiêu hoàn toàn xung đột** (0% ở cả
bốn pha, mọi mật độ khảo sát) chỉ với **vài giây trễ trung bình**. Khi ép độ trễ
tối đa nhỏ, một phần chuyến bị từ chối — được xem là **không vi phạm** theo đúng
quy ước — đổi lấy độ trễ thấp hơn cho phần còn lại.
