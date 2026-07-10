# Đối chứng: baseline vs thuật toán time-departure

So sánh **xác suất xung đột trước và sau khi điều phối**, dùng bộ giải xung đột
`scrp_simple.ConflictResolver` từ repo
[`hunglq-vht/time-departure`](https://github.com/hunglq-vht/time-departure).

Script: [`../compare_resolver.py`](../compare_resolver.py) · Hình:
[`baseline_vs_resolved.png`](../outputs/baseline_vs_resolved.png),
[`resolver_sweep.png`](../outputs/resolver_sweep.png) · Số liệu:
[`compare_results.json`](../outputs/compare_results.json).

## 1. Phương pháp

Mỗi trial: sinh **cùng một** bộ N kế hoạch bay (như baseline), rồi

- **BASELINE** — đo xung đột trực tiếp, không điều phối.
- **RESOLVED** — cho `resolve_batch` duyệt tuần tự theo thứ tự thời gian: mỗi
  yêu cầu có thể bị **dời giờ xuất phát** (delay), được **gán bãi đỗ hạ cánh**,
  hoặc **bị từ chối** nếu delay > `max_wait`. Sau đó đo lại xung đột trên tập
  đã duyệt bằng **đúng bộ phát hiện của baseline**.

**Ánh xạ mô hình** (plan-sim → resolver): điểm cất/hạ cánh → `Vertiport`; bãi đỗ
→ `Pad` với `occupation_duration = pad.max_dwell`; waypoint → `FlightPath`;
ràng buộc **C1↔hành trình, C2↔cất cánh, C3↔hạ cánh, C4↔bãi đỗ (hạ cánh)**.

Hai điều chỉnh để so sánh **công bằng** (áp dụng đồng nhất cho cả hai phía):

1. **Dung sai biên** `EPS_T = 1ms`, `EPS_M = 1cm`: resolver xếp các cửa sổ sát
   nhau đúng biên, sai số dấu phẩy động tạo "chồng lấn" ~nano-giây — không tính.
2. **Chiếm bãi đỗ = `pad.max_dwell`**: đúng bằng lượng resolver đặt chỗ (thay vì
   dwell ngẫu nhiên gắn với bãi đỗ gốc trước khi bị gán lại).

## 2. Kết quả chính (N = 30, 1500 trial, max_wait 600s)

| Pha | Baseline | Resolved | Giảm |
|---|---:|---:|---:|
| Cất cánh | 8.01% | **0.00%** | 100% |
| Hạ cánh | 8.04% | **0.00%** | 100% |
| Trên hành trình | 6.17% | 0.11% | 98.3% |
| Bãi đỗ (tổng) | 21.23% | 12.48% | 41.2% |
| &nbsp;&nbsp;├ bãi đỗ **hạ cánh** | 12.62% | **0.00%** | 100% |
| &nbsp;&nbsp;└ bãi đỗ **cất cánh** | 12.49% | 12.48% | ~0% |
| **Toàn kế hoạch bay** | **26.50%** | **12.54%** | **52.7%** |

Chi phí: **từ chối 0%**, **trễ trung bình 1.9s** (p95 10.6s, max 183s).

## 3. Ba phát hiện

**(1) Xoá sạch 3 pha nó quản lý.** Tuần tự hoá vùng trời cất cánh (C2), hạ cánh
(C3) và gán bãi đỗ hạ cánh (C4) đưa ba pha này về **đúng 0%** — resolver hoạt
động chính xác.

**(2) Hành trình: resolver hard-code 50m, bỏ qua bán kính an toàn theo loại.**
Phần dư 0.11% KHÔNG phải lỗi: mọi trường hợp dư đều có khoảng cách nhỏ nhất
`dmin ≥ 50m` (quan sát 55–73m) — resolver **giữ đúng `MIN_SEPARATION_M = 50`
như cam kết**. Nhưng mô hình của ta có drone cần tới **95m** (`max(r_i, r_j)`),
nên với các loại này vẫn còn xung đột. → Muốn triệt để, resolver cần dùng
khoảng cách an toàn **theo cặp loại drone** thay vì hằng số 50m.

**(3) Bãi đỗ CẤT CÁNH không được quản lý.** `scrp_simple` chỉ gán bãi đỗ **hạ
cánh** (C4); nó không mô hình hoá việc chiếm **bãi đỗ cất cánh** trước khi đi.
Vì vậy pha này **gần như không đổi** (12.49% → 12.48%) và trở thành **phần dư
chi phối** của "toàn KHB" sau điều phối. Nếu quản lý luôn bãi đỗ cất cánh
(hàng đợi/gán bãi tương tự C4), "toàn KHB" resolved sẽ tụt từ 12.5% xuống
còn ~0.1% (chỉ còn khe hở 50m ở mục (2)).

## 4. Theo mật độ giao thông

| N | Baseline any | Resolved any | Từ chối | Trễ TB |
|--:|--:|--:|--:|--:|
| 10 | 8.80% | 3.87% | 0% | 0.6s |
| 20 | 19.50% | 8.31% | 0% | 1.4s |
| 30 | 26.91% | 12.60% | 0% | 1.9s |
| 40 | 34.28% | 15.98% | 0% | 2.6s |
| 50 | 41.19% | 20.14% | 0% | 3.5s |
| 60 | 46.75% | 23.50% | 0% | 4.1s |

Ở mọi mật độ khảo sát, resolver **cắt hơn một nửa** xác suất xung đột chỉ với
**vài giây trễ trung bình** và **không từ chối** chuyến nào (delay luôn dưới
`max_wait = 600s`). Đường "resolved" gần như trùng với phần dư **bãi đỗ cất
cánh** — khẳng định lại phát hiện (3): đó là thành phần còn lại đáng kể duy nhất.

## 5. Chạy lại

```bash
# Cần checkout repo time-departure và trỏ --resolver-path tới nó
python compare_resolver.py --resolver-path /path/to/time-departure \
    --trials 1500 --flights 30 --max-wait 600 --sweep 10 20 30 40 50 60
```

> Script nạp `scrp_simple` theo đường dẫn (không sao chép mã của repo kia vào
> đây). Không có repo `time-departure`, script sẽ báo lỗi và hướng dẫn checkout.

## 6. Kết luận

Thuật toán time-departure là **lời giải hiệu quả cho phần lớn xung đột** với chi
phí trễ rất thấp ở dải mật độ khảo sát. Hai hạng mục nên bổ sung để đạt ~0%:
(a) dùng **khoảng cách an toàn theo loại drone** thay cho hằng số 50m ở C1;
(b) **quản lý bãi đỗ cất cánh** tương tự bãi đỗ hạ cánh. Bộ Monte Carlo này là
thước đo sẵn sàng để đánh giá lại sau mỗi cải tiến đó.
