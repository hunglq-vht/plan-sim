# Mô hình giải tích & phân tích giới hạn xác suất xung đột

Tài liệu này tổng quát hóa thí nghiệm Monte Carlo (xem [`../README.md`](../README.md))
thành **công thức đóng** và dùng **các giới hạn** để chỉ ra xác suất xung đột.
Toàn bộ công thức đã được đối chiếu với mô phỏng — xem
[`../outputs/analytic_vs_mc.png`](../outputs/analytic_vs_mc.png), sinh bởi
[`../make_theory_plot.py`](../make_theory_plot.py).

Ý tưởng cốt lõi: mọi pha xung đột đều quy về một bài toán **"trùng lịch trong
một tài nguyên dùng chung"**, nên toàn hệ rút gọn về **một tham số vô thứ
nguyên** — tải giao thông `ρ` — và **một hàm phổ quát** `P = 1 − e^(−ρ)`.

---

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `N` | số kế hoạch bay (KHB) trong một trial |
| `T` | horizon (s): cửa sổ thời gian rải các sự kiện |
| `R` | số đường bay |
| `P` | số bãi đỗ mỗi điểm cất/hạ cánh |
| `μ_to, μ_la` | thời gian cất / hạ cánh trung bình |
| `μ_d` | thời gian dừng bãi đỗ trung bình |
| `s̄` | khoảng cách an toàn hiệu dụng `= E[max(r_i, r_j)]` |
| `v̄` | vận tốc hành trình trung bình |

Bốn pha: **takeoff** (cất cánh), **landing** (hạ cánh), **enroute** (trên hành
trình), **parking** (bãi đỗ); và **any** = toàn kế hoạch bay.

---

## 2. Bổ đề cốt lõi — xác suất trùng thời gian và giới hạn `1/T`

Mọi pha đều quy về câu hỏi: *hai sự kiện "bận" trong cùng một tài nguyên có
chồng lấn thời gian không?* Hai sự kiện bắt đầu tại `S₁, S₂ ~ iid Unif[0, T]`,
thời lượng bận `τ₁, τ₂`. Chúng chồng lấn khi và chỉ khi

```
S₁ − S₂ ∈ (−τ₂, τ₁).
```

Đặt `D = S₁ − S₂`. Hiệu của hai biến đều nhau có **mật độ tam giác**

```
f_D(d) = (T − |d|) / T² ,   |d| ≤ T.
```

Do đó

```
Pr(overlap | τ₁, τ₂) = ∫_{−τ₂}^{τ₁} (T − |d|)/T² dd
                     = (τ₁ + τ₂)/T − (τ₁² + τ₂²)/(2T²).
```

Lấy kỳ vọng theo thời lượng rồi **cho `T → ∞`**:

```
lim_{T→∞}  T · Pr(overlap) = E[τ₁] + E[τ₂] = 2μ
⇒  Pr(overlap) = 2μ/T + O(T⁻²).                          (★)  ĐỊNH LUẬT 1/T
```

Chồng lấn càng hiếm khi horizon càng rộng. Nếu tài nguyên có **bội `K`** (hai sự
kiện phải rơi vào cùng 1 trong `K` tài nguyên rời rạc), nhân thêm
`Pr(cùng tài nguyên) = 1/K`.

---

## 3. Xác suất xung đột mỗi CẶP theo pha: `q_p = a_p / T`

Áp bổ đề (★) với tài nguyên tương ứng:

- **Cất cánh / Hạ cánh** — tài nguyên là *điểm cất/hạ cánh*, bội `K = R`:
  ```
  q_to = (1/R) · (2 μ_to / T)
  q_la = (1/R) · (2 μ_la / T)
  ```
- **Bãi đỗ** — tài nguyên là *một bãi đỗ cụ thể*, bội `K = R·P`; mỗi chuyến chiếm
  hai bãi (đi + đến) nên có hệ số 2:
  ```
  q_pk = (2 / (R·P)) · (2 μ_d / T)
  ```
- **Trên hành trình** — quy khoảng cách an toàn `s̄` thành *headway thời gian*
  `s̄/v̄` (thời gian bay hết khoảng an toàn); hai drone cùng đường bay xung đột
  khi đi qua một điểm mốc cách nhau dưới headway đó:
  ```
  q_en = (c / R) · (2 s̄ / (v̄ · T))
  ```

Viết gọn `q_p = a_p / T` với **thời gian tương tác** `a_p` (không phụ thuộc `T`).
Giá trị tính trực tiếp từ tham số mô hình (`R=10`, `T=1800s`):

| Pha | `a_p = q_p·T` |
|---|---:|
| Cất cánh | 5.5 s |
| Hạ cánh | 5.5 s |
| Bãi đỗ | 12.3 s |
| Trên hành trình | ≈ 4.0 s |

> `c ≈ 3.5` là hằng số `O(1)` của pha hành trình: bằng **1** nếu mọi drone cùng
> vận tốc (khoảng cách dọc tuyến giữ nguyên ⇒ điều kiện xung đột đúng là
> `|ΔA| < s̄/v̄`), và **lớn hơn 1** do chênh lệch vận tốc cho phép "vượt", làm
> rộng cửa sổ bắt gặp. Đây là pha duy nhất không có công thức đóng hoàn toàn.

---

## 4. Từ cặp sang MỖI kế hoạch bay

Một chuyến dính xung đột ở pha `p` nếu trùng với `≥ 1` trong `N − 1` chuyến còn
lại. Số "đối tác" tuân theo phân phối Nhị thức, hội tụ **Poisson** (biến cố
hiếm):

```
P_p(N) = 1 − (1 − q_p)^(N−1)  ──→  1 − exp(−(N−1) q_p) = 1 − exp(−(N−1) a_p / T).   (♦)
```

**Đối chiếu công thức (♦) với mô phỏng** (2000 trial/điểm):

| `N` | Cất cánh (CT / MC) | Hạ cánh | Hành trình | Bãi đỗ |
|--:|--|--|--|--|
| 30 | 8.48% / 8.18% | 8.48% / 8.17% | 6.31% / 6.07% | 17.96% / 14.52% |
| 60 | 16.50% / 15.57% | 16.50% / 15.66% | 12.42% / 12.11% | 33.15% / 27.51% |

Ba pha đầu khớp trong ~5%. Pha bãi đỗ hơi cao vì horizon hiệu dụng thực tế
`T_eff > T` (thời điểm hạ cánh trải rộng hơn thời điểm tạo do cộng thêm độ trễ
xuất phát và thời gian bay), làm mẫu số trong (★) lớn hơn `T`.

---

## 5. Toàn kế hoạch bay — vì sao KHÔNG cộng được

Nếu 4 pha độc lập thì `P_any = 1 − Π_p (1 − P_p)`. Nhưng thực tế **các pha tương
quan dương mạnh**: một cặp "nguy hiểm" (cùng đường bay + xuất phát sát nhau)
thường gây xung đột ở NHIỀU pha cùng lúc. Vì vậy chỉ có **chặn**
(Fréchet–Bonferroni):

```
max_p P_p  ≤  P_any  ≤  1 − Π_p (1 − P_p)  ≤  Σ_p P_p.
```

Số liệu tại `N = 30`:

```
max_p P_p = 17.96%  ≤  P_any(MC) = 20.5%  ≤  35.6% (cận trên độc lập)  ≤  41% (tổng).
```

Giá trị thật nằm **sát cận dưới** — đúng như dự đoán về tương quan. Trên hình
`analytic_vs_mc.png`, các điểm vuông "Toàn KHB (MC)" nằm thấp hơn hẳn đường đứt
"cận trên độc lập".

---

## 6. Phân tích giới hạn

Đặt **tải** vô thứ nguyên `ρ_p = (N−1) a_p / T`. Khi đó `P_p = 1 − e^(−ρ_p)`.

**(a) Mật độ thấp / horizon rộng** (`ρ_p → 0`): khai triển `e^(−x) ≈ x`
```
P_p ≈ (N−1) a_p / T   ⇒  TUYẾN TÍNH theo N,  tỉ lệ 1/T,  lim_{T→∞} P_p = 0.
```
Đây là lý do các đường quét mật độ gần thẳng ở `N` nhỏ; độ dốc `= a_p / T`.

**(b) Mật độ cao** (`N → ∞`, `T` cố định):
```
P_p(N) = 1 − e^(−(N−1)a_p/T) → 1,   với  1 − P_p ~ e^(−(N−1)a_p/T)  (tiến 1 theo mũ).
```
Xung đột trở thành **gần như chắc chắn**.

**(c) Kỳ vọng SỐ xung đột** (không bão hòa ở 1):
```
E[C_p] = C(N,2) · q_p = [N(N−1)/2] · a_p/T  ~  (a_p/2) · N²/T   (BẬC HAI theo N).
```
Xác suất mỗi-chuyến bão hòa về 1, nhưng **tổng số va chạm** vẫn tăng `∝ N²`.

**(d) Giới hạn nhiệt động / continuum** — dạng phát biểu sạch nhất. Cho
`N → ∞, T → ∞` với **cường độ đến `λ = N/T` cố định**: quá trình tạo KHB hội tụ
về **Poisson(λ)**, số đối tác kỳ vọng của một chuyến `= λ a_p` hữu hạn, nên

```
lim_{N,T→∞, N/T=λ}  P_p = 1 − e^(−λ a_p).
```

Xác suất chỉ phụ thuộc **cường độ giao thông `λ`** và **thời gian tương tác
`a_p`**, không phụ thuộc riêng `N, T` — bất biến co giãn của hệ.

**(e) Mật độ tới hạn.** Đặt `P_p = 1/2` ⇒ `(N* − 1) a_p / T = ln 2`, tức
```
N*_p ≈ 1 + (ln 2 · T) / a_p.
```
Ví dụ pha bãi đỗ `a_pk = 12.3s`, `T = 1800s` ⇒ `N* ≈ 102` chuyến thì một nửa số
chuyến dính xung đột bãi đỗ.

---

## 7. Tổng kết

| Chế độ | Điều kiện | Xác suất |
|---|---|---|
| Tuyến tính | `ρ_p ≪ 1` | `P_p ≈ ρ_p = (N−1)a_p/T` |
| Tổng quát | mọi `ρ_p` | `P_p = 1 − e^(−ρ_p)` |
| Bão hòa | `ρ_p ≫ 1` | `P_p → 1`, phần dư `~ e^(−ρ_p)` |
| Continuum | `N/T = λ` | `P_p = 1 − e^(−λ a_p)` |
| Tới hạn | `P_p = 1/2` | `N* ≈ 1 + ln2·T/a_p` |

Toàn bộ bài toán rút về một tham số **`ρ_p = (N−1)a_p/T`** và hàm phổ quát
**`P_p = 1 − e^(−ρ_p)`**, với ba chế độ: tuyến tính (`ρ→0`), bão hòa về 1
(`ρ→∞`), và điểm tới hạn tại `ρ = ln 2`. Các hằng số `a_p` gói toàn bộ hình học
và tài nguyên: `a ∝ (thời lượng bận) / (số tài nguyên)`.

### Tái tạo số liệu
```bash
python make_theory_plot.py      # sinh outputs/analytic_vs_mc.png + in a_p
```
