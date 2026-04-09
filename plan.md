# GrokAdamW Kalan Optimizasyon Planı

10 optimizasyonun detaylı uygulama planı. Her biri için: hedef dosya/konum, mevcut kod, değişiklik, riskler ve test stratejisi.

---

## Öncelik Sırası

```
O14 (CUDA Graphs)     → Düşük karmaşıklık, devasa etki
O9  (Async CPU)        → Düşük-orta karmaşıklık, %20-40 offload
O8  (autotune)         → Düşük karmaşıklık, %5-15 Triton
O2  (fallback buf)     → Orta karmaşıklık, %25 VRAM
O5  (kernel fusion)    → Orta-yüksek karmaşıklık, %15-25 Triton
O6  (atomic contention)→ Orta karmaşıklık, büyük parametreler
O10 (grad buffer)      → Düşük karmaşıklık, küçük etki
O3  (bf16 states)      → Orta karmaşıklık, %50 VRAM (riskli)
O15 (flat-param)       → Çok yüksek karmaşıklık, %30-50
O4  (8-bit quant)      → Çok yüksek karmaşıklık, %75 VRAM
```

---

## O14: CUDA Graphs Uyumluluğu

### Neden
`_triton_fused.py:179,210-212`'de her adımda `torch.zeros(2, ...)` ile yeni tensör allocate ediliyor. CUDA Graphs, her adımda aynı memory adreslerini kullanılmasını gerektirir. Dinamik allocation, graph capture'ı engeller.

### Mevcut Kod
```python
# _triton_fused.py:179 (grokadamw_fused_step)
norms_buf = torch.zeros(2, device=p.device, dtype=torch.float32)

# _triton_fused.py:210-212 (grokadamw_fused_group)
norms_bufs = [
    torch.zeros(2, device=p.device, dtype=torch.float32)
    for p, _, _, _, _, _ in params_grads_states
]
```

### Yapılacak Değişiklik

**Adım 1:** `grokadamw.py` state init'e `norms_buf` ekle:
```python
# grokadamw.py state init (~line 188-201)
state.update({
    "step": 0,
    "exp_avg": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
    "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
    "grok_ema": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
    "norms_buf": torch.zeros(2, dtype=torch.float32, device=p.device),
})
```

**Adım 2:** `_triton_fused.py` fonksiyonlarını güncelle:
```python
# grokadamw_fused_step: norms_buf parametre olarak al
def grokadamw_fused_step(p, grad, exp_avg, exp_avg_sq, grok_ema,
                         alpha, lamb, layer_beta1, beta2, step_size,
                         lr_wd, eps, norms_buf, grad_norm=None):
    # norms_buf.zero_() ile sıfırla, yeniden allocate etme
    norms_buf.zero_()
    _launch_norms(grad, grok_ema, norms_buf, alpha, lamb)
    _launch_update(...)

# grokadamw_fused_group: norms_bufs parametre olarak al
def grokadamw_fused_group(params_grads_states, alpha, lamb, beta2, eps, norms_bufs):
    for i, buf in enumerate(norms_bufs):
        buf.zero_()
    # ... aynı loop ama norms_bufs dışarıdan geliyor
```

**Adım 3:** `grokadamw.py` çağrı yerlerini güncelle:
```python
# _update_group içinde, triton_batch oluştururken norms_buf'ı da ekle
# grokadamw_fused_group çağrısında norms_bufs listesini geçir
norms_bufs = [self.state[p]["norms_buf"] for p, _, _, _, _, _ in triton_batch]
grokadamw_fused_group(triton_batch, alpha, group["lamb"], beta2, group["eps"], norms_bufs)
```

**Adım 4:** `state_dict`/`load_state_dict`'e norms_buf ekle:
```python
# state_dict: norms_buf kaydetmeye gerek yok (küçük, recreate edilir)
# load_state_dict: norms_buf yoksa oluştur
if "norms_buf" not in state:
    state["norms_buf"] = torch.zeros(2, dtype=torch.float32, device=p.device)
```

### Riskler
- `cpu_offload=True` durumunda `norms_buf` GPU'da olmalı, CPU'da değil
- `load_state_dict` ile eski checkpoint yüklendiğinde norms_buf olmayacak → migration gerekli

### Test
- CUDA Graphs ile capture/playback testi
- Eski checkpoint'ten `load_state_dict` testi

---

## O9: Async CPU↔GPU Transferleri (cpu_offload)

### Neden
`grokadamw.py:207-210` senkron `.to(p.device)` yapıyor. Her parametre için GPU'ya transfer tamamlandıktan sonra ilerleniyor. Bu, GPU'yu bekletir.

### Mevcut Kod
```python
# grokadamw.py:207-210
if cpu_offload:
    exp_avg = exp_avg.to(p.device)        # senkron, GPU bekler
    exp_avg_sq = exp_avg_sq.to(p.device)  # senkron
    grok_ema = grok_ema.to(p.device)      # senkron
```

### Yapılacak Değişiklik

**Adım 1:** State init'te pinned memory kullan:
```python
# grokadamw.py state init
if cpu_offload:
    state.update({
        "step": 0,
        "exp_avg": torch.zeros(p.shape, dtype=torch.float32).pin_memory(),
        "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32).pin_memory(),
        "grok_ema": torch.zeros(p.shape, dtype=torch.float32).pin_memory(),
    })
else:
    # mevcut GPU alloc...
```

**Adım 2:** Non-blocking transfer:
```python
# grokadamw.py:207-210
if cpu_offload:
    exp_avg = exp_avg.to(p.device, non_blocking=True)
    exp_avg_sq = exp_avg_sq.to(p.device, non_blocking=True)
    grok_ema = grok_ema.to(p.device, non_blocking=True)
    # İlk kernel launch'tan önce sync gerekli:
    torch.cuda.current_stream().synchronize()
```

**Adım 3:** Double-buffering (opsiyonel, ileri seviye):
```python
# Param i compute ederken param i+1 transfer et
for i in range(len(params)):
    # Transfer i+1
    if i + 1 < len(params):
        state_next = self.state[params[i+1]]
        exp_avg_next = state_next["exp_avg"].to(p.device, non_blocking=True)
        # ...

    # Compute i (mevcut param)
    _foreach_update(...)

    # Transfer i geri CPU'ya
    state["exp_avg"] = exp_avg.to("cpu", non_blocking=True)
    # ...
```

### Riskler
- `pin_memory()` CPU RAM'de page-locked memory ayırır, büyük modellerde sistem RAM baskısı
- Double-buffering kod karmaşıklığını artırır
- Non-blocking transfer + sync noktası yanlış yerleşirse race condition

### Test
- `test_cpu_offload_true_states_on_cpu` (mevcut)
- Numerical equivalence: cpu_offload=True vs False karşılaştırması
- Large model ile RAM kullanım testi

---

## O8: @triton.autotune

### Neden
Mevcut BLOCK_SIZE/num_warps/num_stages heuristikleri statik. Ampere (A100) vs Hopper (H100) vs Ada (RTX 4099) farklı optimal değerler ister.

### Mevcut Kod
```python
# _triton_fused.py:90-92
bs = min(4096, triton.next_power_of_2(n))
nw = 8 if n > 65536 else (4 if n > 4096 else 2)
ns = 3 if n > 16384 else 2
```

### Yapılacak Değişiklik

**Adım 1:** Her iki kernel'e `@triton.autotune` decorator ekle:
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _norms_kernel(grad_ptr, grok_ema_ptr, ..., n_elements, BLOCK_SIZE: tl.constexpr):
    # mevcut kernel kodu aynı
```

**Adım 2:** `_launch_norms` ve `_launch_update` fonksiyonlarını basitleştir:
```python
def _launch_norms(grad, grok_ema, norms_buf, alpha, lamb):
    n = grad.numel()
    if n == 0:
        return
    grid = ((n + 4096 - 1) // 4096,)  # upper bound, autotune override eder
    norms_buf.zero_()
    _norms_kernel[grid](grad, grok_ema, norms_buf[:1], norms_buf[1:],
                        alpha, lamb, n)
    # BLOCK_SIZE, num_warps, num_stages autotune tarafından belirlenir
```

### Riskler
- İlk çalıştırmada tuning süresi uzun (her config denenir) → warmup süresi artar
- Derlenmiş kernel cache yoksa her işlem başında tuning tekrarlanır
- `key=["n_elements"]` ile sadece element sayısına göre tune edilir; data pattern'i etkisi yok

### Test
- Benchmark karşılaştırma: autotune vs mevcut heuristic
- Farklı param boyutlarında (1K, 10K, 100K, 1M, 10M element) test

---

## O2: Fallback Path grok_grad_buf Kaldırma

### Neden
`grok_grad_buf` her adımda sıfırlanıyor ve yeniden dolduruluyor. Persistent buffer yerine inline hesaplama ile %25 state VRAM tasarrufu.

### Mevcut Kod
```python
# grokadamw.py _foreach_update (lines 35-56)
def _foreach_update(params, grads, exp_avgs, exp_avg_sqs, grok_emas,
                    grok_grad_bufs, alpha, lamb, ...):
    grad_norms = torch._foreach_norm(grads)

    torch._foreach_mul_(grok_emas, alpha)
    torch._foreach_add_(grok_emas, grads, alpha=1 - alpha)
    torch._foreach_mul_(params, 1 - lr_wd)
    torch._foreach_copy_(grok_grad_bufs, grads)       # ← grok_grad_buf kullanılıyor
    torch._foreach_add_(grok_grad_bufs, grok_emas, alpha=lamb)

    grok_grad_norms = torch._foreach_norm(grok_grad_bufs)  # ← batched norm
    scales = [gn / ggn.clamp(min=eps) ...]
    torch._foreach_mul_(grok_grad_bufs, scales)

    for i in range(n):
        grok_grad = grok_grad_bufs[i]   # ← grok_grad_buf kullanılıyor
        exp_avgs[i].mul_(...).add_(grok_grad, ...)
        ...
```

### Yapılacak Değişiklik

**Adım 1:** `_foreach_update` imzasını değiştir, `grok_grad_bufs` parametresini kaldır:
```python
def _foreach_update(params, grads, exp_avgs, exp_avg_sqs, grok_emas,
                    alpha, lamb, layer_beta1_list, beta2,
                    step_size_list, lr_wd, eps):
    n = len(params)
    grad_norms = torch._foreach_norm(grads)

    torch._foreach_mul_(grok_emas, alpha)
    torch._foreach_add_(grok_emas, grads, alpha=1 - alpha)
    torch._foreach_mul_(params, 1 - lr_wd)

    # grok_grad'ı inline hesapla, persistent buffer yok
    for i in range(n):
        grok_grad = grads[i] + lamb * grok_emas[i]
        grok_grad_norm = grok_grad.norm()
        grad_norm = grad_norms[i]
        scale = grad_norm / grok_grad_norm.clamp(min=eps)
        grok_grad = grok_grad * scale

        exp_avgs[i].mul_(layer_beta1_list[i]).add_(grok_grad, alpha=1 - layer_beta1_list[i])
        exp_avg_sqs[i].mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)
        denom = exp_avg_sqs[i].sqrt().add_(eps)
        params[i].addcdiv_(exp_avgs[i], denom, value=-step_size_list[i])
```

**Adım 2:** State allocation'dan `grok_grad_buf`'ı tamamen kaldır:
```python
# grokadamw.py state init — grok_grad_buf yok
state.update({
    "step": 0,
    "exp_avg": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
    "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
    "grok_ema": torch.zeros(p.shape, dtype=torch.float32, device=state_device),
})
```

**Adım 3:** cpu_offload geri yazma'dan grok_grad_buf kaldır:
```python
# grokadamw.py:292-304
if cpu_offload:
    for p, exp_avg, exp_avg_sq, grok_ema in zip(
        active_params, active_exp_avg, active_exp_avg_sq, active_grok_ema
    ):
        state = self.state[p]
        state["exp_avg"] = exp_avg.to("cpu")
        state["exp_avg_sq"] = exp_avg_sq.to("cpu")
        state["grok_ema"] = grok_ema.to("cpu")
```

**Adım 4:** `state_dict`/`load_state_dict`'den grok_grad_buf referanslarını temizle.

### Riskler
- Per-param `.norm()` hesabı, `_foreach_norm` batched hesabından daha yavaş olabilir
- grok_grad inline hesaplanınca memory'de 2× grad büyüklüğünde geçici tensör oluşur (PyTorch caching allocator ile yönetilir)
- Numerical equivalence testi kritik — sonuçlar bayt-bayt aynı olmalı

### Test
- `test_fp32_cpu_vs_gpu_states` (mevcut, atol=1e-5)
- `test_bf16_cpu_vs_gpu_states`
- `test_save_load_roundtrip` — eski checkpoint ile uyumluluk
- Benchmark: per-param norm vs batched norm performans karşılaştırması

---

## O5: _norms_kernel + _update_kernel Füzyonu

### Neden
İki kernel arasında `grok_grad = grad + lamb * grok_ema` iki kez hesaplanıyor (_triton_fused.py:33 ve :76). Tek kernel'de register'da bir kez hesaplanabilir. Ayrıca grad ve grok_ema iki kez global memory'den okunuyor (7R+4W=11 → 5R+4W=9).

### Mevcut Mimari
```
_norms_kernel (param başına 1 launch):
  1. grad, grok_ema OKU
  2. new_ema HESAPLA → grok_ema YAZ
  3. grok_grad HESAPLA (sadece norm için)
  4. grad_norm², grok_grad_norm² → atomic_add

_update_kernel (param başına 1 launch):
  1. grad_norm², grok_grad_norm² OKU
  2. grad, grok_ema TEKRAR OKU
  3. grok_grad TEKRAR HESAPLA
  4. exp_avg, exp_avg_sq, p GÜNCELLE
```

### Yapılacak Değişiklik

**Yaklaşım 1: İki-pass, tek kernel (önerilen)**
```python
@triton.jit
def _fused_kernel_phase1(
    grad_ptr, grok_ema_ptr, scratch_grad_sq_ptr, scratch_grok_sq_ptr,
    alpha, lamb, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Mevcut _norms_kernel ile aynı
    # grok_ema güncelle, norm'ları scratch'e yaz
    pass

@triton.jit
def _fused_kernel_phase2(
    p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, grok_ema_ptr,
    scratch_grad_sq_ptr, scratch_grok_sq_ptr,
    lamb, layer_beta1, beta2, step_size, lr_wd, eps, eps_norm,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Mevcut _update_kernel ile aynı
    # grok_ema'yı TEKRAR OKUMAK yerine phase1'den gelen güncel değeri kullan
    # (zaten phase1 yazdı, aynı memory adresi)
    pass
```

Bu yaklaşımda aslında iki kernel kalır ama aradaki fark:
- Phase 1 ve 2 art arda aynı param için çalışır (mevcut: tüm phase1 → tüm phase2)
- Bu da grok_ema'nın cache'te kalma olasılığını artırır

**Yaklaşım 2: Gerçek tek-kernel füzyonu (Triton 3.0+ num_ctas)**
```python
@triton.jit
def _single_fused_kernel(
    ..., n_elements, BLOCK_SIZE: tl.constexpr, NUM_CTAS: tl.constexpr
):
    pid = tl.program_id(0)
    cta_id = tl.program_id(1)  # 0 = norms pass, 1 = update pass

    if cta_id == 0:
        # norms hesapla, atomic_add ile reduction
        ...
        tl.debug_barrier()  # cross-CTA barrier (NUM_CTAS > 1 gerekli)
    else:
        # update hesapla
        ...
```

**Sorun:** `tl.debug_barrier()` cross-CTA barrier değildir. Sadece aynı CTA içindeki thread'leri senkronize eder. Gerçek cross-block barrier için Triton'da resmi destek yoktur (Henüz). `num_ctas` deneysel bir özelliktir.

**Yaklaşım 3: Per-block partial norm + inline reduction**
```python
@triton.jit
def _fully_fused_kernel(
    p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, grok_ema_ptr,
    workspace_ptr,  # [num_blocks * 2] boyutunda
    alpha, lamb, layer_beta1, beta2, step_size, lr_wd, eps,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Pass 1: grok_ema güncelle + partial norm
    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grok_ema = tl.load(grok_ema_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    new_ema = alpha * grok_ema + (1.0 - alpha) * grad
    grok_grad = grad + lamb * new_ema

    tl.store(grok_ema_ptr + offs, new_ema, mask=mask)

    g_sq = tl.where(mask, grad * grad, 0.0)
    gg_sq = tl.where(mask, grok_grad * grok_grad, 0.0)
    tl.store(workspace_ptr + pid, tl.sum(g_sq))          # per-block slot
    tl.store(workspace_ptr + num_blocks + pid, tl.sum(gg_sq))

    # SORUN: Burada tüm block'ların tamamlanmasını beklemek gerekiyor
    # Triton'da cross-block barrier yok
    # Çözüm: Bu kernel'ı iki ayrı grid olarak çalıştır
```

### Önerilen Son Mimari (pragmatik)
```
Yaklaşım 1'i uygula:
- Kernel kodu AYNI kalır (iki ayrı kernel)
- Launch stratejisini değiştir: param-by-param pair launch

for i in range(n_params):
    _launch_norms(param_i, ...)    # norms kernel
    _launch_update(param_i, ...)   # update kernel (hemen ardından)
    # Böylece param_i'nin grok_ema'sı L2 cache'de hâlâ sıcak

# Eski: Tüm norms → Tüm updates
# Yeni: (norms_0, update_0), (norms_1, update_1), ...
```

Bu yaklaşım kernel sayısını değiştirmez ama cache locality'yi önemli ölçüde artırır.

### Riskler
- Gerçek füzyon Triton'da güvenilir şekilde yapılamaz (cross-block barrier eksik)
- Yaklaşım 1 (pair launch) cache locality iyileştirmesi %5-10 civarı
- Yaklaşım 2/3 Triton versiyon bağımlı, taşınabilir değil

### Test
- Numerical equivalence: pair launch vs mevcut two-phase launch
- Benchmark: cache miss sayısı (ncu profiler ile)

---

## O6: atomic_add Contention Azaltma

### Neden
4096×4096 matris = 16M element, BLOCK_SIZE=4096 ile 4096 block. Her block `tl.atomic_add` ile aynı 2 skaler adrese yazıyor. 4096 atomik işlem serileşiyor.

### Mevcut Kod
```python
# _triton_fused.py:39-40
tl.atomic_add(out_grad_sq_ptr, tl.sum(g_sq))    # tüm blocklar aynı adrese
tl.atomic_add(out_grok_sq_ptr, tl.sum(gg_sq))   # tüm blocklar aynı adrese
```

### Yapılacak Değişiklik

**Yaklaşım 1: Per-block workspace + CPU reduction**
```python
# Kernel: her block kendi slot'una yazar
@triton.jit
def _norms_kernel_v2(grad_ptr, grok_ema_ptr,
                     workspace_ptr,  # [num_blocks, 2] shape'inde
                     alpha, lamb, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grok_ema = tl.load(grok_ema_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    new_ema = alpha * grok_ema + (1.0 - alpha) * grad
    grok_grad = grad + lamb * new_ema

    tl.store(grok_ema_ptr + offs, new_ema, mask=mask)

    g_sq = tl.where(mask, grad * grad, 0.0)
    gg_sq = tl.where(mask, grok_grad * grok_grad, 0.0)

    # Her block kendi slot'una normal store (atomic değil!)
    tl.store(workspace_ptr + pid, tl.sum(g_sq))
    tl.store(workspace_ptr + num_blocks + pid, tl.sum(gg_sq))

# Launch tarafı:
num_blocks = (n + bs - 1) // bs
workspace = torch.zeros(num_blocks * 2, device=grad.device, dtype=torch.float32)
_norms_kernel_v2[grid](grad, grok_ema, workspace, alpha, lamb, n, BLOCK_SIZE=bs)

# CPU'da reduction (birkaç yüz microsecond):
grad_norm_sq = workspace[:num_blocks].sum().item()
grok_norm_sq = workspace[num_blocks:].sum().item()
norms_buf[0] = grad_norm_sq
norms_buf[1] = grok_norm_sq
```

**Yaklaşım 2: Per-block workspace + GPU reduction kernel**
```python
# İkinci bir küçük kernel ile reduction
@triton.jit
def _reduce_kernel(workspace_ptr, output_ptr, n_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_blocks
    vals = tl.load(workspace_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(output_ptr, tl.sum(vals))
```

**Yaklaşım 3: Block-level tree reduction (tek kernel içinde)**
```python
# BLOCK_SIZE'i büyüt, daha az block = daha az contention
# Örnek: 16M element, BLOCK_SIZE=32768 → sadece 512 block
bs = min(32768, triton.next_power_of_2(n))  # 4096 yerine 32768
```

### Riskler
- Yaklaşım 1: CPU reduction GPU sync gerektirir → küçük modellerde yavaş
- Yaklaşım 2: Ek kernel launch overhead → küçük modellerde zararlı
- Yaklaşım 3: Büyük BLOCK_SIZE register baskısı artırır, occupancy düşer
- En iyi yaklaşım param boyutuna göre değişir: küçük params atomic_add, büyük params workspace

### Test
- `ncu --metrics gpu__hardware_atomic_add.sum` ile atomic sayısı karşılaştırma
- Farklı param boyutlarında (1K, 10K, 100K, 1M, 16M) benchmark

---

## O10: Grad fp32 Buffer Ön-allokasyon

### Neden
`grokadamw.py:243`'te `grad = grad.float()` her adımda bf16→fp32 tensör oluşturuyor. PyTorch caching allocator reuse eder ama allocation overhead + type conversion kernel'i per param per step.

### Mevcut Kod
```python
# grokadamw.py:241-243
if not grad.is_contiguous():
    grad = grad.contiguous()
grad = grad.float()  # bf16/fp16 → yeni fp32 tensör
```

### Yapılacak Değişiklik

**State'e fp32 grad buffer ekle:**
```python
# State init'e ekle (sadece param dtype fp32 değilse)
if p.dtype != torch.float32:
    state["grad_fp32_buf"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

# _update_group'ta kullan:
if "grad_fp32_buf" in state:
    grad = state["grad_fp32_buf"].copy_(grad)  # in-place bf16→fp32 conversion
else:
    grad = grad.float()
```

### Riskler
- `copy_()` dtype cast yapar mı? Evet, `tensor.copy_(src)` src'nin dtype'ını cast eder
- Ek VRAM: param dtype fp32 ise buffer allocate edilmez. bf16 modelde: param_size × 1 fp32 buffer
- Bu VRAM zaten `grad.float()` ile allocate ediliyor, sadece persistent hale getiriliyor

### Test
- bf16 model ile numerical equivalence
- `torch.cuda.memory_stats()["num_alloc_retries"]` ile allocation sayısı karşılaştırma

---

## O3: bfloat16 State Saklama

### Neden
3 state tensörü (exp_avg, exp_avg_sq, grok_ema) fp32 yerine bf16 saklanırsa %50 VRAM tasarrufu.

### Yapılacak Değişiklik

**Adım 1:** Yeni constructor parametresi:
```python
def __init__(self, ..., state_dtype: torch.dtype = torch.float32):
    ...
    defaults["state_dtype"] = state_dtype
```

**Adım 2:** State init:
```python
state_dtype = group.get("state_dtype", torch.float32)
state.update({
    "step": 0,
    "exp_avg": torch.zeros(p.shape, dtype=state_dtype, device=state_device),
    "exp_avg_sq": torch.zeros(p.shape, dtype=state_dtype, device=state_device),
    "grok_ema": torch.zeros(p.shape, dtype=state_dtype, device=state_device),
})
```

**Adım 3:** Update sırasında upcast:
```python
# Fallback path'te:
exp_avg = state["exp_avg"].float()  # bf16 → fp32 hesaplama için
exp_avg_sq = state["exp_avg_sq"].float()
grok_ema = state["grok_ema"].float()

# ... hesaplamalar fp32'de ...

# Geri yazarken downcast:
state["exp_avg"] = exp_avg.to(state_dtype)
state["exp_avg_sq"] = exp_avg.to(state_dtype)
state["grok_ema"] = grok_ema.to(state_dtype)
```

**Adım 4:** Triton path'te kernel'in başında upcast, sonunda downcast:
```python
# _update_kernel'de:
exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)  # zaten var
exp_avg_sq = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)
# ... hesaplamalar fp32 ...
tl.store(exp_avg_ptr + offs, exp_avg.to(bf16_dtype), mask=mask)  # downcast
tl.store(exp_avg_sq_ptr + offs, exp_avg_sq.to(bf16_dtype), mask=mask)
```

### Riskler (KRİTİK)

**exp_avg_sq underflow:**
```
β2 = 0.999, grad = 0.01
exp_avg_sq update: β2 * old + (1-β2) * grad²
                  = 0.999 * old + 0.001 * 1e-4
                  = 0.999 * old + 1e-7

bf16 min normal = 9.2e-3 (2^-7)
1e-7 << bf16 min → subnormal veya 0'a round
```

**exp_avg precision loss:**
```
β1 = 0.9, large exp_avg accumulation
bf16: 8 exponent bit, 7 mantissa bit
fp32: 8 exponent bit, 23 mantissa bit
Large + small addition'da small lost (catastrophic cancellation)
```

**Önerilen: Sadece grok_ema ve exp_avg bf16, exp_avg_sq HER ZAMAN fp32**
```python
state.update({
    "exp_avg": torch.zeros(p.shape, dtype=state_dtype, ...),
    "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32, ...),  # her zaman fp32
    "grok_ema": torch.zeros(p.shape, dtype=state_dtype, ...),
})
```
Bu %33 VRAM tasarrufu sağlar (2/3 state bf16) ama underflow riskini ortadan kaldırır.

### Test
- Numerical equivalence: bf16 states vs fp32 states, 100+ step
- Gradient magnitude distribution testi (küçük grad'lı senaryolar)
- Checkpoint save/load uyumluluğu

---

## O15: Flat-Parameter Batching (FusedAdam-style)

### Neden
Her parametre için ayrı kernel launch → 200+ param = 400+ kernel launch. Tüm param'leri flat 1D buffer'da birleştirip tek kernel ile işlemek launch overhead'i dramatik azaltır.

### Mimari Değişiklik

**Adım 1:** Param group'ları flat tensörlere dönüştür:
```python
# Model parametrelerini 1D contiguous tensörlere flatten et
# Her param group için:
flat_param = torch.cat([p.ravel() for p in group_params])
flat_exp_avg = torch.zeros_like(flat_param)
flat_exp_avg_sq = torch.zeros_like(flat_param)
flat_grok_ema = torch.zeros_like(flat_param)

# Offset'leri kaydet (hangi segment hangi param)
offsets = [0]
for p in group_params:
    offsets.append(offsets[-1] + p.numel())
```

**Adım 2:** Gradient'leri flat tensöre topla:
```python
# Forward/backward sonrası:
flat_grad = torch.cat([p.grad.ravel() for p in group_params])
```

**Adım 3:** Tek kernel ile tüm parametreleri güncelle:
```python
@triton.jit
def _flat_update_kernel(
    flat_p_ptr, flat_grad_ptr, flat_exp_avg_ptr, flat_exp_avg_sq_ptr,
    flat_grok_ema_ptr,
    offsets_ptr, layer_beta1_ptr,  # per-segment scalars
    n_segments, total_elements,
    alpha, lamb, beta2, step_size, lr_wd, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Hangi segment'teyiz? (binary search)
    # layer_beta1 segment bazlı, tüm block'lar aynı değer kullanmaz
    # → Bu Triton'da karmaşıklaşır
```

**Temel sorun:** `layer_beta1` her parametre için farklı. Flat tensörde hangi element'in hangi parametreye ait olduğunu bilmek gerekli. Bu, kernel içinde binary search veya lookup table gerektirir.

**Alternatif:** layer_beta1'ı da flat tensör olarak geçir, element bazında lookup:
```python
# Kernel'a param_index tensörü geçir
# Her element için: param_idx[i] → layer_beta1[param_idx[i]]
# AMA bu ek memory bandwidth (1 extra read per element)
```

### Riskler
- Gradient accumulation ile uyumluluk zor (gradient'ler sparse olabilir)
- Model surgery (param ekleme/çıkarma) flat buffer'ı invalidate eder
- `state_dict` formatı tamamen değişir → backward compatibility kırılır
- Param group'lar arası farklı LR/WD desteği karmaşıklaşır
- En büyük risk: doğruluk (segment boundary'lerde yanlış offset)

### Test
- Byte-byte equivalence: flat vs mevcut, 1000+ step
- Model surgery testi
- Checkpoint uyumluluk testi
- Profiler: kernel launch sayısı doğrulama

---

## O4: 8-bit Quantized States

### Neden
%75 state VRAM tasarrufu. Large model training'de kritik.

### Yapılacak Değişiklik

**Block-wise dynamic quantization (bitsandbytes yaklaşımı):**

```python
class QuantizedState:
    def __init__(self, tensor, block_size=2048):
        self.block_size = block_size
        self.shape = tensor.shape
        n = tensor.numel()
        n_blocks = (n + block_size - 1) // block_size

        # Quantize
        flat = tensor.ravel()
        blocks = flat[:n_blocks * block_size].view(n_blocks, block_size)
        absmax = blocks.abs().amax(dim=1)  # per-block max
        scale = absmax / 127.0
        quantized = (blocks / scale.unsqueeze(1)).round().to(torch.int8)

        self.quantized = quantized
        self.scale = scale
        self.block_size = block_size

    def dequantize(self):
        flat = (self.quantized.float() * self.scale.unsqueeze(1)).ravel()
        return flat[:self.shape.numel()].reshape(self.shape)
```

**Triton kernel ile quant/dequant:**
```python
@triton.jit
def _quantize_kernel(input_ptr, output_ptr, scale_ptr,
                     n_elements, block_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_id = pid // (block_size // BLOCK_SIZE)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    val = tl.load(input_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + block_id)
    q = tl.libdevice.llroundf(val / s)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(output_ptr + offs, q.to(tl.int8), mask=mask)
```

### Riskler
- Quantization error accumulation: 1000+ step sonunda hata büyüyebilir
- exp_avg_sq quantization çok küçük değerleri kaybeder (int8 min = -128)
- Dequant → compute → requant döngüsü ek compute overhead
- Training convergence etkisi: genellikle küçük ama model-bağımlı

### Önerilen Aşamalı Yaklaşım
1. Önce sadece `grok_ema` quantize et (en az kritik state)
2. Sonra `exp_avg` quantize et
3. En son `exp_avg_sq` quantize et (en riskli)

Her aşamada convergence testi yap.

### Test
- Training loss curve karşılaştırma: quantized vs fp32, 10K+ step
- Checkpoint uyumluluk
- Memory profiling
- Convergence regression testi: quantized ile aynı accuracy'ye ulaşma

---

## Genel Test Stratejisi

Her optimizasyon için şu test dizisini uygula:

```
1. pytest grokadamw/tests/ -v          → 14/14 geçmeli
2. python -m grokadamw.benchmark        → VRAM ve timing kaydet
3. Numerical equivalence                → cpu_offload vs GPU, fp32 vs bf16
4. Checkpoint save/load                 → eski format ile uyumluluk
5. Edge case: zero-element params       → crash yok
6. Edge case: multiple param groups     → doğru grup ayrımı
```

## VRAM Tasarruf Özeti (Tamamı Uygulanırsa)

```
Mevcut state per param: 3 × fp32 = 12 bytes/param
O1 (yapıldı):            -1 × fp32 = -4 bytes/param (Triton path)
O2:                       -1 × fp32 = -4 bytes/param (fallback path)
O3 (kısmi):               -2 × (fp32-bf16) = -4 bytes/param
O4:                       -3 × (fp32-int8) = -9 bytes/param

1B param model:
  Mevcut (Triton):  12 GB state → O1 sonrası: 8 GB
  +O2:              8 GB → 4 GB (fallback yok)
  +O3 (kısmi):      8 GB → 4 GB
  +O4:               8 GB → 3 GB
```
