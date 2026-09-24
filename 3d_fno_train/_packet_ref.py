"""生成返回数据包字节级参考（与 ue_rcs_service.py 实际格式完全一致，小端）。"""
import struct
import numpy as np

RESP_STRUCT = struct.Struct("<iq5fI")   # 单站包 36B
PAT_HEAD = struct.Struct("<iqdi")       # 方向图头 24B

print("=" * 74)
print("【包1】单站包  36 字节  格式 <iq5fI")
print("=" * 74)
ts = 1788225350030                      # 示例时间戳（UE 原样回传）
los_t, los_p, ef_t, ef_p = 0.0, 359.9, 90.0, 359.9
rcs, valid = -13.3, 1
pkt = RESP_STRUCT.pack(1, ts, los_t, los_p, ef_t, ef_p, rcs, valid)
rows = [
    (0, "i", "Type(int32)"),
    (4, "q", "timestamp(int64)"),
    (12, "f", "los_theta(float32)"),
    (16, "f", "los_phi(float32)"),
    (20, "f", "eff_theta(float32)"),
    (24, "f", "eff_phi(float32)"),
    (28, "f", "rcs_db_sm(float32)"),
    (32, "I", "valid(uint32)"),
]
for off, fmt, name in rows:
    b = pkt[off:off + struct.calcsize(fmt)]
    val = struct.unpack("<" + fmt, b)[0]
    hx = " ".join(f"{x:02x}" for x in b)
    print(f"  偏移{off:>3} {hx:<24}  {name} = {val}")

print("\n=== 包1完整 hex（36B）===")
print("  " + " ".join(f"{x:02x}" for x in pkt))

print("\n" + "=" * 74)
print("【包2】方向图包  24 + 16×n 字节  头格式 <iqdi")
print("=" * 74)
n = 37
theta_deg = 90.0
phi = np.arange(0.0, 360.0 + 1e-6, 10.0)          # 0,10,...,360 → 37 点
rcs_db = np.round(np.linspace(-24.6, -6.5, n), 1)  # 示例值
head = PAT_HEAD.pack(2, ts, theta_deg, n)
print(f"  偏移{0:>3} " + " ".join(f"{x:02x}" for x in head[:4]) + f"{'':<4}  Type(int32) = {2}")
print(f"  偏移{4:>3} " + " ".join(f"{x:02x}" for x in head[4:12]) + f"   timestamp(int64) = {ts}")
print(f"  偏移{12:>3} " + " ".join(f"{x:02x}" for x in head[12:20]) + f"   theta_deg(double) = {theta_deg}")
print(f"  偏移{20:>3} " + " ".join(f"{x:02x}" for x in head[20:24]) + f"{'':<4}  n_sample(int32) = {n}")
print(f"  偏移{24:>3} PhiDegArr double[{n}]：")
for i in range(0, n, 6):
    seg = phi[i:i + 6]
    hx = " ".join(f"{x:02x}" for x in seg.astype('<f8').tobytes())
    print(f"          {hx}   φ={['%.1f' % v for v in seg]}")
print(f"  偏移{24 + 8 * n:>3} RcsDbArr double[{n}]：")
for i in range(0, n, 6):
    seg = rcs_db[i:i + 6]
    hx = " ".join(f"{x:02x}" for x in seg.astype('<f8').tobytes())
    print(f"          {hx}   RCS={['%.1f' % v for v in seg]}")

print(f"\n=== 包2完整大小：{24 + 16 * n} 字节（n=37 → 616B）===")
print("=== UE 接收流程：读36B → Type==1 则再读 24+16×n 字节 ===")
