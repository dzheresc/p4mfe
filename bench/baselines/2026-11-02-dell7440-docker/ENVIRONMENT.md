# Baseline environment

| | |
|---|---|
| Date (UTC) | 2026-11-02 01:15 |
| Git commit | 35a5380364675966fddfed04c56ee6dcc28ec15f (dirty working tree) |
| CPU | Intel(R) Core(TM) Ultra 9 185H |
| Logical CPUs | 22 |
| Memory | 15719 MiB |
| CPU governor | (unavailable) |
| CPU affinity mask | 4 |
| OS | Linux 5.15.167.4-microsoft-standard-WSL2 |
| Distribution | Ubuntu 24.04.5 LTS |
| Compiler | c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| Build type | Release |
| Extra CXX flags | -O3 -DNDEBUG |
| Google Benchmark | 1.9.1 |
| Tier | standard |
| CI | no |

## lscpu

```
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s):                               22
On-line CPU(s) list:                  0-21
Vendor ID:                            GenuineIntel
Model name:                           Intel(R) Core(TM) Ultra 9 185H
CPU family:                           6
Model:                                170
Thread(s) per core:                   2
Core(s) per socket:                   11
Socket(s):                            1
Stepping:                             4
BogoMIPS:                             6143.99
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves avx_vnni umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize flush_l1d arch_capabilities
Virtualization:                       VT-x
Hypervisor vendor:                    Microsoft
Virtualization type:                  full
L1d cache:                            528 KiB (11 instances)
L1i cache:                            704 KiB (11 instances)
L2 cache:                             22 MiB (11 instances)
L3 cache:                             24 MiB (1 instance)
Vulnerability Gather data sampling:   Not affected
Vulnerability Itlb multihit:          Not affected
Vulnerability L1tf:                   Not affected
Vulnerability Mds:                    Not affected
Vulnerability Meltdown:               Not affected
Vulnerability Mmio stale data:        Not affected
Vulnerability Reg file data sampling: Not affected
Vulnerability Retbleed:               Mitigation; Enhanced IBRS
Vulnerability Spec rstack overflow:   Not affected
Vulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                  Not affected
Vulnerability Tsx async abort:        Not affected
```
﻿
## Run conditions (local, recorded 2026-11-02)

- Host: dell7440, Windows 11 Home 10.0.26300; Intel Core Ultra 9 185H (hybrid: 6 P + 8 E + 2 LP-E cores, 22 threads), 31.5 GB RAM.
- Ran in an Ubuntu 24.04 container on Docker Desktop 29.5.2 (WSL2 backend; VM sees 22 vCPUs, 16 GB); repo cloned inside a named volume.
- Pinned with `taskset -c 2` (a VM vCPU; Hyper-V may still schedule it on a P- or E-core). No Linux cpufreq governor is exposed inside the VM.
- Windows power plan: High performance for the run (restored to Balanced afterwards); on AC power.
- Background apps running (not closed): several Claude Code processes, Chrome, Docker Desktop, Logi Options+; idle total CPU about 4-10% before the run.
