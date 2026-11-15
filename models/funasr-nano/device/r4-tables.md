| item | iPhone run 1 (fresh install, cold) | iPhone run 2 (relaunch, warm) | Mac A1 Release (cold for this app) | Mac A2 Release (warm) |
|---|---|---|---|---|
| run / status / pass | 20260927-030040 / done / True | 20260927-030429 / done / True | mac-20260927-025503 / done / True | mac-20260927-025652 / done / True |
| device | iPhone19,2 V63AP, iOS 27.0.0 (24A437), arch h19p | iPhone19,2 V63AP, iOS 27.0.0 (24A437), arch h19p | arm64 Mac16,9, macOS 27.0.0 (26A428), arch h16c | arm64 Mac16,9, macOS 27.0.0 (26A428), arch h16c |
| build | Release, testable True | Release, testable True | Release, testable True | Release, testable True |
| GPU lock (Mac) | - | - | taken by this run (flock; file 0 bytes, last written 2026-09-27 02:52:26; GPU utilization [56, 0, 0, 0, 0]) | taken by this run (flock; file 0 bytes, last written 2026-09-27 02:56:52; GPU utilization [98, 0, 0, 0, 0]) |
| assets: files / missing / md5 differ | 325 / 0 / 0 | 325 / 0 / 0 | 325 / 0 / 0 | 325 / 0 / 0 |
| load1 encoder alone (s) | 2.46 | 0.10 | 2.30 | 0.11 |
| load1 model = decoder + cached encoder (s) | 2.82 | 0.90 | 1.44 | 0.55 |
| load1 encoder warm (s) | 0.14 | 0.15 | 0.11 | 0.12 |
| load1 decoder ≈ model − encoder warm (s) | 2.67 | 0.75 | 1.33 | 0.43 |
| load1 encoder + decoder ≈ (s) | 5.14 | 0.85 | 3.62 | 0.54 |
| Core AI cache before → after load1 (MB) | 0 → 1564 | 1564 → 1564 | 0 → 1564 | 1564 → 1564 |
| warmup first transcription (ms, clip) | 2256 (zh, exact) | 552 (zh, exact) | 221 (zh, exact) | 216 (zh, exact) |
| load2 model, same process (s) | 0.44 | 0.44 | 0.55 | 0.56 |
| e2e clips done / errors | 155 / 0 | 155 / 0 | 155 / 0 | 155 / 0 |
| oracle exact | 150 | 150 | 150 | 150 |
| knife-edge (< 0.1) | 5 | 5 | 5 | 5 |
| exact or knife-edge | 155 | 155 | 155 | 155 |
| mismatch ≥ 0.1 | 0 | 0 | 0 | 0 |
| ids == Python engine | 155 | 155 | 155 | 155 |
| ids == Mac Swift (r3) | 155 | 155 | 155 | 155 |
| text == oracle / Python / Mac Swift | 150 / 155 / 155 | 150 / 155 / 155 | 150 / 155 / 155 | 150 / 155 / 155 |
| cap (512) | 0 | 0 | 0 | 0 |
| front end max|Δ| vs NumPy (clip) | 4.93e-04 (ja_jp_1670) | 4.93e-04 (ja_jp_1670) | 4.93e-04 (ja_jp_1670) | 4.93e-04 (ja_jp_1670) |
| e2e audio / transcription wall (s) | 1836 / 132.1 | 1836 / 141.7 | 1836 / 39.5 | 1836 / 39.2 |
| e2e RTF median / p90 | 0.0745 / 0.0937 | 0.0807 / 0.0994 | 0.0221 / 0.0280 | 0.0220 / 0.0280 |
| e2e RTF worst (clip) | 0.1370 (ko) | 0.1350 (ko) | 0.0434 (ko) | 0.0435 (ko) |
| e2e RTF median, clips started in the first 20 s / after | 0.0780 (28) / 0.0745 | 0.0781 (28) / 0.0811 | 0.0234 (82) / 0.0215 | 0.0234 (83) / 0.0214 |
| e2e median front end / encoder / prefill (ms) | 2.2 / 146.1 / 392.8 | 2.4 / 225.1 / 417.2 | 1.4 / 40.2 / 123.7 | 1.3 / 40.2 / 122.6 |
| e2e median decode (ms/token) | 8.17 | 9.41 | 2.79 | 2.77 |
| bench clip, runs, wait for nominal | ja_jp_1719, 5 after 1 warm-up, waited 46 s (fair → nominal) | ja_jp_1719, 5 after 1 warm-up, waited 302 s (serious → serious) | ja_jp_1719, 5 after 1 warm-up, waited 0 s (nominal → nominal) | ja_jp_1719, 5 after 1 warm-up, waited 0 s (nominal → nominal) |
| bench RTF median / p90 | 0.0665 / 0.0667 | 0.0667 / 0.0676 | 0.0218 / 0.0218 | 0.0216 / 0.0216 |
| bench wall median (ms) | 906.3 | 908.8 | 296.5 | 294.0 |
| bench encoder / prefill (ms), decode (ms/token) | 150.5 / 415.6, 8.18 | 151.0 / 416.0, 8.16 | 42.0 / 136.9, 2.80 | 42.0 / 135.6, 2.77 |
| bench timed runs end at (s) | 5.5 | 5.5 | 1.8 | 1.8 |
| bench thermal start → end | nominal → nominal | serious → fair | nominal → nominal | nominal → nominal |
| peak footprint over the run (MB) | 385 | 411 | 727 | 737 |
| least available memory (MB) | 3157 | 3133 | - | - |
| e2e thermal start / ~20 s / end | nominal (0 s) / nominal (20 s) / fair (134 s) | nominal (0 s) / nominal (20 s) / serious (143 s) | nominal (0 s) / nominal (20 s) / nominal (40 s) | nominal (0 s) / nominal (20 s) / nominal (40 s) |
| thermal at launch → end of run | nominal → nominal | nominal → fair | nominal → nominal | nominal → nominal |
| battery at launch | 80 % charging | 80 % charging | -100 % unknown | -100 % unknown |
| elapsed (s) | 197 | 456 | 50 | 47 |

- iPhone run 1 (fresh install, cold): /Users/USER/code/coreai/coreai-models-community-funasr-wt/apps/FunASRGate/_work/device_runs/20260927-030040/result.json
- iPhone run 2 (relaunch, warm): /Users/USER/code/coreai/coreai-models-community-funasr-wt/apps/FunASRGate/_work/device_runs/20260927-030429/result.json
- Mac A1 Release (cold for this app): /Users/USER/code/coreai/coreai-models-community-funasr-wt/apps/FunASRGate/_work/mac_runs/mac-20260927-025503/result.json
- Mac A2 Release (warm): /Users/USER/code/coreai/coreai-models-community-funasr-wt/apps/FunASRGate/_work/mac_runs/mac-20260927-025652/result.json

## Testability control (Mac, paired per clip: the public-API build without -enable-testing / the testable build; run-to-run lines for scale)
```
A1 testable -> B1 public:
paired clips 155: wall B/A median 0.9933, mean 0.9934, min 0.9759, max 1.0075; sum B / sum A 0.9932; text equal 155/155
bench wall median A 296.483584 ms, B 293.42775000000006 ms
A2 testable -> B1 public:
paired clips 155: wall B/A median 0.9995, mean 0.9993, min 0.9813, max 1.0156; sum B / sum A 0.9992; text equal 155/155
bench wall median A 294.004166 ms, B 293.42775000000006 ms
A2 testable -> B2 public:
paired clips 155: wall B/A median 0.9992, mean 0.9993, min 0.9861, max 1.0112; sum B / sum A 0.9994; text equal 155/155
bench wall median A 294.004166 ms, B 294.58641600000004 ms
A1 testable -> A2 testable (run to run):
paired clips 155: wall B/A median 0.9939, mean 0.9941, min 0.9843, max 1.0100; sum B / sum A 0.9940; text equal 155/155
bench wall median A 296.483584 ms, B 294.004166 ms
B1 public -> B2 public (run to run):
paired clips 155: wall B/A median 1.0002, mean 1.0001, min 0.9872, max 1.0137; sum B / sum A 1.0002; text equal 155/155
bench wall median A 293.42775000000006 ms, B 294.58641600000004 ms
mac-20260927-025503: app priority min 31 (31 = foreground user process; App Nap = 4), demoted samples 0 , load avg start/end 17.7/28.2
mac-20260927-025557-public: app priority min 31 (31 = foreground user process; App Nap = 4), demoted samples 0 , load avg start/end 28.2/15.8
mac-20260927-025652: app priority min 31 (31 = foreground user process; App Nap = 4), demoted samples 0 , load avg start/end 15.8/11.3
mac-20260927-025747-public: app priority min 31 (31 = foreground user process; App Nap = 4), demoted samples 0 , load avg start/end 11.3/7.9
```

## Phone vs Mac, and the phone's speed over the e2e stage
```
iPhone run 1 vs Mac A2 per clip: waveform md5 equal 155/155, Swift front-end features md5 equal 155/155, generated ids equal 155/155; iPhone run 1 vs run 2 ids equal 155/155
iPhone run 1 thermal every 20 s: [(0, 'nominal'), (20, 'nominal'), (40, 'nominal'), (60, 'nominal'), (80, 'nominal'), (100, 'fair'), (120, 'fair'), (134, 'fair')]
  e2e   0- 20 s: 28 clips, wall iPhone/Mac x3.13, encoder 146 ms, decode 8.16 ms/token (medians)
  e2e  20- 40 s: 26 clips, wall iPhone/Mac x3.12, encoder 146 ms, decode 8.17 ms/token (medians)
  e2e  40- 60 s: 26 clips, wall iPhone/Mac x3.12, encoder 146 ms, decode 8.16 ms/token (medians)
  e2e  60- 80 s: 27 clips, wall iPhone/Mac x3.10, encoder 146 ms, decode 8.14 ms/token (medians)
  e2e  80-100 s: 19 clips, wall iPhone/Mac x3.62, encoder 227 ms, decode 8.81 ms/token (medians)
  e2e 100-120 s: 19 clips, wall iPhone/Mac x3.93, encoder 231 ms, decode 10.04 ms/token (medians)
  e2e 120-140 s: 10 clips, wall iPhone/Mac x4.08, encoder 232 ms, decode 10.75 ms/token (medians)
iPhone run 2 thermal every 20 s: [(0, 'nominal'), (20, 'nominal'), (40, 'fair'), (60, 'serious'), (80, 'serious'), (100, 'serious'), (120, 'serious'), (140, 'serious'), (143, 'serious')]
  e2e   0- 20 s: 28 clips, wall iPhone/Mac x3.13, encoder 147 ms, decode 8.17 ms/token (medians)
  e2e  20- 40 s: 26 clips, wall iPhone/Mac x3.14, encoder 147 ms, decode 8.16 ms/token (medians)
  e2e  40- 60 s: 24 clips, wall iPhone/Mac x3.16, encoder 148 ms, decode 8.23 ms/token (medians)
  e2e  60- 80 s: 22 clips, wall iPhone/Mac x3.94, encoder 233 ms, decode 9.87 ms/token (medians)
  e2e  80-100 s: 18 clips, wall iPhone/Mac x3.92, encoder 234 ms, decode 9.94 ms/token (medians)
  e2e 100-120 s: 18 clips, wall iPhone/Mac x3.94, encoder 234 ms, decode 10.11 ms/token (medians)
  e2e 120-140 s: 17 clips, wall iPhone/Mac x4.06, encoder 234 ms, decode 10.49 ms/token (medians)
  e2e 140-160 s:  2 clips, wall iPhone/Mac x4.23, encoder 236 ms, decode 11.17 ms/token (medians)
```
