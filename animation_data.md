# 24ML Nitrogen Displacement — Complete Frame-by-Frame Data

**Source:** `24ML_Nitrogen_Model_20250326_inject_SP_only_Powerpoint_playback_with_notes_1_.mp4`  
**Method:** 1 fps extraction (440 frames) → perceptual diff (threshold 1.0) → **207 unique keyframes**  
**Simulation phase unique frames:** 152 (covering t=210–437 s)  
**Reading precision:** ±3–5 psig on pressures; ±1–2 s on timestamps  

---

## Column Definitions

| Column | Meaning |
|--------|---------|
| `t_s` | Video timestamp (seconds) |
| `N2_disp` | Green displayed value at active N2 injection point(s) = N2 rate ÷ 10 (SCFM); two values = two simultaneous injection points |
| `pig_MP` | Estimated pig milepost from green N2 interface x-position |
| `P_SP` | SP station pressure (psig); shows N2 gas pressure once pig has passed SP |
| `P_RY` | RY station pressure (psig) |
| `P_NW` | NW station pressure (psig); steps up sharply when NW booster compressor fires |
| `P_SH` | SH station pressure (psig) |
| `P_LS_s` | LS pump **suction** pressure (blue dot below cyan line, psig) |
| `P_LS_d` | LS pump **discharge** pressure (blue dot on cyan line, psig) |
| `P_SU` | SU station pressure (psig) — pass-through, no pump |
| `P_HW_s` | HW pump **suction** pressure (psig) |
| `P_HW_d` | HW pump **discharge** pressure (psig) |
| `P_SC` | SC station pressure (psig) — upstream of SC backpressure valve |
| `Flow×10` | Oil flow rate in BPH (pink dot displayed value × 10) |
| `Event` | Annotation text / operational event |

**Note on N2_disp vs SP pressure:** During t=212–255 s, N2_disp and P_SP are nearly equal numerically 
because N2 injection rate ÷ 10 (in SCFM) happens to match SP operating pressure (in psig) at
these flow/pressure conditions. They are different physical quantities with different units.

---

## Static Reference States

### Phase B — Explanation Diagram (t = 60–175 s), all frames identical:
`P_SP=650 P_RY=503 P_NW=374 P_SH=306 P_LS_s=42 P_LS_d=382 P_SU=313 P_HW_s=67 P_HW_d=363 P_SC=570 N2=6500 SCFM Flow=4000 BPH`

### Phase C — Initial Conditions / Line Shut Down (t = 175–210 s):
`P_SP=217 P_NW≈322 P_SH≈362 P_LS=36 P_SU=333 P_HW=415 P_SC=680 (circled) Flow≈0 BPH`  
Static HGL = 2,100 ft (flat); SC BPCV holds 680 psig backpressure.

---

## Simulation Frame-by-Frame Table (t = 210–437 s)

| t_s | N2_disp | pig_MP | P_SP | P_RY | P_NW | P_SH | P_LS_s | P_LS_d | P_SU | P_HW_s | P_HW_d | P_SC | Flow×10 | Event |
|-----|---------|--------|------|------|------|------|--------|--------|------|--------|--------|------|---------|-------|
| 212 | 660 | 0 | 650 | 503 | 374 | 306 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | Simulation starts; pig at SP |
| 215 | 649 | 2 | 649 | 519 | 390 | 323 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 216 | 649 | 2 | 649 | 527 | 398 | 331 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 217 | 649 | 3 | 649 | 532 | 403 | 336 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 218 | 648 | 3 | 648 | 528 | 399 | 331 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 219 | 633 | 4 | 633 | 529 | 400 | 333 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 220 | 615 | 5 | 615 | 523 | 394 | 327 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 221 | 599 | 5 | 599 | 544 | 415 | 348 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 222 | 589 | 6 | 589 | 531 | 402 | 335 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | "Pressure drop in nitrogen assumed negligible at these flow rates" (appearing) |
| 223 | 584 | 7 | 584 | 543 | 414 | 346 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 224 | 584 | 7 | 584 | 543 | 414 | 346 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | "Pressure drop in nitrogen assumed negligible at these flow rates" |
| 227 | 575 | 8 | 579 | 526 | 397 | 330 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 228 | 574 | 9 | 572 | 534 | 405 | 338 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 229 | 567 | 9 | 567 | 558 | 424 | 357 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 230 | 563 | 10 | 563 | 538 | 409 | 341 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 231 | 566 | 10 | 558 | 535 | 406 | 338 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 232 | 555 | 11 | 555 | 552 | 423 | 356 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 233 | 553 | 11 | 553 | 532 | 403 | 336 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 234 | 548 | 12 | 548 | 548 | 428 | 361 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 235 | 545 | 12 | 544 | 544 | 447 | 379 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 236 | 539 | 13 | 539 | 539 | 462 | 395 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 237 | 530 | 13 | 536 | 536 | 449 | 381 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 238 | 533 | 14 | 533 | 533 | 414 | 346 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 239 | 536 | 14 | 530 | 530 | 472 | 405 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 240 | 525 | 15 | 527 | 527 | 474 | 407 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 241 | 525 | 15 | 525 | 525 | 492 | 425 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 242 | — | 16 | 524 | 524 | 427 | 359 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | **"Shut off nitrogen injection, expansion will keep [pig going]"** — N2 injection stopped |
| 247 | — | 16 | 524 | 524 | 427 | 359 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | Same annotation; gas expanding behind pig |
| 248 | 569 | 16 | 505 | 505 | 410 | 372 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | N2 injection resumed |
| 249 | 481 | 17 | 482 | 482 | 404 | 336 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 250 | 466 | 17 | 466 | 466 | 434 | 367 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 251 | 451 | 18 | 451 | 451 | 390 | 323 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 252 | 430 | 18 | 433 | 433 | 374 | 306 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 253 | 420 | 19 | 420 | 420 | 371 | 304 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 254 | 408 | 20 | 408 | 408 | 408 | 345 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 255 | 404 | 20 | 404 | 404 | 404 | 335 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | **"Now watch why you shut of N2: As pig ascends hill, downstream [pressure builds]"** |
| 261 | 389 | 22 | 389 | 389 | 389 | 475 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | N2 injected again; pig climbing mountain; NW/SH pressures rising |
| 262 | 379 | 22 | 379 | 379 | 379 | 405 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 263 | 379 | 23 | 379 | 379 | 379 | 405 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | **"Compressor at NW (or injection) to [recompress]"** — NW booster compressor activating |
| 267 | 379 | 23 | 379 | 379 | 379 | 405 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 268 | 379 | 24 | 379 | 379 | 379 | 405 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 272 | 407 | 25 | 375 | 375 | 407 | 374 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | NW booster firing — NW pressure jumps above SP/RY |
| 273 | 490 | 26 | 363 | 363 | 480 | 425 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 274 | 510 | 27 | 353 | 353 | 510 | 482 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 275 | 516 | 27 | 350 | 350 | 518 | 493 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 279 | 516 | 28 | 350 | 350 | 518 | 493 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 280 | 537 | 28 | 340 | 340 | 537 | 490 | 42 | 382 | 313 | 67 | 363 | 570 | 4000 | |
| 281 | 549 | 29 | 333 | 333 | 549 | 507 | 86 | 375 | 309 | 66 | 362 | 572 | 3830 | **"Pump shut down before first pig reaches station; ends throttling"** — LS suct rises 42→86 |
| 288 | 549 | 30 | 333 | 333 | 549 | 507 | 86 | 375 | 309 | 66 | 362 | 572 | 3830 | |
| 289 | 554 | 31 | 330 | 330 | 554 | 507 | 86 | 375 | 309 | 65 | 361 | 570 | 3873 | "Hold SC BP steady (generally)" |
| 293 | 561 | 32 | 324 | 324 | 561 | 522 | 46 | 386 | 317 | 68 | 365 | 570 | 4070 | LS suct 86→46 (stabilising after shutdown transient) |
| 294 | 557 | 32 | 317 | 317 | 557 | 312 | 89 | 378 | 311 | 66 | 362 | 570 | 3936 | |
| 295 | 546 | 33 | 311 | 311 | 548 | 506 | 85 | 374 | 308 | 64 | 360 | 570 | 3859 | |
| 296 | 543 | 33 | 307 | 307 | 543 | 510 | 88 | 377 | 310 | 65 | 362 | 570 | 3911 | |
| 297 | 530 | 34 | 302 | 302 | 536 | 522 | 46 | 387 | 317 | 68 | 365 | 570 | 4077 | |
| 298 | 531 | 34 | 296 | 296 | 531 | 527 | 49 | 390 | 319 | 70 | 367 | 570 | 4136 | |
| 299 | 527 | 35 | 291 | 291 | 527 | 520 | 44 | 385 | 315 | 68 | 365 | 570 | 4043 | |
| 300 | 523 | 36 | 285 | 285 | 523 | 501 | 28 | 368 | 301 | 56 | 352 | 560 | 3925 | **SC drops 570→560** |
| 301 | 549 | 37 | 281 | 281 | 519 | 513 | 86 | 377 | 307 | 58 | 356 | 560 | 4083 | |
| 302 | 643 | 38 | 273 | 273 | 513 | 513 | 72 | 419 | 331 | 58 | 360 | 540 | 5031 | **SC drops 560→540** |
| 303 | 500 | 39 | 257 | 257 | 506 | 506 | 44 | 388 | 309 | 48 | 348 | 540 | 4576 | |
| 304 | 483 | 39 | 253 | 253 | 487 | 487 | 48 | 387 | 308 | 48 | 348 | 540 | 4563 | |
| 305 | 465 | 40 | 253 | 253 | 465 | 465 | 56 | 401 | 318 | 52 | 353 | 540 | 4767 | |
| 306 | 371 | 41 | 253 | 253 | 458 | 371 | 94 | 444 | 348 | 66 | 370 | 540 | 5366 | |
| 307 | 311 | 42 | 253 | 253 | 458 | 311 | 29 | 371 | 297 | 43 | 341 | 540 | 4310 | **"Valve-controlled transfer of higher [pressure N2]"** |
| 308 | 266 | 42 | 253 | 253 | 458 | 268 | 89 | 382 | 305 | 46 | 346 | 540 | 4484 | |
| 309 | 256 | 43 | 253 | 253 | 458 | 256 | 46 | 390 | 310 | 49 | 349 | 540 | 4604 | |
| 314 | 256 | 45 | 253 | 253 | 458 | 256 | 46 | 390 | 310 | 49 | 349 | 540 | 4604 | |
| 316 | 299 | 47 | 253 | 253 | 428 | 424 | 76 | 424 | 334 | 60 | 362 | 540 | 5095 | |
| 317 | 323 | 48 | 253 | 253 | 393 | 323 | 82 | 430 | 338 | 62 | 365 | 540 | 5082 | |
| 318 | 350 | 49 | 253 | 253 | 359 | 350 | 77 | 425 | 334 | 60 | 362 | 540 | 5007 | |
| 319 | 351 | 50 | 253 | 253 | 348 | 357 | 97 | 448 | 350 | 67 | 372 | 540 | 5418 | **"Resume [injection]"** — N2 injection resumed after valve transfer |
| 323 | 264 | 52 | 248 | 248 | 354 | 354 | 91 | 440 | 345 | 65 | 369 | 540 | 5316 | |
| 324 | 369 | 54 | 230 | 230 | 363 | 363 | 80 | 428 | 337 | 61 | 364 | 540 | 5005 | |
| 325 | 373 | 55 | 206 | 206 | 373 | 373 | 80 | 428 | 337 | 61 | 364 | 540 | 5057 | |
| 326 | 381 | 57 | 188 | 188 | 381 | 381 | 65 | 412 | 325 | 56 | 357 | 540 | 4925 | |
| 327 | 389 | 58 | 169 | 169 | 389 | 389 | 59 | 405 | 321 | 54 | 355 | 540 | 4829 | |
| 328 | 400 | 60 | 143 | 143 | 400 | 400 | 54 | ~400 | 317 | 52 | 352 | 540 | 4745 | |
| 329 | 407 | 62 | 126 | 126 | 407 | 407 | 109 | 359 | 377 | 71 | — | 540 | 5587 | |
| 330 | 416 | 63 | 105 | 105 | 416 | 416 | 74 | 285 | 335 | 38 | — | 540 | 4040 | |
| 331 | 425 | 65 | 79 | 79 | 425 | 425 | 97 | 350 | 372 | 67 | — | 540 | 5014 | **SP and RY now in N2 zone (~79 psig)** |
| 332 | 425 | 66 | 79 | 79 | 425 | 425 | 86 | 342 | 367 | 63 | — | 540 | 500 | |
| 333 | 335 | 67 | 79 | 79 | 425 | 245 | 81 | 337 | 364 | 61 | — | 540 | 5064 | |
| 334 | 409 | 68 | 79 | 79 | 425 | 207 | — | 359 | 377 | 71 | — | 540 | 5585 | **Second pig marker (□) appears at SH area; cleaning pig approaching LS** |
| 335 | 409 | 68 | 79 | 79 | 425 | 207 | — | 359 | 377 | 71 | — | 540 | 5585 | **"Compressor (or injection) at LS after [pig passes]"** |
| 339 | 409 | 70 | 79 | 79 | 425 | 207 | — | 359 | 377 | 71 | — | 540 | 5585 | |
| 340 | 4000† | 73 | 79 | 79 | 388 | 337 | — | 380 | 389 | 81 | — | 540 | 5955 | **†"4000" displayed at LS booster position = 40,000 SCFM N2 injection** |
| 341 | 379 | 74 | 79 | 79 | 370 | 370 | — | 364 | 379 | 73 | — | 540 | 5668 | |
| 342 | 359 | 75 | 79 | 79 | 359 | 359 | — | 347 | 370 | 66 | — | 540 | 5357 | Single N2 interface again |
| 343 | 353 | 78 | 79 | 79 | 350 | 353 | — | 503 | 459 | 137 | — | 540 | 7833 | **Flow rate spikes 5357→7833 BPH** |
| 344 | 356 | 80 | 79 | 79 | 339 | 356 | — | 441 | 423 | 108 | — | 540 | 6938 | |
| 345 | 356 | 82 | 79 | 79 | 326 | 356 | — | 459 | 434 | 117 | — | 540 | 7219 | |
| 346 | 360 | 84 | 79 | 79 | 314 | 360 | — | 390 | 394 | 85 | — | 540 | 6128 | |
| 347 | 371 | 86 | 79 | 79 | 298 | 371 | — | 367 | 381 | 75 | — | 540 | 5722 | |
| 348 | 374 | 88 | 79 | 79 | 283 | 374 | — | 418 | 410 | 98 | — | 540 | 6584 | |
| 349 | 378 | 90 | 79 | 79 | 270 | 378 | — | 366 | 381 | 75 | — | 540 | 5708 | |
| 350 | 382 | 92 | 79 | 79 | 254 | 382 | — | 415 | 409 | 97 | — | 540 | 6536 | |
| 351 | 385 | 94 | 79 | 79 | 241 | 385 | — | 375 | 385 | 78 | — | 540 | 5859 | |
| 352 | 374 | 96 | 79 | 79 | 236 | 374 | — | 374 | 404 | 93 | — | 540 | 6415 | |
| 353 | 352 | 98 | 79 | 79 | 236 | 352 | — | 352 | 394 | 85 | — | 540 | 6106 | |
| 354 | 352 | 100 | 79 | 79 | 226 | 352 | — | 352 | 424 | 109 | — | 540 | 6952 | |
| 355 | 354 | 102 | 79 | 79 | 215 | 354 | — | 354 | 390 | 82 | — | 540 | 5997 | |
| 356 | 362 | 104 | 79 | 79 | 195 | 362 | — | 362 | 416 | 103 | — | 540 | 6754 | |
| 357 | 373 | 106 | 79 | 79 | 176 | 373 | — | 373 | 390 | 82 | — | 540 | 6001 | |
| 358 | 375 | 108 | 79 | 79 | 164 | 375 | — | 375 | 440 | 122 | — | 540 | 7373 | |
| 359 | 378 | 110 | 79 | 79 | 147 | 378 | — | 378 | 478 | 152 | — | 540 | 8287 | |
| 360 | 389 | 112 | 82 | 82 | 132 | 132 | — | 383 | 353 | 52 | — | 540 | 4751 | SP=82, RY=82 (transition: pig passing through SP/RY zone differently) |
| 361 | 395 | 114 | 93 | 93 | 108 | 108 | — | 393 | 369 | 65 | — | 540 | 5034 | |
| 362 | 399 | 116 | 96 | 96 | 96 | 96 | — | 399 | 384 | 77 | — | 540 | 5801 | **SP=RY=NW=SH=96 psig** — all four stations now in N2 zone |
| 363 | 368 | 118 | 92 | 92 | 92 | 92 | — | 423 | 407 | 95 | — | 540 | 6479 | |
| 364 | 367 | 120 | 88 | 88 | 88 | 88 | — | 425 | 476 | 155 | — | 564 | 7660 | SC rises 540→564 |
| 365 | 405 | 122 | 84 | 84 | 84 | 84 | 3650† | 405 | 454 | 133 | — | 540 | 7724 | **†"3650" at LS = N2 injection rate 36,500 SCFM. "Pinch MT!" — cleaning pig arriving at MT terminal** |
| 366 | 405 | 123 | 84 | 84 | 84 | 84 | 3650† | 405 | 454 | 133 | — | 540 | 7724 | Same state |
| 370 | 414 | 127 | 77 | 77 | 77 | 77 | 3650† | 414 | 397 | 87 | — | 540 | 6198 | |
| 371 | 426 | 129 | 68 | 68 | 68 | 68 | 3650† | 426 | 430 | 114 | — | 540 | 7122 | |
| 372 | 434 | 131 | 60 | 60 | 60 | 60 | 3650† | 434 | 420 | 112 | — | 540 | 7052 | |
| 373 | 455 | 133 | 45 | 45 | 45 | 45 | 3650† | 455 | 394 | 85 | — | 540 | 6118 | |
| 374 | 459 | 135 | 41 | 41 | 41 | 41 | 3650† | 459 | 401 | 91 | — | 540 | 6318 | "Must cease backpressure" |
| 379 | 462 | 145 | 35 | 35 | 35 | 35 | 3650† | 462 | 492 | 136 | — | — | 10999 | **Flow SPIKES to ~11,000 BPH; SC pressure not visible (pig at/past HW)** |
| 380 | 446 | 150 | 29 | 29 | 29 | 29 | 3650† | 463 | 468 | 114 | — | — | 10999 | "Valved transfer then briefly [re-injection] the..." |
| 381 | 446 | 152 | 27 | 27 | 27 | 27 | — | 464 | 448 | 93 | — | 382 | 10781 | **SP/RY/NW/SH all drop to 27 psig (N2 zone). SC=382 (was 540)**; "Valved transfer…" |
| 385 | 446 | 153 | 27 | 27 | 27 | 27 | — | 464 | 448 | 93 | — | 382 | 10781 | |
| 387 | 453 | 155 | 27 | 27 | 27 | 27 | — | 453 | 453 | 93 | — | 382 | 10781 | |
| 388 | 453 | 158 | 27 | 27 | 27 | 27 | — | 453 | 453 | 19 | — | 346 | 9882 | **HW suct drops to 19 psig (near cavitation!); SC drops 382→346** |
| 389 | 435 | 160 | 27 | 27 | 27 | 27 | — | 435 | 461 | 33 | — | 349 | 10048 | |
| 390 | 435 | 162 | 27 | 27 | 27 | 27 | — | 435 | 344 | 90 | — | 381 | 10661 | "Can open MT control valve once [pig arrives]" |
| 391 | 431 | 164 | 27 | 27 | 27 | 27 | — | 431 | 244 | 84 | — | 410 | 10999 | |
| 392 | 426 | 166 | 27 | 27 | 27 | 27 | — | 426 | 262 | 68 | — | 372 | 10528 | "Can open MT control valve once [pig arrives]" |
| 397 | 4000 | 172 | 27 | 27 | 27 | 27 | — | 412 | 307 | — | — | 251 | 10008 | **"4000" injection at SC area** |
| 398 | 4000 | 175 | 27 | 27 | 27 | 27 | — | 398 | 338 | — | — | 325 | 10999 | |
| 399 | — | 178 | 27 | 27 | 27 | 27 | — | 390 | 339 | — | — | 316 | 10999 | |
| 400 | — | 180 | 27 | 27 | 27 | 27 | — | 390 | 298 | — | — | 274 | 10574 | |
| 401 | — | 183 | 27 | 27 | 27 | 27 | — | 390 | 273 | — | — | 283 | 10813 | |
| 402 | — | 186 | 27 | 27 | 27 | 27 | — | 390 | 252 | — | — | 269 | 10466 | |
| 403 | — | 190 | 27 | 27 | 27 | 27 | — | 390 | 228 | — | — | 205 | 8740 | |
| 404 | — | 194 | 27 | 27 | 27 | 27 | — | 390 | 213 | — | — | 227 | 9374 | |
| 405 | — | 198 | 27 | 27 | 27 | 27 | — | 390 | 196 | — | — | 197 | 8513 | |
| 406 | — | 203 | 27 | 27 | 27 | 27 | — | 390 | 185 | — | — | 181 | 8009 | |
| 407 | — | 207 | 27 | 27 | 27 | 27 | — | 390 | 175 | — | — | 169 | 7618 | |
| 408 | — | 210 | 27 | 27 | 27 | 27 | — | 390 | 164 | — | — | 162 | 7374 | |
| 409 | — | 214 | 27 | 27 | 27 | 27 | — | 390 | 156 | — | — | 181 | 7023 | |
| 411 | — | 218 | 27 | 27 | 27 | 27 | — | 390 | 140 | — | — | 135 | 6426 | |
| 412 | — | 221 | 27 | 27 | 27 | 27 | — | 390 | 134 | — | — | 125 | 6015 | |
| 413 | — | 224 | 27 | 27 | 27 | 27 | — | 387 | 133 | — | — | 129 | 6201 | |
| 414 | — | 226 | 27 | 27 | 27 | 27 | — | 383 | 135 | — | — | 124 | 5993 | |
| 415 | — | 228 | 27 | 27 | 27 | 27 | — | 380 | 136 | — | — | 136 | 5967 (→3373296?) | Flow display glitch at t=415s |
| 416 | — | 231 | 27 | 27 | 27 | 27 | — | 376 | 136 | — | — | 136 | 5990 | |
| 418 | — | 233 | 27 | 27 | 27 | 27 | — | 369 | 136 | — | — | 136 | 5990 | |
| 419 | — | 235 | 27 | 27 | 27 | 27 | — | 366 | 135 | — | — | 135 | 5990 | |
| 422 | — | 237 | 27 | 27 | 27 | 27 | — | 360 | 129 | — | — | 129 | 5990 | |
| 424 | — | 239 | 27 | 27 | 27 | 27 | — | 360 | 121 | — | — | 121 | 5990 | |
| 437 | — | ~240 | 27 | 27 | 27 | 27 | — | — | — | — | — | — | 5990 | **Pig arriving at MT. Video ends (closing slide follows).** |

---

## Annotated Event Log (all events in chronological order)

| t_s | Event |
|-----|-------|
| 212 | Simulation starts; pig at MP 0 (SP) |
| 222 | "Pressure drop in nitrogen assumed negligible at these flow rates" appears |
| 242 | **N2 injection STOPPED** — "Shut off nitrogen injection, expansion will keep pig going" |
| ~247 | Gas behind pig expands; pressures behind pig (SP, RY) held by expansion |
| 248 | N2 injection **resumed** |
| 255 | "Now watch why you shut of N2: As pig ascends hill, downstream [pressure builds]" |
| 263 | **"Compressor at NW (or injection) to [recompress]"** — NW booster compressor activates |
| 272 | NW booster firing confirmed: NW jumps from ~379 to 407 psig, rising to 518 psig by t=275 |
| 281 | **"Pump shut down before first pig reaches station; ends throttling"** — LS pump shuts down; LS suction rises 42→86 psig; flow drops 4000→3830 BPH; SC rises 570→572 |
| 289 | "Hold SC BP steady (generally)" |
| 300 | SC backpressure drops 570→560 |
| 302 | SC drops 560→540 |
| 307 | "Valve-controlled transfer of higher [pressure N2 to next section]" |
| 319 | **"Resume [injection]"** — N2 injection resumes after valve transfer sequence |
| 331 | SP and RY both show 79 psig — pig has passed RY; both upstream stations in N2 zone |
| 334 | Second pig marker (□, cleaning pig) appears near SH area |
| 335 | **"Compressor (or injection) at LS after [pig passes]"** — LS booster activating |
| 340 | "4000" displayed at LS booster = 40,000 SCFM injection rate |
| 343 | Flow rate SPIKES from 5357 to 7833 BPH as pig passes through critical zone |
| 362 | SP=RY=NW=SH=96 psig — all four uphill stations in N2 zone; pig has cleared SH |
| 364 | SC rises 540→564 |
| 365 | **"Pinch MT!"** — cleaning pig arriving at MT terminal; "3650" at LS = 36,500 SCFM booster injection |
| 374 | "Must cease backpressure" |
| 379 | Flow SPIKES to ~11,000 BPH — pig in critical downhill section |
| 381 | SP/RY/NW/SH all stabilize at 27 psig; **SC drops dramatically 540→382** |
| 387 | HW suction = 93 psig |
| 388 | **HW suction drops to 19 psig** — pump near cavitation limit |
| 390 | "Can open MT control valve once [pig arrives]" |
| 391–392 | "Can open MT control valve once [pig arrives]" (repeated) |
| 397 | "4000" marker at SC/HW area = N2 injection there; HW discharge ~307 psig |
| 403 | HW intermediate area: 228 psig; downstream 205 psig |
| 437 | Pig approaching MT (~MP 240); video ends |

---

## Critical Implementation Notes for Simulator

### 1. N2 Injection Events (ordered)
1. **t=212 s** — Injection begins at SP (MP 0), rate ≈ 6,600 SCFM
2. **t=242 s** — Injection **stopped** (pig ascending mountain, gas expands)
3. **t=248 s** — Injection **resumed** at SP
4. **t=263 s** — **NW booster compressor fires** (MP ~40); NW pressure rises sharply
5. **t=307 s** — "Valve-controlled transfer" — complex pressure transfer sequence
6. **t=319 s** — Injection resumed after transfer
7. **t=335 s** — **LS booster fires** (MP ~105); rate shown as "409" then "4000" (×10 = 40,000 SCFM)
8. **t=365–374 s** — LS booster running at 36,500 SCFM ("3650") through pig passage
9. **t=397 s** — Additional injection at SC/HW area ("4000" marker)

### 2. Pump Station Shutdown Sequence
- **LS pump**: Shuts down ~t=281 s (annotation says "before first pig reaches station")
  - LS suction: 42 → 86 psig (rises when pump no longer draws down suction)
  - Flow: 4000 → 3830 BPH
- **Other stations (SP, RY, NW, SH)**: Enter N2 zone as pig passes; pressures drop to ~27–96 psig

### 3. SC Backpressure Profile
| t_s | P_SC (psig) |
|-----|------------|
| 212–299 | 570 |
| 300 | 560 |
| 302–331 | 540 |
| 381 | 382 |
| 388 | 346 |
| 392 | 372 |
| 402 | 269 |
| 409 | ~181 |
| 416–424 | ~136→121 |

### 4. Flow Rate Envelope
| Phase | Flow (BPH) |
|-------|-----------|
| t=212–280 | ~4,000 (steady) |
| t=281–290 | ~3,830–3,873 (LS pump shutdown) |
| t=290–329 | 4,000–5,418 (variable) |
| t=329–363 | 4,040–8,287 (rising, highly variable) |
| t=363–378 | 6,000–8,287 |
| t=379–392 | ~10,000–11,000 (flow spike — pig in downhill cascade) |
| t=403–437 | 5,990–10,813 (declining to final) |

### 5. HW Pump Suction Extremes
| t_s | P_HW_s (psig) | Note |
|-----|--------------|------|
| 212–280 | 67 | Stable |
| 281–297 | 63–70 | Slight variation |
| 302–315 | 28–56 | Dropping |
| 329 | 109 | Spike |
| 343 | 137 | Spike (flow surge) |
| 347–365 | 71–155 | Highly variable |
| 381–390 | 93 | After LS region |
| 388 | **19** | **Near cavitation** |
| 389 | 33 | Recovering |
| 390 | 90 | |
| 391 | 84 | |

