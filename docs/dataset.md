# Dataset

## dry sources

FSD50K: general sound
ESC-50: environment sound
MUSDB: music & voice
DISCO: noise

## random spatial direction

HRTF / BRIR convolution

s_L = Σ (x_k * h_{k,L})
s_R = Σ (x_k * h_{k,R})

target-only binaural signal

## binaural sources (L/R)

for each mixture:

1. target sounds: 2
2. interfering sounds: 1-2
3. background noise: 1

SNR settings:

1. target sounds: 5–15 dB
2. interfering sounds: 0–5 dB
3. background: baseline

- mixture length = 6s
- target/interfering sound = 3–5s
- background noise = full duration
