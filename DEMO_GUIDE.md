# FaceLock — Live Demonstration & Presentation Guide

This guide provides a structured **3 to 5 minute live presentation script** designed for hackathons, college final project reviews, portfolio demos, and technical interviews.

---

## Presentation Timeline Overview

| Timestamp | Phase | Key Action & Talking Points |
| :---: | :--- | :--- |
| **0:00** | **Introduction** | State the problem: traditional password vulnerabilities, account takeover, and weaknesses in unauthenticated facial systems. |
| **0:30** | **Registration** | Press **`[U]`** to enroll a new user. Explain 15-frame centroid aggregation and AES-128 template encryption. |
| **1:15** | **Liveness** | Press **`[A]`** to demonstrate the anti-spoofing challenge. Explain Eye Aspect Ratio (EAR) blink detection. |
| **2:00** | **Authentication** | Blink naturally twice. Show instant 1:N identity resolution and distance metric on the HUD. |
| **2:30** | **Password Reset** | Press **`[R]`** to demonstrate quick facial credential reset for the primary account. |
| **3:00** | **Spoof Attempt** | Present a printed photo or phone screen (or remain motionless without blinking) to show spoof rejection. |
| **3:30** | **Evaluation Dashboard** | Run `python -m evaluation.evaluator` in terminal. Show ROC curve, FAR/FRR trade-off, and EER. |
| **4:00** | **Architecture & Security** | Conclude with architecture diagram, zero-plaintext storage, and privacy-preserving audit logs. |

---

## Detailed Step-by-Step Walkthrough

### 0:00 — Introduction (30 seconds)
- **Presenter**: *"Welcome everyone. Today I'm presenting FaceLock — an AI-powered facial biometric authentication and credential recovery system."*
- **Context**: *"Traditional passwords suffer from phishing, credential stuffing, and forgotten logins. However, naive facial recognition is vulnerable to static photo spoofing and insecure plaintext template storage. FaceLock solves this with a 4-tier pipeline: multi-sample quality enrollment, temporal blink anti-spoofing, AES-encrypted template persistence, and measurable threshold evaluation."*

---

### 0:30 — Registration & Centroid Enrollment (45 seconds)
1. Launch FaceLock:
   ```powershell
   python main.py
   ```
2. The HUD opens displaying `STATE: IDLE` and hotkey controls.
3. Press **`[U]`** to enroll a new user (e.g., `User 1`).
4. **Talking Points**:
   - *"Notice how FaceLock doesn't just take one photo. It collects 15 valid frames across a temporal window."*
   - *"Each frame is checked for exactly one face $\ge 60\text{px}$. The system averages the 15 vectors into a central centroid and normalizes it to the unit hypersphere."*
   - *"The template is immediately encrypted with AES-128-CBC + HMAC-SHA256 before writing to disk."*

---

### 1:15 — Liveness & Anti-Spoofing Challenge (45 seconds)
1. Press **`[A]`** to start authentication.
2. The HUD transitions to `STATE: LIVENESS_CHECK` with prompt *"Please blink your eyes naturally (0/2 complete)"*.
3. Point out the live telemetry card:
   - **`LIVENESS: CHALLENGE ACTIVE`**
   - **`EAR: 0.32 [OPEN]`**
4. **Talking Points**:
   - *"We calculate the Eye Aspect Ratio (EAR) using 6 facial landmarks per eye. When open, EAR is ~0.30; when closed, it drops below 0.22."*
   - *"To defeat static photos, the system enforces a strict open $\to$ closed $\to$ open transition sequence."*

---

### 2:00 — Authentication & 1:N Identification (30 seconds)
1. Blink naturally twice within the 8.0-second window.
2. The HUD immediately flashes `STATE: SUCCESS` with an emerald green card:
   - **`LIVENESS: ✓ VERIFIED`**
   - **`IDENTITY: ✓ MATCHED (User 1)`**
   - **`DISTANCE: 0.3814 (Threshold: <= 0.60)`**
   - **`STATUS: AUTHENTICATED`**
3. **Talking Points**:
   - *"Liveness passes, candidate vector is extracted and scanned across all enrolled encrypted profiles in 1:N Euclidean space in under 50 milliseconds."*

---

### 2:30 — Credential Reset (30 seconds)
1. Press **`[R]`** to demonstrate quick facial credential reset.
2. The HUD captures 15 new frames, computes the new biometric centroid, and updates `primary_user`.
3. **Talking Points**:
   - *"In a production account recovery scenario, this replaces insecure SMS OTPs with authenticated biometric ownership verification."*

---

### 3:00 — Spoof Attack Simulation (30 seconds)
1. Press **`[A]`** again.
2. Stare directly at the camera without blinking, or hold up a printed photo/phone portrait.
3. After 8.0 seconds, the challenge times out:
   - HUD card turns crimson red: `STATE: DENIED`
   - **`LIVENESS: ✗ FAILED (SPOOF / TIMEOUT)`**
   - **`STATUS: ACCESS DENIED`**
4. Check the terminal / audit log:
   - An immutable security event `LIVENESS_FAILURE` is logged.

---

### 3:30 — Evaluation Dashboard & Metrics (30 seconds)
1. In the terminal, run the evaluation CLI:
   ```powershell
   python -m evaluation.evaluator
   ```
2. Display the generated terminal report:
   - **Equal Error Rate (EER)**: Operating balance point where $\text{FAR} = \text{FRR}$.
   - **ROC AUC**: Discrimination power across threshold sweeps.
3. Open `evaluation/plots/roc_curve.png` and `evaluation/plots/distance_distributions.png`.
4. **Talking Points**:
   - *"We don't guess our threshold. Our evaluation module sweeps thresholds from 0.20 to 0.80 across genuine and impostor distance distributions to empirically validate $\tau = 0.60$."*

---

### 4:00 — Architecture, Privacy & Conclusion (30 seconds)
1. Press **`[Q]`** to close the camera cleanly.
2. Open `docs/ARCHITECTURE.md` or `docs/SECURITY.md`.
3. **Key Takeaways**:
   - **Zero Plaintext Passwords or Raw Embeddings**: Templates encrypted with AES/Fernet at rest.
   - **Privacy-First Audit Logging**: Biometric vectors and keys are automatically sanitized from logs.
   - **Tested & Validated**: 95 unit, security, and integration tests ensuring resilience against path traversal, SQL injection, and corrupted templates.
4. *"Thank you! I am now happy to answer any technical questions."*
