# Voice Verification Strategies

## Overview

The VoicePrint accent-aware voice verification system supports three different strategies for matching a user's voice against their enrolled language profiles. These strategies provide flexibility in how the system attempts to verify a user's identity when they may have enrolled in multiple languages.

---

## The Three Strategies

### 1. Best of All (`best_of_all`)

**Default Strategy**

The **Best of All** strategy provides maximum flexibility by comparing the verification audio against **all enrolled language profiles** and returning the highest matching score.

#### How It Works

```
┌────────────────────────────────────────────────────────────┐
│                    BEST OF ALL STRATEGY                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input Audio ──┬──► Compare with Hindi Profile ──► Score 1 │
│                │                                           │
│                ├──► Compare with English Profile ──► Score 2│
│                │                                           │
│                └──► Compare with Marathi Profile ──► Score 3│
│                                                            │
│                         ↓                                  │
│              Pick Highest Score (e.g., Score 2)           │
│                         ↓                                  │
│              Return: Match/No Match + Best Score           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Behavior
- Tries **ALL** enrolled languages for the user
- Returns the **highest** matching score among all comparisons
- If the best score exceeds the threshold, verification succeeds

#### Pros
- ✅ Maximum flexibility - works regardless of which enrolled language the user speaks
- ✅ Best for users who frequently switch between languages
- ✅ Highest success rate for legitimate users

#### Cons
- ⚠️ Slightly higher computational cost (compares against all profiles)
- ⚠️ May be slightly slower for users with many enrolled languages

#### When to Use
- As the default strategy
- When you want the most robust verification
- When users might speak in any of their enrolled languages

---

### 2. Accent Matched (`accent_matched`)

**Optimized for Accuracy**

The **Accent Matched** strategy first attempts to **detect the language** of the verification audio, then compares against that specific language profile. If detection fails or the detected language isn't enrolled, it falls back to the Best of All strategy.

#### How It Works

```
┌────────────────────────────────────────────────────────────┐
│                  ACCENT MATCHED STRATEGY                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input Audio ──► Detect Language (e.g., Hindi)            │
│                         ↓                                  │
│              Is Hindi enrolled? ──Yes──► Compare with      │
│                    │                      Hindi Profile    │
│                    │                           ↓           │
│                    No                    Return Score      │
│                    ↓                                       │
│              Fallback to Best of All                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Behavior
1. **Step 1**: Detect the language/accent of the input audio
2. **Step 2**: If the detected language is enrolled, compare against that profile
3. **Step 3**: If match found, return result
4. **Step 4**: If detection uncertain or language not enrolled, fallback to `best_of_all`

#### Pros
- ✅ Fastest when accent detection is accurate
- ✅ Single comparison when detection works
- ✅ Still reliable due to fallback mechanism

#### Cons
- ⚠️ Depends on accuracy of accent detection model
- ⚠️ May fall back to best_of_all frequently if detection is uncertain

#### When to Use
- When accent detection is highly accurate for your user base
- When you want to optimize for speed in common cases
- When users typically speak in consistent languages

---

### 3. Declared Language Fallback (`declared_language_fallback`)

**Respects User Preference Order**

The **Declared Language Fallback** strategy honors the user's declared language preferences by trying languages in the exact order they were specified during enrollment: **primary → secondary → optional**.

#### How It Works

```
┌────────────────────────────────────────────────────────────┐
│             DECLARED LANGUAGE FALLBACK STRATEGY            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  User enrolled with:                                       │
│    Primary: Hindi (hi-IN)                                  │
│    Secondary: English (en-IN)                              │
│    Optional: Marathi (mr-IN)                               │
│                                                            │
│  Input Audio ──► Compare with Hindi ──► Match? ──Yes──► ✓  │
│                         │                                  │
│                         No                                 │
│                         ↓                                  │
│              Compare with English ──► Match? ──Yes──► ✓    │
│                         │                                  │
│                         No                                 │
│                         ↓                                  │
│              Compare with Marathi ──► Match? ──Yes──► ✓    │
│                         │                                  │
│                         No                                 │
│                         ↓                                  │
│                   Return: No Match                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Behavior
1. Try the **primary language** first
2. If no match, try the **secondary language**
3. If no match, try **optional languages** (if any)
4. If all fail, return no match

#### Pros
- ✅ Respects user's declared language preferences
- ✅ Predictable behavior based on enrollment order
- ✅ Can short-circuit early if primary language matches

#### Cons
- ⚠️ May fail if user speaks in a non-primary language with lower scores
- ⚠️ Less flexible than best_of_all

#### When to Use
- When you want to respect the user's language preference hierarchy
- In applications where primary language usage is expected
- When predictable, order-based verification is preferred

---

## Comparison Table

| Aspect | Best of All | Accent Matched | Declared Order |
|--------|-------------|----------------|----------------|
| **Flexibility** | Highest | High (with fallback) | Medium |
| **Speed** | Slower (all profiles) | Fast when detection works | Medium (ordered checks) |
| **Accuracy** | Highest match potential | Depends on detection | Depends on order |
| **Fallback** | N/A | Falls back to best_of_all | Returns failure |
| **Best For** | General use | Optimized scenarios | Preference-based apps |

---

## Technical Implementation

The strategies are implemented in the `VerificationPolicy` class:

```python
class VerificationStrategy(str, Enum):
    ACCENT_MATCHED = "accent_matched"
    DECLARED_LANGUAGE_FALLBACK = "declared_language_fallback"
    BEST_OF_ALL = "best_of_all"
    DUAL_MODEL_FUSION = "dual_model_fusion"  # Advanced option
```

### API Usage

When calling the verification endpoint, specify the strategy:

```bash
curl -X POST http://localhost:8000/api/v1/voice/accent/verify \
  -F "user_id=user_123" \
  -F "strategy=best_of_all" \
  -F "audio=@verification.wav"
```

### Response Fields

All strategies return:
- `matched`: Boolean indicating verification success
- `score`: Final similarity score
- `strategy_used`: The strategy that was actually applied
- `matched_language`: Which language profile matched (if any)
- `confidence_level`: very_high / high / medium / low

---

## Recommendations

1. **Default to `best_of_all`** for maximum reliability
2. Use **`accent_matched`** when you have confidence in language detection and want faster verification
3. Use **`declared_language_fallback`** when user language preferences are important to your application

---

## Key Principles

> **User-declared language is authoritative** — The system never overrides the user's language choice during enrollment

> **Accent detection is advisory** — Used for optimizing verification routing, not for making final decisions

> **Language isolation** — Embeddings from different languages are never mixed; each language has its own voice profile
