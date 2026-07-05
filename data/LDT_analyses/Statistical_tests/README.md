# FA Statistical Test Results

Summary of spreading-activation and lexical-decision-task (LDT) analyses across association networks (Humans, Gemma, Gpt5_mini, Haiku, Llama3, Mistral).

---

## 1. Wilcoxon LDT: Related vs Unrelated Primes


| Model     | p-value | Effect |
| --------- | ------- | ------ |
| Gemma     | 1.4e-09 | 0.84   |
| Gpt5_mini | 4.5e-10 | 0.87   |
| Haiku     | 4.5e-10 | 0.87   |
| Humans    | 3.8e-10 | 0.87   |
| Llama3    | 4.0e-10 | 0.87   |
| Mistral   | 6.1e-10 | 0.86   |


**Note:** One-sample Wilcoxon signed-rank test on *related − unrelated* activation differences across 50 LDT triplets. Alternative: activation higher for related primes. All models show strong priming (related > unrelated); p-values highly significant, effect sizes large (~0.84–0.87).

---

## 2. LDT Correlation: Activation vs Reaction Time


| Model     | Spearman r | p-value |
| --------- | ---------- | ------- |
| Gemma     | −0.62      | 5.0e-12 |
| Gpt5_mini | −0.67      | 3.4e-14 |
| Haiku     | −0.66      | 6.2e-14 |
| Humans    | −0.63      | 3.1e-12 |
| Llama3    | −0.61      | 1.1e-11 |
| Mistral   | −0.61      | 1.0e-11 |


**Note:** Spearman correlation between network activation level and human reaction time (RT). Negative r = higher activation predicts *faster* RT (shorter response time). All models align with human RT patterns; Gpt5_mini and Haiku show strongest correlations.

---

## 3. Gender Female Targets: Wilcoxon (Female vs Male Primes)


| Model     | p-value | Effect |
| --------- | ------- | ------ |
| Gemma     | 2.0e-05 | 0.38   |
| Gpt5_mini | 4.8e-09 | 0.54   |
| Haiku     | 2.2e-14 | 0.70   |
| Humans    | 1.3e-13 | 0.68   |
| Llama3    | 0.12    | 0.11   |
| Mistral   | 0.002   | 0.27   |


**Note:** Wilcoxon test on (female-related primes − male-related primes) activation for female-related targets. Positive effect = stronger activation for female primes. Llama3 is non-significant; others show female-prime preference.

---

## 4. Gender Male Targets: Wilcoxon (Female vs Male Primes)


| Model     | p-value | Effect |
| --------- | ------- | ------ |
| Gemma     | 1.2e-07 | −0.46  |
| Gpt5_mini | 1.9e-15 | −0.70  |
| Haiku     | 2.0e-06 | −0.41  |
| Humans    | 4.7e-15 | −0.69  |
| Llama3    | 3.5e-16 | −0.72  |
| Mistral   | 0.013   | −0.20  |


**Note:** Same test for male-related targets. Negative effect = stronger activation for male primes (expected). Mistral is weakest; Llama3 and Gpt5_mini show strongest male-prime preference.

---

## 5. Gender Correlation with Humans: Female-Related Targets


| LLM       | Spearman r | p-value |
| --------- | ---------- | ------- |
| Gemma     | 0.60       | 4.7e-24 |
| Gpt5_mini | 0.62       | 3.7e-26 |
| Haiku     | 0.61       | 1.6e-24 |
| Llama3    | 0.31       | 2.2e-06 |
| Mistral   | 0.68       | 3.4e-32 |


**Note:** Spearman correlation between each LLM’s gender activation matrix (female-related targets) and the human network. Llama3 correlates weakly; Mistral is closest to humans.

---

## 6. Gender Correlation with Humans: Male-Related Targets


| LLM       | Spearman r | p-value |
| --------- | ---------- | ------- |
| Gemma     | 0.61       | 5.1e-27 |
| Gpt5_mini | 0.71       | 4.2e-40 |
| Haiku     | 0.62       | 5.1e-28 |
| Llama3    | 0.64       | 1.0e-30 |
| Mistral   | 0.63       | 1.9e-29 |


**Note:** Same correlation for male-related targets. All LLMs correlate significantly with humans; Gpt5_mini has the strongest alignment.

---

## 7. Pairwise Correlation Matrices (Gender Activation)

Spearman r between every pair of models (including Humans) for the flattened gender activation matrices.

**Female-related targets:**

|           | Gemma | Gpt5_mini | Haiku | Humans | Llama3 | Mistral |
|-----------|-------|-----------|-------|--------|--------|---------|
| Gemma     | 1.00  | 0.55      | 0.46  | 0.60   | 0.22   | 0.69    |
| Gpt5_mini | 0.55  | 1.00      | 0.67  | 0.62   | 0.39   | 0.58    |
| Haiku     | 0.46  | 0.67      | 1.00  | 0.61   | 0.33   | 0.43    |
| Humans    | 0.60  | 0.62      | 0.61  | 1.00   | 0.31   | 0.68    |
| Llama3    | 0.22  | 0.39      | 0.33  | 0.31   | 1.00   | 0.45    |
| Mistral   | 0.69  | 0.58      | 0.43  | 0.68   | 0.45   | 1.00    |

**Male-related targets:**

|           | Gemma | Gpt5_mini | Haiku | Humans | Llama3 | Mistral |
|-----------|-------|-----------|-------|--------|--------|---------|
| Gemma     | 1.00  | 0.74      | 0.72  | 0.61   | 0.60   | 0.70    |
| Gpt5_mini | 0.74  | 1.00      | 0.78  | 0.71   | 0.70   | 0.73    |
| Haiku     | 0.72  | 0.78      | 1.00  | 0.62   | 0.68   | 0.72    |
| Humans    | 0.61  | 0.71      | 0.62  | 1.00   | 0.64   | 0.63    |
| Llama3    | 0.60  | 0.70      | 0.68  | 0.64   | 1.00   | 0.55    |
| Mistral   | 0.70  | 0.73      | 0.72  | 0.63   | 0.55   | 1.00    |

**Note:** Rows/columns show which model’s activation pattern is correlated with which. Llama3 correlates weakly with Humans and others on female targets; all models correlate more strongly on male targets.

---

## 8. Steiger’s Z: Comparing Human–LLM Alignment

Steiger’s Z test compares two dependent correlations (e.g., Human–Gpt5 vs Human–Mistral) when both share the same reference (Humans). *p_a_more_aligned* = one-tailed p-value that model A is more aligned with Humans than model B. Small p (< 0.05) ⇒ A is significantly more aligned.

**Female targets (notable results):**

| Comparison       | r_Human_A | r_Human_B | p (more aligned) | Interpretation                                |
|------------------|-----------|-----------|------------------|-----------------------------------------------|
| Gemma vs Llama3   | 0.60 | 0.31 | 1.0e-07          | Gemma significantly more aligned than Llama3  |
| Gpt5_mini vs Llama3 | 0.62 | 0.31 | 7.5e-09        | Gpt5_mini significantly more aligned          |
| Haiku vs Llama3   | 0.61 | 0.31 | 8.0e-08         | Haiku significantly more aligned              |
| Gemma vs Mistral  | 0.60 | 0.68 | 0.97            | Mistral more aligned (p≈0.03 for Mistral)     |
| Gpt5_mini vs Mistral | 0.62 | 0.68 | 0.93          | Mistral more aligned (p≈0.07 for Mistral)     |
| Llama3 vs Mistral | 0.31 | 0.68 | 1.0             | Mistral significantly more aligned (p≈0)     |

**Male targets (notable results):**

| Comparison        | r_Human_A | r_Human_B | p (more aligned) | Interpretation                                |
|-------------------|-----------|-----------|------------------|-----------------------------------------------|
| Gemma vs Gpt5_mini   | 0.61 | 0.71 | 0.002            | Gpt5_mini significantly more aligned          |
| Gpt5_mini vs Haiku   | 0.71 | 0.62 | 0.004 | Gpt5_mini significantly more aligned          |
| Gpt5_mini vs Mistral | 0.71 | 0.63 | 0.009 | **Gpt5_mini significantly more aligned than Mistral** |
| Gpt5_mini vs Llama3  | 0.71 | 0.64 | 0.017 | Gpt5_mini significantly more aligned          |

**Note:** On female targets, Mistral is most aligned with Humans; Llama3 is least. On male targets, **Gpt5_mini is significantly more aligned with Humans than Mistral** (p ≈ 0.009). You can make model-by-model claims only within a given dimension (female vs male).

---

## Summary

- **LDT priming:** All networks show strong related > unrelated activation; effect sizes ~0.84–0.87.
- **Activation–RT link:** Negative Spearman r (around −0.61 to −0.67) indicates higher activation predicts faster RT in all models.
- **Gender biases:** Female targets favor female primes, male targets favor male primes across most models; Llama3 and Mistral show weaker or inconsistent effects.
- **Human–LLM alignment:** On female targets, Mistral is closest to humans; on male targets, Gpt5_mini is closest.
- **Statistical comparison:** Steiger’s Z confirms that Gpt5_mini is significantly more aligned with Humans than Mistral on male-related targets (p ≈ 0.009). On female targets, Mistral is more aligned than Gpt5_mini, Gemma, and Haiku (p &gt; 0.9 for “A more aligned” when A is the latter).

