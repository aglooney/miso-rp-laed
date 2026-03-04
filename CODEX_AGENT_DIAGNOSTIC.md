# Ramp Stress Diagnostic Instructions

## Purpose

Some scenarios currently produce **no binding ramp constraints**, even under tighter ramp limits. When ramp constraints are not binding, the dual variables

\[
\overline{\mu}_{g,t},\ \underline{\mu}_{g,t}
\]

are zero, causing

\[
\pi^{TLMP}_{g,t} = \pi^{LA}_t
\]

and eliminating any difference between TLMP and LMP.

Before running price volatility analysis, the agent must perform **ramp stress diagnostics** to determine whether ramp constraints are capable of binding.

---

## Step 1 — Compute Net Load Ramp Requirements

For the demand trajectory \(d_t\):

\[
\Delta d_t = d_t - d_{t-1}
\]

Compute the following statistics:

```python
delta_d = demand[1:] - demand[:-1]

max_abs_delta_d = max(abs(delta_d))
mean_abs_delta_d = mean(abs(delta_d))
```

Store:

```
max_abs_delta_d
mean_abs_delta_d
```

---

## Step 2 — Compute Total System Ramp Capability

For ramp regime multiplier \(\alpha\):

\[
R_g^{(\alpha)} = \alpha R_g
\]

Compute total system ramp capability:

\[
R_{sys} = \sum_g R_g^{(\alpha)}
\]

Log:

```
sum_ramp_capability
max_generator_ramp
```

---

## Step 3 — Compare Required Ramp vs Available Ramp

Compute ratio:

\[
\rho = \frac{\max |\Delta d_t|}{R_{sys}}
\]

Interpretation:

| Ratio (\(\rho\)) | Expected Ramp Behavior        |
| --------------: | ----------------------------- |
| < 0.1           | No ramp constraints will bind |
| 0.1–0.4         | Rare ramp binding             |
| 0.4–0.8         | Frequent ramp binding         |
| > 1.0           | Ramp infeasible without slack |

Log:

```
ramp_stress_ratio
```

If

```
ramp_stress_ratio < 0.1
```

print warning:

```
WARNING: Demand trajectory does not stress ramp limits.
TLMP will collapse to LMP.
```

---

## Step 4 — After Solving Dispatch

After each LAED solve compute actual ramp usage:

\[
\Delta p_{g,t} = p_{g,t} - p_{g,t-1}
\]

Compute:

```python
max_ramp_usage = max(abs(delta_p))
```

Log:

```
max_ramp_usage
max_ramp_limit
```

---

## Step 5 — Detect Binding Ramp Constraints

For ramp constraints:

\[
p_{g,t} - p_{g,t-1} \le R_g
\]

\[
p_{g,t-1} - p_{g,t} \le R_g
\]

Count binding constraints:

```
count_binding_ramps
fraction_binding_ramps
```

Definition of binding:

```
constraint_slack < 1e-6
```

---

## Step 6 — Check TLMP Activity

Compute TLMP adjustment term:

\[
A_{g,t} =
(\overline{\mu}_{g,t}-\underline{\mu}_{g,t})
-
(\overline{\mu}_{g,t+1}-\underline{\mu}_{g,t+1})
\]

Log:

```
mean_abs_TLMP_adjustment
max_abs_TLMP_adjustment
```

If

```
max_abs_TLMP_adjustment == 0
```

print:

```
WARNING: TLMP adjustments are zero. Ramp constraints not binding.
```

---

## Step 7 — Diagnostic Output File

Each scenario must write:

```
ramp_diagnostics.json
```

Example contents:

```json
{
  "max_abs_delta_d": 52.1,
  "sum_ramp_capability": 780.0,
  "ramp_stress_ratio": 0.067,
  "max_ramp_usage": 40.2,
  "max_ramp_limit": 120.0,
  "count_binding_ramps": 0,
  "fraction_binding_ramps": 0.0,
  "max_abs_TLMP_adjustment": 0.0
}
```

---

## Step 8 — Runner Behavior

If **no ramp constraints bind in all scenarios**, the runner must print:

```
RAMP DIAGNOSTIC: No ramp scarcity detected.
Consider reducing ramp factors or increasing load ramp rate.
```

---

## Expected Outcome

After these diagnostics, at least one scenario should show:

```
fraction_binding_ramps > 0
max_abs_TLMP_adjustment > 0
```

Only those scenarios will produce meaningful TLMP volatility differences.

