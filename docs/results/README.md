# Result reports

These documents preserve evaluation evidence. They are not temporary notes and
they are not pipeline entry points.

| Report | Status | Main conclusion |
|---|---|---|
| [Final full-flight runs](final-run-results.md) | Current reference | `safe` is the supported default; `adaptive_experimental` is opt-in |
| [Temporal validation](temporal-validation.md) | Failed acceptance | The first adaptive correction was unstable across cold starts |
| [Confidence cross-validation](confidence-cross-validation.md) | Failed acceptance | Confidence fallback reduced risk but did not pass the predefined gate |

A failed acceptance test is useful evidence: it prevents an in-sample gain from
being misrepresented as a generally reliable improvement.
