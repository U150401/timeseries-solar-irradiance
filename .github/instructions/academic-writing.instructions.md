---
applyTo: "{paper,presentation}/**/*.md"
---

# Academic Writing Standards

## Paper Structure (IMRaD)

1. **Abstract** (150-300 words)
2. **Introduction** - Problem, motivation, contributions
3. **Background** - Key concepts
4. **Related Work** - Literature review
5. **Methodology** - Approach, implementation
6. **Results** - Findings with figures
7. **Discussion** - Interpretation, limitations
8. **Conclusions** - Summary, future work

## Citations

Use BibTeX keys from `references.bib`:

```markdown
As shown by [@author2024title], the method achieves...
```

## Figures

```markdown
![Caption describing the figure](figures/fig1.png)

*Figure 1: Detailed caption.*
```

## Tables

```markdown
| Method | Metric 1 | Metric 2 |
|--------|----------|----------|
| Ours   | **95.2** | **0.12** |
| Prior  | 92.1     | 0.18     |

*Table 1: Comparison. Bold = best.*
```

## Writing Style

- Use active voice: "We propose" not "It is proposed"
- Be concise and precise
- Avoid AI-sounding phrases (run `/humanizer` after writing)
