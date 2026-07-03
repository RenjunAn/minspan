# Reference data

`agentdyn_appendix_g.csv` — published AgentDyn results (arXiv 2602.03117v3,
Appendix G, Table 17): one row per `(model, defense, metric, suite)` cell for
the 12 published backends. Used by `scripts/collect_agentdyn.py` as the
baseline against which our DeepSeek runs are added.

To re-extract from the source paper:

```bash
curl -sL "https://arxiv.org/html/2602.03117v3" -o _paper.html
python parse_agentdyn_appendix_g.py
```
