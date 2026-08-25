# Contributing

RadMeasure is a research prototype, not a medical device. Contributions must
not describe outputs as diagnoses or claim clinical validation.

1. Create a focused branch and keep generated data, checkpoints, radiographs,
   credentials, and patient information out of Git.
2. Add tests for behavior changes.
3. Run `pip install -e '.[dev]'` and `pytest -q`.
4. Explain safety-policy changes and any new executable tool permissions in the
   pull request.
5. Report benchmark changes with the dataset size, split, metric definition,
   and limitations.

Security issues or suspected sensitive-data exposure should be reported
privately rather than through a public issue.
