# AI assistance disclosure

This project was developed with substantial OpenAI Codex assistance. The assistant helped write and revise preprocessing, CSP and band-power models, evaluation, prediction, tests and documentation. It ran experiments on the supplied public dataset and helped interpret results. This is not presented as unaided work.

The user provided the recruitment requirements and asked questions about event semantics, preprocessing, CSP, splitting and scores. The applicant should review the implementation and explain it in their own words before submission.

During development, numerical runtime warnings prompted changes to covariance regularization, feature scaling, channel selection and decision-score computation. Some early explanations proposed rank deficiency, units or outliers as causes without isolating the underlying runtime cause. Those explanations are hypotheses, not demonstrated findings. A clean run is a software check, not scientific validation.

Results in the report come from saved experiment outputs, not invented example accuracies. The final near-chance result and timing-only comparator are reported rather than hidden. No claim is made that the model is clinically useful, online-ready, or a reliable neural decoder.
