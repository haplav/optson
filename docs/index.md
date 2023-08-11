```{toctree}
:maxdepth: 2
:hidden:
:glob:
Installation<installation>
Tutorials<tutorial>
API Reference<api/modules>
Source on GitLab<https://gitlab.com/swp_ethz/optson>
```

# Welcome to Optson!
**Optson** is a flexible python package for numerical optimization.
We currently support Adam, steepest descent, trust-region- and line-search L-BFGS.
What is unique about Optson is the flexibility to use all update methods with stochastic (mini-batch) optimization and (mono) batch optimization.

We achieve this by overlapping the samples of two subsequent mini-batches.
We call this intersection the control group.
This makes it possible to use the same line-search or trust-region methods in the stochastic context.
Algorithms that do not rely line-search nor trust-region methods, such as Adam can still be used without the need of a control-group.

Another useful feature is the ability to perform checkpointing and store intermediate results to disk.
This makes it possible to interrupt and resume long running optimization processes that are typical in the context of full-waveform inversion.

Please have a look at the [`Installation`](installation) and [`Tutorials`](tutorials) pages for additional information.
If you have any further questions or feature requests, feel free to reach out to the developers on Gitlab or by email.

**Dirk-Philip van Herwaarden**:
dirkphilip.vanherwaarden@erdw.ethz.ch

**Vaclav Hapla**
vaclav.hapla@erdw.ethz.ch
