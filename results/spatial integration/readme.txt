The reasons there are intermediate results saved for each step are:


subset integration: 
    So this computationally intensive task can be run (and immedeatly written to disk) independently of  other analyses.

subset ibcorr:
    Results from this stepstep are needed in their form at some point during analyses.

subset 3d pvalues:
    (see above)