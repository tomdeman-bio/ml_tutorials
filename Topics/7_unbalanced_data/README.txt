Dealing with Unbalanced Data
Tom Brettin
• Undersampling – Random, TOMEK
• Oversampling – SMOTE

Introduction to the problem.
  Consider - 
    * About 2% of credit card accounts are defrauded per year1. (Most fraud
      detection domains are heavily imbalanced.)
    * Medical screening for a condition is usually performed on a large
      population of people without the condition, to detect a small minority
      with it (e.g., HIV prevalence in the USA is ~0.4%).
    * Disk drive failures are approximately ~1% per year.
    * The conversion rates of online ads has been estimated to lie between
      10-3 to 10-6.
    * Factory production defect rates typically run about 0.1%. 

  A real-world example in cancer -
  
Undersampling

  Tomek links are pairs of instances of opposite classes who are their
  own nearest neightbors.

  NearMiss-1      selects majority examples that are close to some of the
  minority examples. Selects examples whose average distances to three
  closest minority examples are the smallest.

  NearMiss-2      selects majority examples that are close to all majority
  examples. examples are selected based on average distance to three
  farthest majority examples.

  NearMiss-3      selects a number of closest majority examples for each
  majority example. guarentees every majority example is surrounded by
  some majority examples.

Oversampling

  Synthesizing new examples: SMOTE

  Another direction of research has involved not resampling of examples,
  but synthesis of new ones. The best known example of this approach is
  Chawla’s SMOTE (Synthetic Minority Oversampling TEchnique) system. The
  idea is to create new minority examples by interpolating between
  existing ones.

  It is important to note a substantial limitation of SMOTE. Because it
  operates by interpolating between rare examples, it can only generate
  examples within the body of available examples—never outside. Formally,
  SMOTE can only fill in the convex hull of existing minority examples,
  but not create new exterior regions of minority examples.



Example of Oversampling

This example uses a LinearSVC model wihch implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained:

Precision = True Positives / (True Positives + False Positives)
Recall    = True Positives / (True Positives + False Negatives)

When assessing whether or not someting is A (we have 12 dog pictures)
False Positives are when something is predicted to be A but in fact is B
False Negatives are when something is predicted to be B but in fact is A


