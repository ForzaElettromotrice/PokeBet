from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier

def create_logistic_regression(
        *,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 3000,
        multi_class: str = "auto",
        verbose: int = 0,
        class_weight: dict | None = None,
        random_state: int | None = None,
) -> LogisticRegression:
    """
    Return a simple sklearn LogisticRegression model configured with common defaults.

    Parameters:
    - penalty: Regularization penalty ('l1', 'l2', 'elasticnet', or 'none')
    - C: Inverse of regularization strength; smaller values specify stronger regularization.
    - solver: Algorithm to use in the optimization problem. 'liblinear' is good for small datasets, 'lbfgs' is the default.
    - max_iter: Maximum number of iterations for the solver.
    - multi_class: 'auto', 'ovr', or 'multinomial'. 'ovr' is for binary classification.
    - class_weight: Optional dict specifying class weights or 'balanced'.
    - random_state: Random seed for reproducibility when applicable.

    Returns:
    - An untrained sklearn.linear_model.LogisticRegression instance.
    """
    return LogisticRegression(
        penalty = penalty,
        C = C,
        solver = solver,
        max_iter = max_iter,
        class_weight = class_weight,
        random_state = random_state,
        verbose = verbose,
    )

def create_gradient_boosting_classifier(
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        max_features: float | int | str | None = None,
        loss: str = "log_loss",
        validation_fraction: float = 0.1,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
        verbose: int = 0,
) -> GradientBoostingClassifier:
    """
    Return a sklearn.ensemble.GradientBoostingClassifier configured with sensible defaults.

    Notes:
    - max_features replaces XGBoost's colsample_bytree (can be float for fraction, int for number of features, 'sqrt', 'log2', or None).
    - subsample controls row subsampling (1.0 = no subsampling).
    - n_iter_no_change and validation_fraction can be used for early stopping.
    """

    return GradientBoostingClassifier(
        loss = loss,
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        subsample = subsample,
        criterion = "friedman_mse",
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_depth = max_depth,
        random_state = random_state,
        max_features = max_features,
        verbose = verbose,
        validation_fraction = validation_fraction,
        n_iter_no_change = n_iter_no_change,
        tol = tol,
    )
#
# def create_xgb_classifier(
#         *,
#         n_estimators: int = 100,
#         learning_rate: float = 0.1,
#         max_depth: int = 3,
#         subsample: float = 1.0,
#         colsample_bytree: float = 1.0,
#         objective: str = "binary:logistic",
#         eval_metric: str = "logloss",
#         use_label_encoder: bool = False,
#         random_state: int | None = None,
#         verbosity: int = 0,
# ) -> "XGBClassifier":
#     """
#     Return an xgboost.XGBClassifier configured with sensible defaults.
#
#     Notes:
#     - colsample_bytree controls feature subsampling (1.0 = no subsampling).
#     - subsample controls row subsampling (1.0 = no subsampling).
#     """

    # return XGBClassifier(
    #     objective = objective,
    #     eval_metric = eval_metric,
    #     learning_rate = learning_rate,
    #     n_estimators = n_estimators,
    #     subsample = subsample,
    #     max_depth = max_depth,
    #     colsample_bytree = colsample_bytree,
    #     use_label_encoder = use_label_encoder,
    #     random_state = random_state,
    #     verbosity = verbosity,
    # )

def get_model(model: str, **kwargs) -> LogisticRegression | GradientBoostingClassifier:
    """
    Simple factory to choose a model by name and forward kwargs to the specific creator.

    Supported names:
    - LogisticRegression: "lr"
    - GradientBoostingClassifier: "gbc"

    Example:
    >>> get_model("lr", C=0.5, max_iter=200)
    """
    key = model.strip().lower()
    if key == "lr":
        return create_logistic_regression(**kwargs)
    if key == "gbc":
        return create_gradient_boosting_classifier(**kwargs)
    # if key == "xgb":
    #     return create_xgb_classifier(**kwargs)
    raise ValueError(f"Unknown model '{model}'. Supported: lr, gbc")
