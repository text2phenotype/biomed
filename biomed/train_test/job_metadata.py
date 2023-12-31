import os
from dataclasses import dataclass, field, fields, asdict
import inspect

from text2phenotype.common import common
from biomed import RESULTS_PATH


@dataclass
class JobMetadata:
    job_id: str = None
    train: bool = False
    test: bool = False
    active_learning: bool = False
    test_ensemble: bool = False
    train_ensemble_voter: bool = False
    update_model: bool = False
    k_fold_validation: bool = False
    sync_data: bool = False

    class_weight: dict = field(default_factory=dict)  # eg: PHI_WEIGHT {0:1.0, ....}
    epochs: int = 3  # iterations to train
    learning_rate: float = 0.01

    # percentage of hidden units to drop out in model for regularization
    #    For BiLSTMs, this goes directly to the LSTM layer, used in dropout and recurrent_dropout
    #    For BERT, this goes to the config.hidden_dropout_prob parameter
    dropout: float = 0.1
    model_loss: str = None
    batch_size: int = 128  # number of sequences that the model runs gradient descent on for every model update
    narrow_band: float = 0.2
    use_max: bool = False
    comparison_test_folder: str = None
    return_uncertainty: bool = False
    url_base: str = None
    dir_to_replace: str = None
    write_binary_report: bool = True
    add_dense: bool = False
    reduced_dim: int = None
    hidden_dim: int = 128  # hidden dimensions in the model
    k_fold_subfolders: list = field(default_factory=list)
    k_folds: int = None
    exclude_validation: bool = False
    max_train_failures_pct: float = 0.05
    max_test_failures_pct: float = 0.05

    # if not None, set random seed for train, giving model consistency between runs
    random_seed: int = None  # job-based random seed, to force consistency in model training
    train_embedding_layer: bool = True  # bool, if true, fine-tune the embedding layer (eg bert embeddings)
    full_output: bool = False  # if true, provide all probabilities for experimental purposes

    def __post_init__(self):
        """Do any attribute dependency changes here"""
        self.k_folds = self.k_folds or len(self.k_fold_subfolders)

    @classmethod
    def from_dict(cls, config):
        """load class from dictionary, skipping values that arent fields"""
        # filter only the keys that are fields
        d = {
            k: v for k, v in config.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(**d)

    def to_json(self):
        return asdict(self)

    def save(self):
        path = os.path.join(RESULTS_PATH, self.job_id, 'job_metadata.json')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return common.write_json(self.to_json(), path)
