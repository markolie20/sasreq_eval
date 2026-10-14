# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""LRURec: linear recurrent units for next-item recommendation.

Yue et al., *Linear Recurrent Units for Sequential Recommendation*, WSDM 2024.

:class:`LRURec` is the architecture of §3, :class:`LRURecConfig` its published
settings, and :class:`LRURecTrainer` the next-item objective plus the sample
construction the paper leaves to its implementation. Where the paper and the
authors' released code disagree, the code is followed and the disagreement is
noted where it is decided.
"""

from .config import LRURecConfig
from .model import LRURec
from .trainer import LRURecTrainer

__all__ = ["LRURec", "LRURecConfig", "LRURecTrainer"]
