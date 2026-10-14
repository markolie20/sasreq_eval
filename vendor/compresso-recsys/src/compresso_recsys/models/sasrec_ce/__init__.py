# Modified for seqrec_eval: differs from upstream compresso-recsys; see vendor/compresso-recsys/VENDORED.md
"""SASRecCE: SASRec trained with a full softmax and input augmentation.

A second SASRec, kept apart from :mod:`compresso_recsys.models.sasrec` so that
one stays exactly as published. :class:`SASRecCE` is the same network;
:class:`SASRecCETrainer` replaces the binary sampled objective with cross
entropy over the whole catalog -- the change Klenitskiy and Vasilev, *Turning
Dross Into Gold Loss: is BERT4Rec really better than SASRec?* (RecSys 2023),
call SASRec+ -- keeps the user's own history out of that softmax unless told
otherwise, and augments training by masking inputs and, optionally, by
inserting or substituting co-occurring items with a popularity or tail bias.
:mod:`.augmentation` holds the augmentation stages and says where new ones
belong.
"""

from .config import SASRecCEConfig
from .model import SASRecCE
from .trainer import SASRecCETrainer

__all__ = ["SASRecCE", "SASRecCEConfig", "SASRecCETrainer"]
