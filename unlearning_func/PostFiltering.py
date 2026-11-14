import copy
import time

from logger import Logger

logger = Logger.get_logger("Unlearning(PostFiltering)")


class PostFiltering:
    """Post-filtering reference point.

    The model is left untouched and the outdated items are masked only in the
    output recommendation list, following the reference point in Section 5.1.3
    of the paper.  This isolates the contribution of parameter-level revision:
    ``RR = 1`` and ``RDR = 0`` because the scoring model and its parameters are
    unchanged.
    """

    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device

    def unlearn(self):
        start = time.time()
        logger.info("Applying Post-filtering (output masking only)...")
        revised_model = copy.deepcopy(self.model)
        if hasattr(revised_model, "clear_cache"):
            revised_model.clear_cache()
        elapsed = time.time() - start
        logger.info("Post-filtering complete. Time: %.2fs.", elapsed)
        return revised_model, self.data.target_edges, elapsed
