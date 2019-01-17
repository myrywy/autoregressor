import argparse 
import logging
import pickle

import numpy as np
from pytest import approx

from lm_training_process import eval_lm_on_cached_simple_examples_with_glove_check
from hparams import hparams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RTestMode:
    UPDATE = "update"
    CHECK = "check"

RESULTS_FILE_PATH = "rtest_expected.pickle"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[RTestMode.CHECK, RTestMode.UPDATE])
    parser.add_argument("model_dir")
    parser.add_argument("--cached_dataset_dir")
    parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()


    if args.hparams:
        hparams.parse(args.hparams)
    predictions = eval_lm_on_cached_simple_examples_with_glove_check(args.cached_dataset_dir, args.model_dir, "train", hparams)

    if args.mode == RTestMode.CHECK:
        with open(RESULTS_FILE_PATH, "rb") as expected_file:
            expected_predictions = pickle.load(expected_file)
        
        for i, (prediction, expected) in enumerate(zip(predictions, expected_predictions)):
            try:
                assert np.allclose(prediction["probabilities"], expected["probabilities"],atol=0.0001,rtol=0) # Using approx here is EXTREMALLY inefficient for some reason 
                assert (prediction["predictions_ids"]==expected["predictions_ids"]).all()
                assert (prediction["predictions_tokens"]==expected["predictions_tokens"]).all()
            except:
                import pdb; pdb.set_trace()
                raise
        logger.info("Compared {} examples; results are the same".format(i))
    
    if args.mode == RTestMode.UPDATE:
        predictions = eval_lm_on_cached_simple_examples_with_glove_check(args.cached_dataset_dir, args.model_dir, "train", hparams)
        predictions = [*predictions]
        with open(RESULTS_FILE_PATH, "wb") as expected_file:
            pickle.dump(predictions, expected_file)
        logger.info("Model's output saved in {}".format(RESULTS_FILE_PATH))
