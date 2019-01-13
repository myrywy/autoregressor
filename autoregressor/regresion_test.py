import argparse 
import logging
import pickle

from lm_training_process import eval_lm_on_cached_simple_examples_with_glove_check
from hparams import hparams

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
            assert (prediction["predictions_ids"]==expected["predictions_ids"]).all()
            assert (prediction["predictions_tokens"]==expected["predictions_tokens"]).all()
            assert (prediction["probabilities"] == expected["probabilities"]).all()
        logging.info("Compared {} examples; results are the same".format(i))
        print("Compared {} examples; results are the same".format(i))
    
    if args.mode == RTestMode.UPDATE:
        predictions = eval_lm_on_cached_simple_examples_with_glove_check(args.cached_dataset_dir, args.model_dir, "train", hparams)
        predictions = [*predictions]
        with open(RESULTS_FILE_PATH, "wb") as expected_file:
            pickle.dump(predictions, expected_file)
        logging.info("Model's output saved in {}".format(RESULTS_FILE_PATH))
        print("Model's output saved in {}".format(RESULTS_FILE_PATH))

