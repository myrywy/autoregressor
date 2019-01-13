import datetime
import argparse 
import logging

from lm_training_process import eval_lm_on_cached_simple_examples_with_glove_check
from hparams import hparams


now = datetime.datetime.now()
time_formatted = now.strftime("%Y%m%d_%H-%M")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("logs/{}_run_lm.log".format(time_formatted)))


def format_prediction(prediciton):
    prediciton_log = """Predicted tokens:
    {}
    Predicted ids:
    {}
    Probabilities:
    {}
    """.format(prediction["predictions_ids"], prediction["predictions_ids"], prediction["probabilities"])
    return prediciton_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("cached_dataset_dir")
    parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
    parser.add_argument("--dataset_subset", default="train")
    args = parser.parse_args()


    if args.hparams:
        hparams.parse(args.hparams)
    logger.info("Running with parameters: {}".format(hparams.to_json()))
    predictions = eval_lm_on_cached_simple_examples_with_glove_check(
        args.cached_dataset_dir, 
        args.model_dir, args.dataset_subset, 
        hparams)
    
    for prediction in predictions:
        logger.info(format_prediction(prediction))
