import argparse
from .utils import *
from .example_generator import *
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, default='text-davinci-003',
                        help='The name of the model to test')
    parser.add_argument('--questionnaire', required=True, type=str, default='PANAS',
                        help='Comma-separated list of questionnaires.')
    parser.add_argument('--emotion', required=True, type=str, default='ALL',
                        help='Comma-separated list of emotions.')
    parser.add_argument('--select-count', type=int, default=999,
                        help='Numbers of situations to select per factor. Defaults to 999 (select all situations).')
    parser.add_argument('--default-shuffle-count', required=True, type=int, default=0,
                        help='Numbers of different orders in Default Emotion Measures. If set zero, run only the original order. If set n > 0, run the original order along with its n permutations. Defaults to zero.')
    parser.add_argument('--emotion-shuffle-count', required=True, type=int, default=0,
                        help='Numbers of different orders in Evoked Emotion Measures. If set zero, run only the original order. If set n > 0, run the original order along with its n permutations. Defaults to zero.')
    parser.add_argument('--test-count', required=True, type=int, default=1,
                        help='Numbers of runs for a same order. Defaults to one.')
    parser.add_argument('--mbti', type=str, default='infp',
                        help='The MBTI type of the model. Defaults to infp.')
    parser.add_argument('--significance-level', type=float, default=0.01,
                        help='The significance level for testing the difference of means between human and LLM. Defaults to 0.01.')
    parser.add_argument('--mode', type=str, default='auto',
                        help='For debugging.')

    
    args = parser.parse_args()

    run_emotionbench(args, example_generator)
