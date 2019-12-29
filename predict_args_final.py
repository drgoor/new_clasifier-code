import argparse


parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store')

parser.add_argument('checkpoint', action='store')

parser.add_argument('--top_k ', action='store',
                    dest='top_k',
                    default=5,
                    type=int,
                    help='Return top K most likely classes')

parser.add_argument('--category_names ', action='store',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='Use a mapping of categories to real names')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results
