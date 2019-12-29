
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    default='./checkpoint.pth',
                    help='Set checkpoint dir')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg19',
                    help='Choose architecture')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.01,
                    type=float,
                    help='Set learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    default=1024,
                    type=int,
                    help='Set hidden units number')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    default=8,
                    type=int,
                    help='Set epochs')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results
