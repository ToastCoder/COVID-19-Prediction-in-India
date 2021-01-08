import os
import argparse

def strBool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean Value.')

def parse():
    parser = argparse.ArgumentParser(description = 'Command Line Interface for COVID-19 Prediction in India')
    parser.add_argument('-tr','--train', type = str, help = 'Argument taken for training model(s).', default = "none")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse()

    if (args.train == 'cases'):
        os.system('python3 train_cases.py')

    if (args.train == 'deaths'):
        os.system('python3 train_deaths.py')

    if (args.train == 'all'):
        os.system('python3 train_cases.py')
        os.system('python3 train_deaths.py')
    


        