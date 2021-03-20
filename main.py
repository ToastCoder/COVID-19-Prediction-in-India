#-------------------------------------------------------------------------------------------------------------------------------

# COVID19 PREDICTION IN INDIA

# FILE NAME: main.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Regression, Machine Learning, TensorFlow

#-------------------------------------------------------------------------------------------------------------------------------

# SET OF DESCRIPTIONS
description = [ 'Command Line Interface for COVID-19 Prediction in India',
                'Argument taken for training model(s).',
                'Argument taken for installing requirements',
                'Argument taken for visualizing metrics',
                'Argument for testing with custom input',
                'Argument for mentioning the number of Epochs',
                'Argument for mentioning the amount of Batch Size',
                'Argument for mentioning the Loss Function',
                'Argument for mentioning the Optimizer']

# FUNCTION TO CONVERT STR INPUT TO BOOL
def strBool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean Value.')

# FUNCTION FOR PARSING ARGUMENTS
def parse():
    parser = argparse.ArgumentParser(description = description[0])
    parser.add_argument('-tr','--train', type = str, help = description[1], default = "none")
    parser.add_argument('-req','--install_requirements', type = strBool, help = description[2], default = False)
    parser.add_argument('-v','--visualize', type = strBool, help = description[3], default = False)
    parser.add_argument('-t','--test', type = strBool, help = description[4], required = True)
    parser.add_argument('-e','--epochs',type = int, help = description[5], default = 500, required = False)
    parser.add_argument('-bs', '--batch_size',type = int, help = description[6], default = 150, required = False)
    parser.add_argument('-l','--loss',type = str, help = description[7],default = 'huber', required = False)
    parser.add_argument('-op','--optimizer', type = str, help = description[8], default = 'adamax', required = False)
    args = parser.parse_args()
    return args

# MAIN FUNCTION
if __name__ == "__main__":

    # IMPORTING REQUIRED MODULES
    import os
    import argparse
    
    # DISABLING TENSORFLOW DEBUG INFORMATION
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print("TensorFlow Debugging Information is hidden.")

    args = parse()

    if (args.install_requirements):
        os.system('sudo apt install python3-pip')
        os.system('pip3 install -r requirements.txt')
    
    if (args.train == 'cases'):
        os.system(f'python3 src/train_cases.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')

    if (args.train == 'deaths'):
        os.system(f'python3 src/train_deaths.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')

    if (args.train == 'all'):
        os.system(f'python3 src/train_cases.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')
        os.system(f'python3 src/train_deaths.py --epochs={args.epochs} --batch_size={args.batch_size} --loss={args.loss} --optimizer={args.optimizer}')

    if (args.visualize):
        os.system('python3 src/visualize.py')
    
    if (args.test):
        os.system('python3 src/test.py')
    



        
