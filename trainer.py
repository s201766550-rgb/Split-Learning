from ImageClassification_Task.ic_trainer import ICTrainer
from utils.argparser import parse_arguments
from pprint import pprint

if __name__ == '__main__':
    args = parse_arguments()

    pprint(vars(args))

    if args.dataset == 'CIFAR10':
        if args.use_key_value_store:
            trainer = ICTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    else:
        print('invalid dataset -_-')
