from ImageClassification_Task.ic_trainer import ICTrainer
from utils.argparser import parse_arguments
from pprint import pprint

if __name__ == '__main__':
    args = parse_arguments()

    # This prints your configuration (Split 1, CIFAR-10, etc.)
    pprint(vars(args))

    if args.dataset == 'CIFAR10':
        # This matches your '-kv' flag
        if args.use_key_value_store:
            trainer = ICTrainer(args)
        else:
            # Note: The paper focuses on the KV store for efficiency
            raise NotImplementedError
        
        # This runs the Generalized and Personalized phases
        trainer.fit()
        # This checks the final accuracy (ESL-P)
        trainer.inference()

    else:
        print('Kaggle Note: Only CIFAR10 is available in this fork.')
        print('Invalid dataset -_-')