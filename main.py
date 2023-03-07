import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from numpy import ComplexWarning
import warnings

from utils import *
from mode import *
import parse

warnings.simplefilter("ignore", ComplexWarning)

if __name__ == "__main__":
    args = parse.get_config()

    print(f"[+] {args.mode} mode")
    if args.mode == 'train':
        print("[+] train data load...", end=' ')
        train(args)
        test(args)
        key_train(args)
        key_test(args)

    elif args.mode == 'key_train':
        print("[+] key train data load...", end=' ')
        key_train(args)
        key_test(args)

    elif args.mode == 'test':
        print("[+] test data load...", end=' ')
        test(args)

    elif args.mode == 'key_test':
        print("[+] key test data load...", end=' ')
        key_test(args)

    elif args.mode == 'evaluate':
        print("[+] evaluate data load...", end=' ')
        evaluate(args)
        
    else:
        print("[-] Error! Wrong mode (main.py)\n")
        exit(1)
