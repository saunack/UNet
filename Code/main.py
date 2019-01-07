import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs",type=int,dest="epochs",help="number of epochs",default=100)
    parser.add_argument("-lr",type=float,dest="lr",help="learning rate",default=0.001)
    #parser.add_argument("-d","--decay",type=float,dest="decay",help="weight decay",default=0.005)
    #parser.add_argument("-m","--momentum",type=float,dest="momentum",help="learning momentum",default=0.9)
    parser.add_argument("--display", action = 'store_true')
    parser.add_argument("--save", action = 'store_true')
    parser.add_argument("--load", action = 'store_true')
    args = parser.parse_args()
    #train(args.epochs, args.lr, args.momentum, args.decay, args.display)

    train(args.epochs, args.lr, args.display, save=args.save, load=args.load)

get_options()
