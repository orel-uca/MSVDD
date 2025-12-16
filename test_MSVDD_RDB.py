__author__ = 'raulpaez'
__docs__ = 'Methods for evaluating MSVDD models on real datasets'

# To evaluate Real Database (RDB) from LIBSVM, in which the size of instances varies with the percentage of anomalous data.

import os, sys, argparse
from utilities import evaluate, metrics, plot_cross_val

if __name__ == '__main__':
    
    ''' Example:
        --------
        %run test_MSVDD_RDB.py --dir_data "RDB/Data_iris" --dir_sufix "iris" --do_evaluation --do_metrics --do_cross_val --reps 1 2 --ks 1 2 --Cs 0.070 0.075 --num_train 50 52 --num_val 32 34 --num_test 83 86 --anom_frac 0.10 0.15
    '''
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MSVDD Model Parameters')
    parser.add_argument('--anom_frac', nargs='+', type=float, default=[0.1], help='Anomaly fraction (#outliers) in data generation')
    parser.add_argument('--num_train', nargs='+', type=int, help='List of training set sizes')
    parser.add_argument('--num_val', nargs='+', type=int, help='List of validation set sizes')
    parser.add_argument('--num_test', nargs='+', type=int, help='List of test set sizes')
    parser.add_argument('--use_kernel', action='store_true', help='Use kernel RBF')
    parser.add_argument('--do_evaluation', action='store_true', help='Evaluate MSVDD models')
    parser.add_argument('--do_metrics', action='store_true', help='Calculate metrics for MSVDD')
    parser.add_argument('--do_cross_val', action='store_true', help='Perform cross validation MSVDD')
    parser.add_argument('--reps', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='List of numbers of the repetitions')
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3], help='List of k values')
    parser.add_argument('--Cs', nargs='+', type=float, help='List of C values')
    parser.add_argument('--sigmas', nargs='+', type=float, help='List of sigma values')
    parser.add_argument('--dir_data', type=str, default='Data', help='Directory for input data files')
    parser.add_argument('--dir_sufix_out', type=str, default='', help='Sufix for output directories: summary and data output')
    args = parser.parse_args()
    
    # If num_train not specified, exit
    if args.num_train is None:
        sys.exit("--num_train is required")
    else:
        num_train = args.num_train
    
    # If num_val not specified, calculate from num_test as 2/3 of num_train
    if args.num_val is None:
        num_val = [int(t*2/3) for t in num_train]
    else:
        num_val = args.num_val
    
    # If num_train not specified, calculate from num_test as 5/3 of num_train
    if args.num_test is None:
        num_test = [int(t*5/3) for t in num_train]
    else:
        num_test = args.num_test

    use_kernel = args.use_kernel  
    do_evaluation = args.do_evaluation
    do_metrics = args.do_metrics
    do_cross_val = args.do_cross_val
    Cs = args.Cs
    sigmas = args.sigmas
    ks = args.ks
    reps = args.reps
    anom_frac = args.anom_frac
    
    if use_kernel:
        DorP = 'Dual'
        if sigmas is None:
            sys.exit("--sigmas is required when --use_kernel is specified")
    else:
        sigmas = [-1.0]
        DorP = 'Prim'
            
    # Create directories and store paths
    dir_data = f'{args.dir_data}'
        
    if args.dir_sufix_out:
        data_DorP = f'Data_{DorP}_{args.dir_sufix_out}'
        output_DorP = f'Summary_{DorP}_{args.dir_sufix_out}'
    else:
        data_DorP = f'Summary_{DorP}'
        output_DorP = f'Summary_{DorP}'

    if len(num_train) != len(anom_frac):
        sys.exit("num_train and anom_frac must have the same length")
    else:
        for t in range(len(num_train)):
            os.makedirs(output_DorP, exist_ok=True)
            if Cs:
                res_filename = f'{output_DorP}/res_anom_{DorP}_{num_test[t]}_{anom_frac[t]}_{len(reps)}_{len(ks)}_{len(Cs)}_{len(sigmas)}_ms.npz'
                
            if do_evaluation:
                os.makedirs(data_DorP, exist_ok=True)
                os.makedirs(output_DorP, exist_ok=True)
                evaluate(Cs, sigmas, ks, reps, num_train[t], num_test[t], num_val[t], anom_frac[t], use_kernel, dir_data, data_DorP, output_DorP) 

            if do_metrics:
                metrics(res_filename, Cs, sigmas, ks, reps, num_train[t], num_test[t], num_val[t], anom_frac[t], use_kernel, None, dir_data, data_DorP, output_DorP) 
                
            if do_cross_val:
                if not os.path.exists(res_filename):
                    print(f'Error: {res_filename} not exist. Please run do_metrics before do_cross_val.')
                    sys.exit(1)
                else:
                    plot_cross_val(res_filename)
                
    print('DONE :)')
