import os
import time
import random
import argparse
import numpy as np


# def gen_bash(args, timestamp, config, val, cal, adv):
#     bash = ['#!/bin/bash', f"cd ~/{args.dir_name}"]
#     bash.append(f"python main.py --automl --no_bar --timestamp {timestamp} {config}")
#     if val == '1':
#         bash.append(f"python main.py --automl --no_bar --checkpoint {timestamp} --validate")
#     if cal == '1':
#         bash.append(f"python main.py --automl --no_bar --checkpoint {timestamp} --calibrate")
#     if adv == '1':
#         bash.append(f"python main.py --automl --no_bar --checkpoint {timestamp} --attack --attack_ei 4 --attack_iter 1 --attack_eps 4")
#         bash.append(f"python main.py --automl --no_bar --checkpoint {timestamp} --attack --attack_ei 1 --attack_iter 10 --attack_eps 4")
#     return bash

#根据参数生成bash指令
def gen_bash(args, timestamp, config, val, cal, adv):
    bash = ['#!/bin/bash', f"cd ~/{args.dir_name}"]
    bash.append(f"python learn.py --automl  {timestamp} {config}")
    return bash

#读取csv，生成对应的bash和时间戳
def initialize(args):
    ''' generate experiments '''
    configs = np.loadtxt(f"{args.config_name}.csv", dtype=str, delimiter=',')
    timestamp_mat = [['timestamp'] * args.repeats]
    for i in range(1, len(configs)):
        print(f"generating config #{i}, total #{len(configs)-1}")
        timestamps = list()
        val = configs[i][0]
        cal = configs[i][1]
        adv = configs[i][2]
        config = list()
        for j in range(3, len(configs[0])):
            if configs[i][j].lower() == 'true':
                config.append(f"--{configs[0][j]}")
            elif configs[i][j].lower() == 'false':
                continue
            else:
                config.append(f"--{configs[0][j]} {configs[i][j]}")
        config = ' '.join(config)
        for t in range(args.repeats):
            timestamp = str(int(time.time())) + format(random.randint(0, 1000), '03')
            timestamps.append(timestamp)
            bash = gen_bash(args, timestamp, config, val, cal, adv)
            np.savetxt(os.path.join('shells', f"{args.config_name}-{i}-{t+1}.sh"), bash, fmt='%s', delimiter=None)
            time.sleep(1)
        timestamp_mat.append(timestamps)
    timestamp_mat = np.array(timestamp_mat)
    configs = np.concatenate((configs, timestamp_mat), axis=1)
    np.savetxt(f"{args.config_name}-run.csv", configs, fmt='%s', delimiter=',')
    ''' generate shell '''
    unavailable = '-x dell-gpu-04,dell-gpu-11,dell-gpu-16,dell-gpu-24,dell-gpu-29,dell-gpu-32'
    sugon = '-p sugon' if args.sugon else ''
    shell = ['#!/bin/bash', f"cd ~/{args.dir_name}/shells", 'chmod +x ../autorun.sh']
    for i in range(args.gresnum):
        shell.append(f"sbatch --gres=gpu {sugon} {unavailable} ../autorun.sh {i*10} {args.gresnum*10} {args.config_name} {args.repeats} {args.dir_name}")
    np.savetxt(f"{args.config_name}.sh", shell, fmt='%s', delimiter=None)
    print(f"generating completed, totally {(len(configs)-1)*args.repeats} trials.")  #n组参数*每组参数重复的次数


def allocate(args):
    configs = np.loadtxt(f"{args.config_name}.csv", dtype=str, delimiter=',')
    for i in range(1, len(configs)):
        for t in range(args.repeats):
            if os.path.exists(os.path.join('shells', f"{args.config_name}-{i}-{t+1}.sh")):
                os.rename(os.path.join('shells', f"{args.config_name}-{i}-{t+1}.sh"), os.path.join('shells', f"{args.config_name}-{i}-{t+1}-running.sh"))
                print(f"{args.config_name}-{i}-{t+1}-running.sh")
                return
    print('Finished')


#记录实验结果
def finish(args):
    configs = np.loadtxt(f"{args.config_name}-run.csv", dtype=str, delimiter=',')
    if args.nlp:
        log_mat = [['top1 acc', 'top5 acc', 'test NLL', 'duration'] * args.repeats]
    else:
        log_mat = [['top1 error', 'top5 error', 'test NLL', 'duration'] * args.repeats]
    log_mat[0] += ['erm train', 'erm test'] * args.repeats + ['ece'] * args.repeats
    log_mat[0] += ['fgsm'] * args.repeats + ['pgd'] * args.repeats
    for i in range(1, len(configs)):
        print(f"processing config #{i}, total #{len(configs)-1}")
        val = configs[i][0]
        cal = configs[i][1]
        adv = configs[i][2]
        log_mat.append(['-'] * args.repeats * 9)
        for t in range(args.repeats):
            timestamp = configs[i][-args.repeats+t]
            result = np.loadtxt(os.path.join('autologs', f"{timestamp}_train.log"), dtype=float)
            log_mat[i][4*t] = f"{result[0]:.2f}"
            log_mat[i][4*t+1] = f"{result[1]:.2f}"
            log_mat[i][4*t+2] = f"{result[2]:.4f}"
            log_mat[i][4*t+3] = f"{int(result[3])}"
            if val == '1':
                val_result = np.loadtxt(os.path.join('autologs', f"{timestamp}_val.log"), dtype=float)
                log_mat[i][4*args.repeats+2*t] = f"{val_result[0]:.4f}"
                log_mat[i][4*args.repeats+2*t+1] = f"{val_result[1]:.4f}"
            if cal == '1':
                cal_result = np.loadtxt(os.path.join('autologs', f"{timestamp}_cal.log"), dtype=float)
                log_mat[i][6*args.repeats+t] = f"{cal_result:.4f}"
            if adv == '1' and (not args.nlp):
                adv_fgsm_result = np.loadtxt(os.path.join('autologs', f"{timestamp}_adv_fgsm.log"), dtype=float)
                log_mat[i][7*args.repeats+t] = f"{adv_fgsm_result:.2f}"
                adv_pgd_result = np.loadtxt(os.path.join('autologs', f"{timestamp}_adv_pgd.log"), dtype=float)
                log_mat[i][8*args.repeats+t] = f"{adv_pgd_result:.2f}"
    log_mat = np.array(log_mat)
    configs = np.concatenate((configs, log_mat), axis=1)
    np.savetxt(f"{args.config_name}-fin.csv", configs, fmt='%s', delimiter=',')
    print(f"collecting completed, totally {(len(configs)-1)*args.repeats} results.")


def inspect(args):
    clipstr = lambda x, maxlen: x if len(x) <= maxlen else x[0:maxlen-2] + '..'
    configs = np.loadtxt(f"{args.config_name}-run.csv", dtype=str, delimiter=',')
    while True:
        width, height = os.get_terminal_size().columns, os.get_terminal_size().lines
        if width < 70:
            print('Too small width!!!')
            continue
        if height < 16:
            print('Too small height!!!')
            continue
        multicol = width // 50
        maxdisplay = (height - 14) * multicol
        stats = list()
        results = list()
        start_time, end_time = 0, 0
        displayed, succeed, running, pending = 0, 0, 0, 0
        for i in range(1, len(configs)):
            settings = dict()
            for j in range(3, len(configs[0])-args.repeats):
                settings[configs[0][j]] = configs[i][j]
            n_epoch = int(settings['num_epoch']) if 'num_epoch' in settings else 200
            dataset = clipstr(settings['dataset'] if 'dataset' in settings else 'cifar10', 9)
            model = clipstr(settings['model'] if 'model' in settings else 'preactresnet18', 7)
            method = clipstr(settings['method'] if 'method' in settings else 'base', 8)
            if args.nlp:
                l_tra, s_tra, l_val, s_val, s_fin = 99.0, 0.0, 99.0, 0.0, 0.0
            else:
                l_tra, s_tra, l_val, s_val, s_fin = 99.0, 100.0, 99.0, 100.0, 100.0
            for t in range(args.repeats):
                timestamp = configs[i][-args.repeats+t]
                try:
                    autolog = np.loadtxt(os.path.join('autologs', f"{timestamp}_running.log"), dtype=float)
                    epoch = int(autolog[0])
                    if epoch == n_epoch:
                        status = 'SUCCEED'
                        succeed += 1
                        try:
                            autores = np.loadtxt(os.path.join('autologs', f"{timestamp}_train.log"), dtype=float)
                            s_fin = max(s_fin, autores[0]) if args.nlp else min(s_fin, autores[0])
                        except Exception:
                            pass
                    else:
                        status = 'RUNNING'
                        running += 1
                    ratio = int(epoch * 10 / n_epoch)
                    rawdur = int(autolog[2] - autolog[1])
                    start_time = min(start_time, autolog[1]) if start_time else autolog[1]
                    end_time = max(end_time, autolog[2]) if end_time else autolog[2]
                    l_tra = min(l_tra, autolog[3])
                    s_tra = max(s_tra, autolog[4]) if args.nlp else min(s_tra, autolog[4])
                    l_val = min(l_val, autolog[5])
                    s_val = max(s_val, autolog[6]) if args.nlp else min(s_val, autolog[6])
                except:
                    epoch, ratio, rawdur = 0, 0, 0
                    status = 'PENDING'
                    pending += 1
                dur = f"{rawdur//3600:3d}h{rawdur%3600//60:02d}m{rawdur%60:02d}s"
                if displayed < maxdisplay:
                    displayed += 1
                    stats.append(f"{i:3d} {t+1:4d} {status:>8s} [{'>'*ratio}{' '*(10-ratio)}] {epoch:3d}/{n_epoch:3d} {dur:>10s}")
            results.append((s_fin, s_val, f"{i:3d} {dataset:>9s} {model:>7s} {method:>8s} {l_tra:7.4f} {s_tra:7.2f} {l_val:7.4f} {s_val:7.2f} {s_fin:6.2f}"))
        overall_dur = int(end_time - start_time)
        overall_dur = f"{overall_dur//3600}h{overall_dur%3600//60:02d}m{overall_dur%60:02d}s"
        results.sort(reverse=args.nlp)
        os.system('clear')
        print(f"Config name: {args.config_name}, Repeats: {args.repeats}, Duration: {overall_dur}, Refresh: {args.interval}")
        print(f"Trials: {(len(configs)-1)*args.repeats}, Displayed: {displayed}, Succeed: {succeed}, Running: {running}, Pending: {pending}")
        print('||'.join(['ID | No | Status | ProcessBar | Epoch | Duration '] * multicol))
        for i, line in enumerate(stats):
            print(line, end='')
            if (i+1) % multicol == 0 or i+1 == len(stats):
                print()
            else:
                print('||', end='')
        print('ID | Dataset | Model | Method | L_tra | S_tra | L_val | S_val | S_fin', end='')
        for i in range(min(len(results), 10)):
            print('\n'+results[i][2], end='')
        time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_name', type=str, default=None, help='Configuration of experiments.')
    parser.add_argument('--repeats', type=int, default=3, help='Repeat each experiment x times.')
    parser.add_argument('--gresnum', type=int, default=16, help='Number of GPU resources.')
    parser.add_argument('--dir_name', type=str, default='alpsmixup', help='Path to the main folder.')
    parser.add_argument('--allocate', default=False, action='store_true', help='Allocate experiments.')
    parser.add_argument('--finish', default=False, action='store_true', help='Collect experimental results.')
    parser.add_argument('--inspect', default=False, action='store_true', help='Inspect experiment performances.')
    parser.add_argument('--interval', type=int, default=90, help='Inspect interval.')
    parser.add_argument('--nlp', default=False, action='store_true', help='Use NLP dataset.')
    parser.add_argument('--sugon', default=False, action='store_true', help='Use sugon partition.')
    args = parser.parse_args()
    if not os.path.exists('shells'):
        os.mkdir('shells')
    if args.allocate:
        allocate(args)
    elif args.finish:
        finish(args)
    elif args.inspect:
        inspect(args)
    else:
        initialize(args)

        
datasets = ['WN18RR', 'FB237', 'YAGO3-10', 'ICEWS14', 'ICEWS05-15', 'GDELT']
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tensor Factorization for Knowledge Graph Completion"
    )

    parser.add_argument(
        '--dataset', choices=datasets,
        help="Dataset in {}".format(datasets)
    )

    parser.add_argument(
        '--model', type=str, default='CP'
    )

    parser.add_argument(
        '--regularizer', type=str, default='NA',
    )

    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument(
        '--optimizer', choices=optimizers, default='Adagrad',
        help="Optimizer in {}".format(optimizers)
    )

    parser.add_argument(
        '--max_epochs', default=50, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid', default=10, type=float,
        help="Number of epochs before valid."
    )
    parser.add_argument(
        '--rank1', default=200, type=int,
        help="Factorization rank for entity."
    )
    parser.add_argument(
        '--rank2', default=200, type=int,
        help="Factorization rank for relation or timestamp."
    )
    parser.add_argument(
        '--batch_size', default=1000, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--reg', default=0, type=float,
        help="Regularization weight"
    )
    parser.add_argument(
        '--init', default=1e-3, type=float,
        help="Initial scale"
    )
    parser.add_argument(
        '--learning_rate', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--decay1', default=0.9, type=float,
        help="decay rate for the first moment estimate in Adam"
    )
    parser.add_argument(
        '--decay2', default=0.999, type=float,
        help="decay rate for second moment estimate in Adam"
    )

    parser.add_argument(
        '--ratio', default=0.64, type=float,
        help="the ratio of temporal embedding dimension"
    )

    parser.add_argument(
        '--dropout', default=0.1, type=float,
        help="dropout rate for fact network"
    )

    parser.add_argument('-train', '--do_train', action='store_true')
    parser.add_argument('-test', '--do_test', action='store_true')
    parser.add_argument('-save', '--do_save', action='store_true')
    parser.add_argument('-weight', '--do_ce_weight', action='store_true')
    parser.add_argument('-path', '--save_path', type=str, default='../logs/')
    parser.add_argument('-id', '--model_id', type=str, default='0')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='')
    args = parser.parse_args()