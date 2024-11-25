import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument("-do_train", type=str2bool, default=False)
    parser.add_argument("-do_test", type=str2bool, default=False)
    parser.add_argument("-do_predict", type=str2bool, default=False)
    parser.add_argument("-do_process", type=str2bool, default=False)
    parser.add_argument("-do_process_train", type=str2bool, default=True)
    parser.add_argument("-do_edit_train", type=str2bool, default=False)
    parser.add_argument("-do_edit_test", type=str2bool, default=False)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-output_dir", type=str, default='models/qmsum')
    parser.add_argument("-result_dir", type=str, default='results/qmsum')
    parser.add_argument("-log_file", type=str, default='logs/qmsum.log')

    # seq2seq dataset args
    parser.add_argument("-dataset", type=str, default='/home/yoon/Desktop/SKKU/NLP/MSummarizer_modified/OLDS/process/data/qmsum/qmsum') # QMsum absolute path. 
    parser.add_argument("-max_source_length", type=int, default=1024)
    parser.add_argument("-max_target_length", type=int, default=128)
    parser.add_argument("-preprocessing_num_workers", type=int, default=1)
    parser.add_argument("-prefix", type=str, default="")
    parser.add_argument("-save_path", type=str, default='data/qmsum') # 원래는 data/samsum  (근데 이렇게 하면 원래 쓰던 데이터를 덮어쓰는데??)  -> 추후 수정 

    # omission dataset args
    parser.add_argument("-data_split_size", type=int, default=10)
    parser.add_argument("-domain", type=str, default='all',
        choices=["bart_large", "bart_base", "t5_base", "t5_small", "baseline", "pegasus", "all"] 
    )  
    ### # 1: full dial; 2: omission + rest; 3: omission + oracle rest + rest
    parser.add_argument("-post_edit_split_num", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("-post_edit_output_raw_results", type=str2bool, default=False)
    parser.add_argument("-post_edit_cut_ratio", type=float, default=0.)
    parser.add_argument("-post_edit_add_ratio", type=float, default=0.)
    parser.add_argument("-post_edit_keep_size", type=str2bool, default=False)
    parser.add_argument("-confidence_ratio", type=float, default=None)
    
    # model args
    parser.add_argument("-model", type=str, default='facebook/bart-base') # 6개 model 
    parser.add_argument("-checkpoint", type=str, default='')
    parser.add_argument("-baseline", type=str2bool, default=False)
    parser.add_argument("-num_labels", type=int, default=3)
    parser.add_argument("-mode", type=str, default='pair', choices=['pair', 'seq', 'span']) # 이 모드가 뭘 의미하는 걸까요? 흠. 

    # training args
    parser.add_argument("-per_device_train_batch_size", type=int, default=4)
    parser.add_argument("-num_train_epochs", type=int, default=2)  # epoch 2로 수정
    parser.add_argument("-max_train_steps", type=int, default=None)
    parser.add_argument("-max_grad_norm", type=float, default=1.0)
    parser.add_argument("-learning_rate", type=float, default=5e-5)
    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "-lr_scheduler_type", type=str, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("-num_warmup_steps", type=int, default=0)
    parser.add_argument("-save_strategy", type=str, default='epoch', choices=['step', 'epoch'])
    parser.add_argument("-save_steps", type=int, default=2000)

    # inference args
    parser.add_argument("-per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("-val_max_target_length", type=int, default=128)
    parser.add_argument("-val_min_target_length", type=int, default=1)
    parser.add_argument("-num_beams", type=int, default=4)
    parser.add_argument("-do_sample", type=str2bool, default=False)
    parser.add_argument("-output_attentions", type=str2bool, default=False)
    parser.add_argument("-output_attention_path", type=str, default='results/samsum')

    args = parser.parse_args()

    return args
