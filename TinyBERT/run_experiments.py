import logging
import os
import sys

from utils import is_folder_empty

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODEL_FOLDER = os.path.join(DATA_FOLDER, 'models')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join(DATA_FOLDER, 'multiemo2')

REP_NUM = 1

batch_size = 16
num_train_epochs = 3
learning_rate = 5e-5
warmup_steps = 0
weight_decay = 0.01
max_seq_length = 128

task_name = 'multiemo_en_all_sentence'

models = ['TinyBERT_General_4L_312D', 'TinyBERT_General_6L_768D']


def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 scripts/download_dataset.py --data_dir data/multiemo2'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased')):
        logger.info("Downloading bert-base-uncased model")
        cmd = 'python3 download_bert.py'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased', 'multiemo_en_all_sentence')):
        cmd = 'python3 -m fine_tune_bert '
        options = [
            '--pretrained_model', 'data/models/bert-base-uncased',
            '--data_dir', 'data/multiemo2',
            '--task_name', task_name,
            '--output_dir', f'data/models/bert-base-uncased/{task_name}',
            '--max_seq_length', str(max_seq_length),
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--weight_decay', str(weight_decay),
            '--warmup_proportion', str(warmup_steps),
            '--train_batch_size', str(batch_size),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Fine tuning bert-base-uncased on {task_name}")
        run_process(cmd)

    for model in models:
        if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'huawei-noah', model)):
            logger.info(f"Downloading {model} from hugging face repo")
            cmd = f'python3 download_model_from_hugging_face.py --model_name huawei-noah/{model}'
            run_process(cmd)
            logger.info("Downloading finished")

        for i in range(REP_NUM):
            tmp_tinybert_output_dir = manage_output_dir(f"data/models/TMP_KD_{model}", task_name)
            teacher_model_dir = f'data/models/bert-base-uncased/{task_name}'
            general_tinybert_dir = f'data/models/huawei-noah/{model}'

            cmd = 'python3 task_distill.py '
            options = [
                '--teacher_model', teacher_model_dir,
                '--student_model', general_tinybert_dir,
                '--data_dir', 'data/multiemo2',
                '--task_name', task_name,
                '--output_dir', tmp_tinybert_output_dir,
                '--max_seq_length', str(max_seq_length),
                '--train_batch_size', str(batch_size),
                '--learning_rate', str(learning_rate),
                '--num_train_epochs', str(num_train_epochs),
                '--weight_decay', str(weight_decay),
                '--warmup_proportion', str(warmup_steps),
                '--do_lower_case'
            ]
            cmd += ' '.join(options)
            logger.info(f"Training Temp {model} model on {task_name}")
            run_process(cmd)

            tinybert_output_dir = manage_output_dir(f"data/models/KD_{model}", task_name)
            cmd = 'python3 task_distill.py '
            options = [
                '--pred_distill',
                '--teacher_model', teacher_model_dir,
                '--student_model', tmp_tinybert_output_dir,
                '--data_dir', 'data/multiemo2',
                '--task_name', task_name,
                '--output_dir', tinybert_output_dir,
                '--max_seq_length', str(max_seq_length),
                '--train_batch_size', str(batch_size),
                '--learning_rate', str(learning_rate),
                '--num_train_epochs', str(num_train_epochs),
                '--weight_decay', str(weight_decay),
                '--warmup_proportion', str(warmup_steps),
                '--do_lower_case'
            ]
            cmd += ' '.join(options)
            logger.info(f"Training KD_{model} model on {task_name}")
            run_process(cmd)

            cmd = 'python3 -m test '
            options = [
                '--do_eval',
                '--student_model', tinybert_output_dir,
                '--data_dir', '../data/multiemo2',
                '--task_name', task_name,
                '--output_dir', tinybert_output_dir,
                '--max_seq_length', str(max_seq_length),
                '--do_lower_case'
            ]
            cmd += ' '.join(options)
            logger.info(f"Evaluating KD_{model} for {task_name}")
            run_process(cmd)

    # cmd = f'python3 -m gather_results --task_name {task_name}'
    # logger.info(f"Gathering results to csv for {task_name}")
    # run_process(cmd)


def run_process(proc):
    os.system(proc)


def manage_output_dir(output_dir: str, task_name: str) -> str:
    output_dir = os.path.join(output_dir, task_name)
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == '__main__':
    main()
