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
num_train_epochs = 1
learning_rate = 5e-5
warmup_steps = 0
weight_decay = 0.01
max_seq_length = 128

models = ['General_TinyBERT_4L_312D']

mode_level = 'sentence'
domains = ['hotels', 'medicine', 'products', 'reviews']


def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 scripts/download_dataset.py --data_dir data/multiemo2'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert-base-uncased')):
        logger.info("Downloading bert-base-uncased model")
        cmd = 'python3 download_bert.py'
        run_process(cmd)
        logger.info("Downloading finished")

    if not all([os.path.exists(os.path.join(MODEL_FOLDER, m)) for m in models]):
        logger.info(f"Downloading General TinyBERT models")
        cmd = f'python3 download_general_tinyberts.py'
        run_process(cmd)
        logger.info("Downloading finished")

    for model in models:
        student_model_name = model.split('General_')[1]
        general_tinybert_dir = f'data/models/{model}'

        # # SINGLE DOMAIN RUNS
        # for domain in domains:
        #     task_name = f'multiemo_en_{domain}_{mode_level}'
        #     if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert-base-uncased', task_name)):
        #         fine_tune_bert_base(task_name)
        #
        #     for i in range(REP_NUM):
        #         teacher_model_dir = f'data/models/bert-base-uncased/{task_name}'
        #         tmp_tinybert_output_dir = manage_output_dir(f"data/models/TMP_{student_model_name}", task_name)
        #         tinybert_output_dir = manage_output_dir(f"data/models/{student_model_name}", task_name)
        #
        #         task_distill_tinybert(task_name, student_model_name, teacher_model_dir, general_tinybert_dir,
        #                               tmp_tinybert_output_dir, tinybert_output_dir)
        #
        #         evaluate_tinybert(student_model_name, task_name, tinybert_output_dir)

        # DOMAIN-OUT RUNS
        for domain in domains:
            task_name = f'multiemo_en_N{domain}_{mode_level}'
            eval_task_name = f'multiemo_en_{domain}_{mode_level}'

            if not os.path.exists(os.path.join(MODEL_FOLDER, 'bert-base-uncased', task_name)):
                fine_tune_bert_base(task_name)

            for i in range(REP_NUM):
                teacher_model_dir = f'data/models/bert-base-uncased/{task_name}'
                tmp_tinybert_output_dir = manage_output_dir(f"data/models/TMP_{student_model_name}", task_name)
                tinybert_output_dir = manage_output_dir(f"data/models/{student_model_name}", task_name)

                task_distill_tinybert(task_name, student_model_name, teacher_model_dir, general_tinybert_dir,
                                      tmp_tinybert_output_dir, tinybert_output_dir)

                evaluate_tinybert(student_model_name, eval_task_name, tinybert_output_dir)

    # cmd = f'python3 -m gather_results --task_name {task_name}'
    # logger.info(f"Gathering results to csv for {task_name}")
    # run_process(cmd)


def fine_tune_bert_base(task_name):
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


def task_distill_tinybert(task_name, student_model_name, teacher_model_dir, general_tinybert_dir,
                          tmp_tinybert_output_dir, tinybert_output_dir):
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
    logger.info(f"Training Temp {student_model_name} model on {task_name}")
    run_process(cmd)

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
    logger.info(f"Training {student_model_name} model on {task_name}")
    run_process(cmd)


def evaluate_tinybert(student_model_name, task_name, tinybert_output_dir):
    cmd = 'python3 task_distill.py '
    options = [
        '--do_eval',
        '--student_model', tinybert_output_dir,
        '--data_dir', 'data/multiemo2',
        '--task_name', task_name,
        '--output_dir', tinybert_output_dir,
        '--eval_batch_size', str(batch_size),
        '--max_seq_length', str(max_seq_length),
        '--do_lower_case'
    ]
    cmd += ' '.join(options)
    logger.info(f"Evaluating {student_model_name} for {task_name}")
    run_process(cmd)


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
