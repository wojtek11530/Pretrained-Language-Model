# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from data_processing import convert_examples_to_features, \
    compute_metrics, get_dataset_and_labels, processors, output_modes, MultiemoProcessor, SmartCollator
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from utils import dictionary_to_json, result_to_text_file

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write("")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    all_logits = None

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = {k: v.to(device) for k, v in batch_.items()}

        with torch.no_grad():
            input_ids = batch_['input_ids']
            input_mask = batch_['attention_mask']
            segment_ids = batch_['token_type_ids']
            label_ids = batch_['labels']

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        else:
            raise ValueError('Not known output model')

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if output_mode == "classification":
        preds = np.argmax(all_logits, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(all_logits)
    else:
        raise ValueError('Not known output model')

    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result, all_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)

    args = parser.parse_args()

    # intermediate distillation default parameters
    default_params = {
        "multiemo": {"max_seq_length": 128, "train_batch_size": 16},
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", "multiemo"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]
    elif 'multiemo' in task_name:
        args.max_seq_length = default_params['multiemo']["max_seq_length"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]
        elif 'multiemo' in task_name:
            args.num_train_epoch = default_params['multiemo']["num_train_epochs"]

    if task_name not in processors and 'multiemo' not in task_name:
        raise ValueError("Task not found: %s" % task_name)

    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        processor = processors[task_name]()

    if 'multiemo' in task_name:
        output_mode = 'classification'
    else:
        output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    logger.info('The args: {}'.format(args))

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)

        train_dataset, _ = get_dataset_and_labels(output_mode, train_features)
        collator = SmartCollator(0)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collator.collate_batch, pin_memory=False)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_dataset_and_labels(output_mode, eval_features)
    collator = SmartCollator(0)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collator.collate_batch, pin_memory=False)

    if not args.do_eval:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(device)

    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)
    student_model.to(device)
    if args.do_eval:
        ##########################################
        ##             TEST                     ##
        ##########################################

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer,
                                                     output_mode)
        test_data, test_labels = get_dataset_and_labels(output_mode, test_features)
        collator = SmartCollator(0)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collator.collate_batch, pin_memory=False)

        logger.info("***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()

        eval_start_time = time.monotonic()
        result, y_logits = do_eval(student_model, task_name, test_dataloader,
                                   device, output_mode, test_labels, num_labels)
        eval_end_time = time.monotonic()

        diff = timedelta(seconds=eval_end_time - eval_start_time)
        diff_seconds = diff.total_seconds()
        result['eval_time'] = diff_seconds

        result_to_text_file(result, os.path.join(args.output_dir, "test_results.txt"))
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        y_pred = np.argmax(y_logits, axis=1)
        print('\n\t**** Classification report ****\n')
        print(classification_report(test_labels.numpy(), y_pred, target_names=label_list))

        report = classification_report(test_labels.numpy(), y_pred, target_names=label_list, output_dict=True)
        report['eval_time'] = diff_seconds
        dictionary_to_json(report, os.path.join(args.output_dir, "test_results.json"))

    else:
        training_start_time = time.monotonic()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)

        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'

        optimizer = BertAdam(
            optimizer_grouped_parameters,
            schedule=schedule,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps
        )
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                input_mask = batch['attention_mask']
                segment_ids = batch['token_type_ids']
                label_ids = batch['labels']

                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask,
                                                                           is_student=True)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    if output_mode == "classification":
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                logger.info("***** Running evaluation *****")
                logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                student_model.eval()

                loss = tr_loss / nb_tr_steps
                cls_loss = tr_cls_loss / nb_tr_steps
                att_loss = tr_att_loss / nb_tr_steps
                rep_loss = tr_rep_loss / nb_tr_steps

                result = {}
                if args.pred_distill:
                    result, _ = do_eval(student_model, task_name, eval_dataloader,
                                        device, output_mode, eval_labels, num_labels)
                result['global_step'] = global_step
                result['cls_loss'] = cls_loss
                result['att_loss'] = att_loss
                result['rep_loss'] = rep_loss
                result['loss'] = loss
                print(json.dumps(result))

                result_to_file(result, output_eval_file)

                if not args.pred_distill:
                    save_model = True
                else:
                    save_model = False

                    if (task_name in acc_tasks or 'multiemo' in task_name) > best_dev_acc:
                        best_dev_acc = result['acc']
                        save_model = True

                    if task_name in corr_tasks and result['corr'] > best_dev_acc:
                        best_dev_acc = result['corr']
                        save_model = True

                    if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                        best_dev_acc = result['mcc']
                        save_model = True

                if save_model:
                    logger.info("***** Save model *****")

                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                    model_name = WEIGHTS_NAME
                    # if not args.pred_distill:
                    #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                    output_model_file = os.path.join(args.output_dir, model_name)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)

                student_model.train()

        # Measure End Time
        training_end_time = time.monotonic()

        diff = timedelta(seconds=training_end_time - training_start_time)
        diff_seconds = diff.total_seconds()

        training_parameters = vars(args)
        training_parameters['training_time'] = diff_seconds

        output_training_params_file = os.path.join(args.output_dir, "training_params.json")
        dictionary_to_json(training_parameters, output_training_params_file)


if __name__ == "__main__":
    main()
