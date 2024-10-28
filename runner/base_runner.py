import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import os
import sys
import datetime
import copy
import wandb


from dataset.data_process import DataProcess


class DualOutput:
    """
    # Usage
    sys.stdout = DualOutput("output.txt")

    # Example prints
    print("This will go to both terminal and file.")
    print("Another line, both on terminal and in the file.")
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()


def get_time_stamp():
    # Get current date and time
    now = datetime.datetime.now()

    # Format date and time
    date_time_format = now.strftime("%b%d_%H-%M-%S")

    # # Get hostname
    # hostname = socket.gethostname()

    # # Concatenate strings
    # formatted_string = f"{date_time_format}_{hostname}"

    # return formated_string
    return date_time_format


def cal_metrics(all_labels, all_predictions, is_debug=False):
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    true_positive = sum((all_labels == all_predictions) & (all_labels == 1))
    false_positive = sum((all_labels != all_predictions) & (all_labels == 0))
    false_negative = sum((all_labels != all_predictions) & (all_labels == 1))
    true_negative = sum((all_labels == all_predictions) & (all_labels == 0))

    specificity = true_negative / (true_negative + false_positive)
    ppv = true_positive / (true_positive + false_positive)
    npv = true_negative / (true_negative + false_negative)

    if is_debug:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        # print(f"Accuracy_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='accuracy') * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        # print(f"Precision_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='precision') * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"Specificity: {specificity * 100:.2f}%")
        print(f"PPV: {ppv * 100:.2f}%")
        print(f"NPV: {npv * 100:.2f}%")
        # print(f"Recall_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='recall') * 100:.2f}%")
        print(f"F1: {f1 * 100:.2f}%")
        # print(f"F1_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='f1') * 100:.2f}%")
        print(f"True Positive: {true_positive}", end=" ")
        print(f"False Positive: {false_positive}", end=" ")
        print(f"False Negative: {false_negative}", end=" ")
        print(f"True Negative: {true_negative}", end=" ")
        print(f"gt 1: {sum(all_labels == 1)}", end=" ")
        print(f"gt 0: {sum(all_labels == 0)}")
        print(f"num of all labels: {len(all_labels)}")
        print(f"num of all predictions: {len(all_predictions)}")

    ans = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }

    return ans


class BestRecord:
    # TODO: add support for `specificity`
    ls_record_item = ["acc", "precision", "recall", "f1", "loss"]

    def __init__(self):
        dc_init_best_record = {
            "epoch": -1,
            "acc": -1,
            "precision": -1,
            "recall": -1,
            "f1": -1,
            "loss": 999999,
            "true_positive": -1,
            "false_positive": -1,
            "false_negative": -1,
            "true_negative": -1,
        }
        self.best_acc = -1
        self.best_acc_record = copy.deepcopy(dc_init_best_record)

        self.best_precision = -1
        self.best_precision_record = copy.deepcopy(dc_init_best_record)

        self.best_recall = -1
        self.best_recall_record = copy.deepcopy(dc_init_best_record)

        self.best_f1 = -1
        self.best_f1_record = copy.deepcopy(dc_init_best_record)

        self.best_loss = 999999
        self.best_loss_record = copy.deepcopy(dc_init_best_record)

    def update(
        self,
        acc=None,
        precision=None,
        recall=None,
        f1=None,
        true_positive=None,
        false_positive=None,
        true_negative=None,
        false_negative=None,
        loss=None,
        epoch=None,
    ):
        # TODO: add support for `specificity`
        
        current_record = {
            "epoch": epoch,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": loss,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        }
        state = []
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_acc_record.update(current_record)

            state.append("acc")

        if precision > self.best_precision:
            self.best_precision = precision
            self.best_precision_record.update(current_record)

            state.append("precision")

        if recall > self.best_recall:
            self.best_recall = recall
            self.best_recall_record.update(current_record)

            state.append("recall")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_f1_record.update(current_record)

            state.append("f1")

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_record.update(current_record)

            state.append("loss")
        return state

    def _gen_str(self, record_item):
        # TODO: add support for `specificity`

        assert record_item in self.ls_record_item
        best_item_val = None
        best_record = None
        if record_item == "acc":
            best_item_val = self.best_acc
            best_record = self.best_acc_record
        elif record_item == "precision":
            best_item_val = self.best_precision
            best_record = self.best_precision_record
        elif record_item == "recall":
            best_item_val = self.best_recall
            best_record = self.best_recall_record
        elif record_item == "f1":
            best_item_val = self.best_f1
            best_record = self.best_f1_record
        elif record_item == "loss":
            best_item_val = self.best_loss
            best_record = self.best_loss_record
        else:
            raise ValueError(f"Unknown record_item: {record_item}")

        if record_item == "loss":
            ans_str = f"Best {record_item}: {best_item_val:.4f}, "
        else:
            ans_str = f"Best {record_item}: {best_item_val*100:.2f}%, "
        ans_str += f"recall: {best_record['recall']*100:.2f}%, "
        ans_str += f"precision: {best_record['precision']*100:.2f}%, "
        ans_str += f"f1: {best_record['f1']*100:.2f}%, "
        ans_str += f"acc: {best_record['acc']*100:.2f}%, "
        ans_str += f"loss: {best_record['loss']:.4f}, "
        ans_str += f"TP: {best_record['true_positive']}, "
        ans_str += f"FP: {best_record['false_positive']}, "
        ans_str += f"TN: {best_record['true_negative']}, "
        ans_str += f"FN: {best_record['false_negative']}, "
        ans_str += f"epoch: {best_record['epoch']}"
        ans_str += f"\n"

        return ans_str

    def __str__(self) -> str:
        # TODO: add support for `specificity`

        ans_str = ""
        for record_item in self.ls_record_item:
            ans_str += self._gen_str(record_item)
        return ans_str


class ModelnData:
    """
    Model and Dataloader
    For the experiments with multiple clusters, we need to train a model for each cluster. This class is used to run a model with the corrsponding dataloader.
    """

    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        classify_input=None,
        cluster_method="cluster_v1",
        cluster_idx=None,
        # exp_name=None,
        model_name=None,
        lr=1e-6,
        dropout_rate=0,
        ls_hidden_dim=[100, 300, 200, 20],
        criterion=None,
        num_epochs=1000,
        batch_size=32,
        load_path=None,
        is_print=True,
        is_debug=False,
    ) -> None:
        """
        cluster_idx: 0, 1, 2, None. None means all clusters combined.
        """
        self.cluster_idx = cluster_idx
        # self.exp_name = exp_name
        self.model_name = model_name
        self.is_debug = is_debug
        self.is_print = is_print

        self.ls_hidden_dim = ls_hidden_dim
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.learning_rate = lr

        if self.is_print:
            print(f"===== Model: {self.model_name} =====")
            print(
                f"   -- Cluster: Index: {self.cluster_idx}, Method: {cluster_method}, ls_hidden_dim: {self.ls_hidden_dim}, lr: {self.learning_rate}"
            )

        self.train_loader = self.gen_dataloader(
            mode="train",
            csv_filepath=train_data_path,
            classify_input=classify_input,
            cluster_method=cluster_method,
            cluster_idx=cluster_idx,
            batch_size=batch_size,
        )

        self.test_loader = self.gen_dataloader(
            mode="test",
            csv_filepath=test_data_path,
            classify_input=classify_input,
            cluster_method=cluster_method,
            cluster_idx=cluster_idx,
            batch_size=batch_size,
        )

        for i in self.train_loader:
            self.input_size = i[0].shape[1]
            break

        # ## Set up the model
        self.net = Net(
            input_dim=self.input_size,
            ls_hidden_dim=self.ls_hidden_dim,
            n_classes=self.num_classes,
            dropout_rate=dropout_rate,
        )
        if self.is_print:
            print(f"   -- Print Model Architecture:")
            print(self.net)
            print(f"   -- End of Print Model Architecutre")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # ## Load model
        if load_path is not None:
            self.load_model(pth_file_path=load_path)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        if criterion == None or criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
            print(f"   -- Criterion: CrossEntropyLoss")
        elif criterion == "focal_loss":
            self.criterion = FocalLoss()
            print(f"   -- Criterion: FocalLoss")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        self.train_best_record = BestRecord()
        self.test_best_record = BestRecord()

    def gen_dataloader(
        self,
        mode="train",
        csv_filepath=None,
        classify_input="without_lab_encoded",
        cluster_method=None,
        cluster_idx=None,
        batch_size=32,
    ):
        assert csv_filepath is not None
        assert mode in ["train", "test"]

        print(
            f"===== Generating {mode} dataloader from: {csv_filepath}, classify_input: {classify_input}, cluster_method: {cluster_method}, cluster_idx: {cluster_idx} ====="
        )

        # ## Train data
        df = DataProcess(csv_filepath).data_process_cluster(
            classify_input=classify_input,
            cluster_method=cluster_method,
            cluster_idx=cluster_idx,
        )
        labels = df["diabetes_label"]
        data = df.drop("diabetes_label", axis=1)

        if self.is_debug:
            print(f"===== {mode} dataset label composition =====")
            print(labels.value_counts())
            # print(data.head())

        labels = labels.values
        data = data.values

        dataset = MyDataset(data, labels)

        if mode == "train":
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif mode == "test":
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return dataloader

    def save_model(self, epoch=None, save_dir=None, comment=None):
        assert comment is not None and save_dir is not None

        save_path = os.path.join(
            save_dir, f"{comment}_{self.model_name}_{str(epoch).zfill(4)}.pth"
        )
        best_save_path = os.path.join(save_dir, f"{comment}_{self.model_name}.pth")

        # torch.save(self.net.state_dict(), save_path)
        # print(f"===== Save model to: {save_path} =====")
        torch.save(self.net.state_dict(), best_save_path)
        print(f"===== Save model to: {best_save_path} =====")

    def load_model(
        self, pth_file_path=None, load_dir=None, comment=None, pth_file_name=None
    ):
        """
        pth_file_path: The full path of the pth file. This parameter has the highest priority and will override load_dir and pth_file_name.
        """
        if pth_file_path is not None:
            assert comment is None and pth_file_name is None
            self.net.load_state_dict(
                torch.load(pth_file_path, map_location=self.device), strict=True
            )
            print(
                f"===== Load model ({self.model_name}) by pth_file_path from: {pth_file_path} ====="
            )
            return
        else:
            if pth_file_name is None:
                pth_file_name = f"{comment}_{self.model_name}.pth"
            pth_file_path = os.path.join(load_dir, pth_file_name)
            assert os.path.exists(pth_file_path), f"File not found: {pth_file_path}"
            print(f"===== Load model from: {pth_file_path} =====")

    def cal_metrics_in_run(self, all_labels, all_predictions):
        dc_metrics = cal_metrics(all_labels, all_predictions)

        accuracy = dc_metrics['accuracy']
        precision = dc_metrics['precision']
        recall = dc_metrics['recall']
        f1 = dc_metrics['f1']
        specificity = dc_metrics['specificity']
        true_positive = dc_metrics['true_positive']
        false_positive = dc_metrics['false_positive']
        false_negative = dc_metrics['false_negative']
        true_negative = dc_metrics['true_negative']

        if self.is_debug:
            print(f"===== Model {self.model_name} Metrics =====")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            # print(f"Accuracy_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='accuracy') * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            # print(f"Precision_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='precision') * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            # print(f"Recall_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='recall') * 100:.2f}%")
            print(f"F1: {f1 * 100:.2f}%")
            print(f"Specificity: {specificity * 100:.2f}%")
            # print(f"F1_cal_weighted_average: {self.cal_weighted_average(ls_ans, item='f1') * 100:.2f}%")
            print(f"True Positive: {true_positive}", end=" ")
            print(f"False Positive: {false_positive}", end=" ")
            print(f"False Negative: {false_negative}", end=" ")
            print(f"True Negative: {true_negative}", end=" ")
            print(f"gt 1: {sum(all_labels == 1)}", end=" ")
            print(f"gt 0: {sum(all_labels == 0)}")
            print(f"num of all labels: {len(all_labels)}")
            print(f"num of all predictions: {len(all_predictions)}")

        return precision, recall, f1, accuracy, specificity

    def epoch_print(self, mode, epoch, loss_avg_epoch, precision, recall, f1, accuracy, specificity):
        if not self.is_print:
            return

        assert mode in ["train", "test"]
        if mode == "train":
            best_record = self.train_best_record
        elif mode == "test":
            best_record = self.test_best_record
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.model_name is not None:
            print(f"Model: {self.model_name}", end=" ")
        print(f"Epoch: {epoch} {mode.upper()}", end=" ")
        print(f"Loss: {loss_avg_epoch:.4f}", end=" ")
        print(f"f1: {f1:.4f}", end=" ")
        print(f"Recall: {recall * 100:.2f}%", end=" ")
        print(f"Precision: {precision * 100:.2f}%", end=" ")
        print(f"Accuracy: {accuracy * 100:.2f}%", end=" ")
        print(f"Specificity: {specificity * 100:.2f}%")
        print(f"Best f1: {best_record.best_f1 * 100:.2f}%", end=" ")
        print(f"Best recall: {best_record.best_recall * 100:.2f}%")
        print(
            f"Best precision: {best_record.best_precision * 100:.2f}%",
            end=" ",
        )
        print(f"Best accuracy: {best_record.best_acc * 100:.2f}%")

    def test(self, epoch):
        test_loader = self.test_loader
        if self.is_debug:
            print(f"===== Test epoch: {epoch} =====")
            print(f"len(test_loader.dataset): {len(test_loader.dataset)}")
        self.net.eval()  # Set the model to evaluation mode

        loss_avg_epoch = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                # ## Move data to GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                loss_avg_epoch += loss.item()

                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())

        loss_avg_epoch /= len(self.test_loader)

        # Calculate precision and recall
        precision, recall, f1, accuracy = self.cal_metrics_in_run(all_labels, all_predictions)
        self.test_best_record.update(
            acc=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            loss=loss_avg_epoch,
            epoch=epoch,
        )

        if self.is_print:
            self.epoch_print(
                mode="test",
                epoch=epoch,
                loss_avg_epoch=loss_avg_epoch,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
            )

        ans = {
            "loss": loss_avg_epoch,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_acc": self.test_best_record.best_acc,
            "best_precision": self.test_best_record.best_precision,
            "best_recall": self.test_best_record.best_recall,
            "all_labels": all_labels,
            "all_predictions": all_predictions,
        }

        return ans

    def train(self, epoch):
        self.net.train()  # Set the model to training mode
        # for epoch in tqdm(range(self.num_epochs)):  # Number of epochs
        loss_avg_epoch = 0

        all_labels = []
        all_predictions = []

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(
                self.device
            )  # Move data to GPU

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            # Log information to TensorBoard
            # self.writer.add_scalar(
            #     "Loss_step/train", loss.item(), epoch * len(self.train_loader) + i
            # )

            loss_avg_epoch += loss.item()

        loss_avg_epoch /= len(self.train_loader)

        # Calculate precision and recall
        precision, recall, f1, accuracy = self.cal_metrics_in_run(all_labels, all_predictions)
        self.train_best_record.update(
            acc=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            loss=loss_avg_epoch,
            epoch=epoch,
        )

        ans = {
            "loss": loss_avg_epoch,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_acc": self.train_best_record.best_acc,
            "best_precision": self.train_best_record.best_precision,
            "best_recall": self.train_best_record.best_recall,
            "all_labels": all_labels,
            "all_predictions": all_predictions,
        }

        if self.is_print:
            self.epoch_print(
                mode="train",
                epoch=epoch,
                loss_avg_epoch=loss_avg_epoch,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
            )

        return ans

    def run(self):
        self.test(epoch=0)

        self.net.train()  # Set the model to training mode
        for epoch in tqdm(range(self.num_epochs)):  # Number of epochs
            self.train(epoch + 1)
            self.test(epoch=epoch + 1)

        self.writer.close()


class BaseRunner:
    def __init__(
        self,
        exp_name=None,
        cluster_method=None,
        classify_input="without_lab_encoded",
        train_data_path=None,
        test_data_path=None,
        hidden_dim=[128, 384, 256, 20],
        criterion="cross_entropy",
        exp_save_dir="runs",
        lr=1e-6,
        dropout_rate=0,
        num_epochs=5000,
        dc_load_path=None,
        LUPI_load_dir=None,
        is_debug=False,
        debug_params=None,
    ) -> None:
        """
        cluster_method: cluster_v1, cluster_v2, None. None means all clusters combined.
        """
        self.is_debug = is_debug
        self.debug_params = debug_params

        self.ls_hidden_dim = hidden_dim
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.wandbTracking = False

        # ## Set up the experiment name, save_dir, writer, and log file
        self.exp_name = self.gen_exp_name(
            exp_name=exp_name,
            cluster_method=cluster_method,
            classify_input=classify_input,
            hidden_dim=hidden_dim,
            lr=lr,
            dropout_rate=dropout_rate,
            is_debug=is_debug,
        )

        # ## Set up the save_dir and writer
        self.save_dir = os.path.join(exp_save_dir, self.exp_name)
        self.writer = SummaryWriter(log_dir=self.save_dir)  # For TensorBoard
        sys.stdout = DualOutput(self.save_dir + ".txt")

        # ## Set up the model and dataloader
        self.setup_model_and_dataloader(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            cluster_method=cluster_method,
            classify_input=classify_input,
            dc_load_path=dc_load_path,
            ls_hidden_dim=hidden_dim,
            criterion=criterion,
            lr=lr,
            dropout_rate=dropout_rate,
        )

        # ## Set up the best record
        self.train_best_record = BestRecord()
        self.test_best_record = BestRecord()

        self.debug()

    @staticmethod
    def gen_exp_name(
        exp_name=None,
        cluster_method=None,
        classify_input=None,
        hidden_dim=None,
        lr=None,
        dropout_rate=None,
        LUPI_loss_mode=None,
        is_debug=None,
    ):
        # ## Code for cluster_method
        if cluster_method is None:
            nm_code_clus = "all"
        elif cluster_method == "cluster_v1":
            nm_code_clus = "clusV1"
        elif cluster_method == "cluster_v2":
            nm_code_clus = "clusV2"
        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        # ## Code for classify_input
        if classify_input == "full_raw":
            nm_code_input = "fullRaw"
        elif classify_input == "wo_lab_raw":
            nm_code_input = "woLabRaw"
        elif classify_input == "wo_lab_enc":
            nm_code_input = "woLabEnc"
        elif classify_input == "iu_raw":
            nm_code_input = "iuRaw"
        elif classify_input == "iu_enc":
            nm_code_input = "iuEnc"
        elif "_" not in classify_input:
            nm_code_input = classify_input
        # elif classify_input == "wo_lab":
        #     nm_code_input = "woLab"
        # elif classify_input == "full":
        #     nm_code_input = "fullFeat"
        # elif classify_input == "iu":
        #     nm_code_input = "iu"
        # elif classify_input.startswith("cov>"):
        #     nm_code_input = classify_input
        # elif classify_input.startswith("fullcov>"):
        #     nm_code_input = classify_input
        # elif classify_input == "wo_lab_diet":
        #     nm_code_input = "woLabDiet"
        # elif classify_input == "without_lab_encoded":
        #     nm_code_input = "woLabEnc"
        # elif classify_input == "no_lab_strict_encoded":
        #     nm_code_input = "noLabEnc"
        # elif classify_input == "no_lab_strict":
        #     nm_code_input = "noLab"
        # elif classify_input == "priviledged_info_encoded":
        # nm_code_input = "prInfEnc"
        else:
            raise ValueError(f"Unknown classify_input: {classify_input}")

        if exp_name is None or exp_name == "":
            exp_name = f"{nm_code_clus}_{nm_code_input}"
        else:
            exp_name = f"{exp_name}_{nm_code_clus}_{nm_code_input}"

        exp_name = (
            get_time_stamp()
            + "_"
            + exp_name
            + "_hiddenDim="
            + str(hidden_dim)
            + "_lr="
            + str(lr)
            + "_dropout="
            + str(dropout_rate)
        )

        if LUPI_loss_mode is not None:
            if LUPI_loss_mode == "diff_level":
                nm_code_LUPI = "diffLevel"
            elif LUPI_loss_mode == "soft_label":
                nm_code_LUPI = "softLabel"
            else:
                raise ValueError(f"Unknown LUPI loss mode: {LUPI_loss_mode}")
            exp_name += "_LUPI=" + nm_code_LUPI

        # if LUPI_load_dir is not None:
        #     self.exp_name = "LUPI_" + self.exp_name

        if is_debug:
            exp_name = "DEBUG_" + exp_name

        return exp_name

    def setup_model_and_dataloader(
        self,
        train_data_path=None,
        test_data_path=None,
        cluster_method=None,
        classify_input="without_lab_encoded",
        dc_load_path=None,
        ls_hidden_dim=None,
        criterion=None,
        lr=None,
        dropout_rate=0,
    ):
        """
        cluster_method: cluster_v1, cluster_v2, None. None means all clusters combined.
        dc_load_path: a dictionary of load paths for each cluster. None if not loading. {"model_name": load_path}"}
        """
        assert train_data_path is not None and test_data_path is not None
        assert cluster_method in ["cluster_v1", "cluster_v2", None]

        if cluster_method is None:
            # use data from all clusters combined
            print(f"Setting up model with all clusters combined")
            self.ls_mlp = [
                ModelnData(
                    train_data_path=train_data_path,
                    test_data_path=test_data_path,
                    classify_input=classify_input,
                    cluster_method=cluster_method,  # Here cluster_method is None
                    cluster_idx=None,
                    model_name="all_clus",
                    ls_hidden_dim=ls_hidden_dim,
                    criterion=criterion,
                    lr=lr,
                    dropout_rate=dropout_rate,
                    load_path=dc_load_path["all_clus"] if dc_load_path else None,
                    is_debug=self.is_debug,
                    is_print=True,
                )
            ]
        elif cluster_method in ["cluster_v1", "cluster_v2"]:
            print(f"Setting up model with cluster_method: {cluster_method}")
            # use data from cluster 0, 1, 2
            self.ls_mlp = [
                ModelnData(
                    train_data_path=train_data_path,
                    test_data_path=test_data_path,
                    classify_input=classify_input,
                    cluster_method=cluster_method,
                    cluster_idx=i,
                    model_name=f"clus_{i}",
                    ls_hidden_dim=ls_hidden_dim,
                    criterion=criterion,
                    lr=lr,
                    dropout_rate=dropout_rate,
                    load_path=dc_load_path[f"clus_{i}"]
                    if dc_load_path is not None
                    and os.path.exists(dc_load_path[f"clus_{i}"])
                    else None,
                    is_debug=self.is_debug,
                    is_print=True,
                )
                for i in range(3)
            ]

    def cal_weighted_average(self, ls_ans, item="accuracy"):
        weighted_average = 0
        total = 0
        for i in range(len(ls_ans)):
            weighted_average += ls_ans[i][item] * len(ls_ans[i]["all_labels"])
            total += len(ls_ans[i]["all_labels"])
        weighted_average /= total
        return weighted_average

    def cal_overall_metrics(self, ls_ans):
        all_labels, all_predictions = [], []
        for i in range(len(ls_ans)):
            all_labels.extend(ls_ans[i]["all_labels"])
            all_predictions.extend(ls_ans[i]["all_predictions"])

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        dc_metrics = cal_metrics(all_labels, all_predictions)

        acc = dc_metrics["accuracy"]
        precision = dc_metrics["precision"]
        recall = dc_metrics["recall"]
        f1 = dc_metrics["f1"]
        specificity = dc_metrics["specificity"]
        true_positive = dc_metrics["true_positive"]
        false_positive = dc_metrics["false_positive"]
        false_negative = dc_metrics["false_negative"]
        true_negative = dc_metrics["true_negative"]

        # acc = accuracy_score(all_labels, all_predictions)
        # precision = precision_score(all_labels, all_predictions, zero_division=0)
        # recall = recall_score(all_labels, all_predictions, zero_division=0)
        # f1 = f1_score(all_labels, all_predictions)

        # true_positive = sum((all_labels == all_predictions) & (all_labels == 1))
        # false_positive = sum((all_labels != all_predictions) & (all_labels == 0))
        # false_negative = sum((all_labels != all_predictions) & (all_labels == 1))
        # true_negative = sum((all_labels == all_predictions) & (all_labels == 0))

        # specificity = 0
        # if true_negative + false_positive != 0:
        #     specificity = true_negative / (true_negative + false_positive)

        # assert acc - self.cal_weighted_average(ls_ans, item="accuracy") < 1e-5

        # NOTE: The calculation of precision and recall is different from the weighted average
        # assert precision - self.cal_weighted_average(ls_ans, item="precision") < 1e-5
        # assert recall - self.cal_weighted_average(ls_ans, item="recall") < 1e-5
        # assert f1 - self.cal_weighted_average(ls_ans, item="f1") < 1e-5

        ans = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity" : specificity,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "all_labels": all_labels,
            "all_predictions": all_predictions,
        }
        return ans

    def terminal_print(self, mode="train", ans=None, epoch=None):
        print()
        if mode == "train":
            print(f"Epoch: {epoch} TRAIN", end=" ")
            best_recorder = self.train_best_record
        elif mode == "test":
            print(f"Epoch: {epoch} TEST", end=" ")
            best_recorder = self.test_best_record

        print(f"Loss: {ans['loss']:.4f}", end=" ")

        print(f"F1: {ans['f1'] * 100:.2f}%", end=" ")
        print(f"Recall: {ans['recall'] * 100:.2f}%", end=" ")
        print(f"Precision: {ans['precision'] * 100:.2f}%", end=" ")
        print(f"Accuracy: {ans['accuracy'] * 100:.2f}%")

        print(f"True Positive: {ans['true_positive']}", end=" ")
        print(f"False Positive: {ans['false_positive']}", end=" ")
        print(f"False Negative: {ans['false_negative']}", end=" ")
        print(f"True Negative: {ans['true_negative']}", end=" ")
        print(f"gt 1: {sum(ans['all_labels'] == 1)}", end=" ")
        print(f"gt 0: {sum(ans['all_labels'] == 0)}")

        print(f"Exp Save_dir: {self.save_dir}")
        print(f"Exp: {self.exp_name}")
        print(f"===== Best {mode.upper()} Record =====")
        print(best_recorder)
        print()

    def run_and_organize_ans(self, mode="train", epoch=None, savemodel=False):
        assert mode in ["train", "test"]

        best_record = (
            self.train_best_record if mode == "train" else self.test_best_record
        )

        ls_ans = []
        for mlp in self.ls_mlp:
            if mode == "train":
                ls_ans.append(mlp.train(epoch=epoch))
            elif mode == "test":
                ls_ans.append(mlp.test(epoch=epoch))
        # accuracy = self.cal_weighted_average(ls_ans, item="accuracy")
        # precision = self.cal_weighted_average(ls_ans, item="precision")
        # recall = self.cal_weighted_average(ls_ans, item="recall")
        loss = self.cal_weighted_average(ls_ans, item="loss")

        dc_metrics = self.cal_overall_metrics(ls_ans)
        dc_metrics["loss"] = loss

        accuracy = dc_metrics["accuracy"]
        precision = dc_metrics["precision"]
        recall = dc_metrics["recall"]
        specificity = dc_metrics["specificity"]
        f1 = dc_metrics["f1"]

        if mode == "train":
            self.train_best_record.update(
                acc=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                loss=loss,
                epoch=epoch,
            )
        elif mode == "test":
            state = self.test_best_record.update(
                acc=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                loss=loss,
                epoch=epoch,
            )
            # ## Save model
            if "acc" in state:
                self.save_model(epoch=epoch, comment="acc")
            if "recall" in state:
                self.save_model(epoch=epoch, comment="recall")
            if "precision" in state:
                self.save_model(epoch=epoch, comment="precision")
            if "f1" in state:
                self.save_model(epoch=epoch, comment="f1")

        best_acc = best_record.best_acc
        best_precision = best_record.best_precision
        best_recall = best_record.best_recall
        best_f1 = best_record.best_f1

        # # Log precision and recall to TensorBoard
        self.writer.add_scalar(f"Precision/{mode}", precision, epoch)
        self.writer.add_scalar(f"Recall/{mode}", recall, epoch)
        self.writer.add_scalar(f"Accuracy/{mode}", accuracy, epoch)
        self.writer.add_scalar(f"F1/{mode}", f1, epoch)
        self.writer.add_scalar(f"Specificity/{mode}", specificity, epoch)

        self.writer.add_scalar(f"Best_Precision/{mode}", best_precision, epoch)
        self.writer.add_scalar(f"Best_Recall/{mode}", best_recall, epoch)
        self.writer.add_scalar(f"Best_Accuracy/{mode}", best_acc, epoch)
        self.writer.add_scalar(f"Best_F1/{mode}", best_f1, epoch)

        self.writer.add_scalar(f"Loss/{mode}", loss, epoch)

        self.writer.add_text(f"Best_Record/{mode}", str(best_record), epoch)

        if self.wandbTracking:
            metrics = {"Precision" : accuracy,
                       "Recall" : recall,
                       "Accuracy" : accuracy,
                       "F1" : f1,
                       "Loss" : loss,
                       "Specificity" : specificity}
            if mode == 'train':
                metrics = {"train" : metrics}
                wandb.log(metrics, commit=False)
            else:
                metrics = {"test" : metrics}
                wandb.log(metrics)

        self.terminal_print(mode=mode, ans=dc_metrics, epoch=epoch)

    def save_model(self, epoch, comment=None):
        for i in range(len(self.ls_mlp)):
            self.ls_mlp[i].save_model(
                epoch=epoch, save_dir=self.save_dir, comment=comment
            )

    def load_model(self, load_dir=None, load_file_prefix=None):
        assert load_dir is not None
        assert os.path.exists(load_dir, load_file_prefix)

        for i in range(len(self.ls_mlp)):
            self.ls_mlp[i].load_model(
                load_dir=load_dir, load_file_prefix=load_file_prefix
            )

    def train(self, epoch):
        self.run_and_organize_ans(mode="train", epoch=epoch)

    def test(self, epoch):
        self.run_and_organize_ans(mode="test", epoch=epoch)

    def run(self):
        self.test(epoch=0)
        for epoch in tqdm(range(self.num_epochs)):  # Number of epochs
            self.train(epoch + 1)
            self.test(epoch=epoch + 1)

        self.writer.close()

    def debug(self):
        if not self.is_debug or self.debug_params is None:
            return
        print(f"===== DEBUG =====")
        print(f"Exp: {self.exp_name}")

        if self.debug_params is not None:
            print(f"===== BEGIN DEBUG PARAMS =====")
            print(self.debug_params)
            print(f"===== END DEBUG PARAMS =====")

        save_dir = self.debug_params["load_dir"]
        print(f"===== SANITY CHECK MODEL LOADING =====")

        print(f"Before loading:")
        self.test(epoch=0)
        self.test(epoch=0)
        # self.test(epoch=0)

        for mlp in self.ls_mlp:
            mlp.save_model(pth_save_path="tmp_save.pth")

        # tmp_mlp = []
        for mlp in self.ls_mlp:
            mlp.load_model(
                load_dir=save_dir, pth_file_name="bestRecall_all_clus_0276.pth"
            )
            # tmp_mlp.append(mlp)

        # for i in range(len(self.ls_mlp)):
        #     print(f"{i}, {self.ls_mlp[i] == tmp_mlp[i]})")

        print(f"After loading bestRecall_all_clus_0276.pth:")
        self.test(epoch=0)
        self.test(epoch=0)
        # self.test(epoch=0)

        print(f"Load pth before loading:")
        for mlp in self.ls_mlp:
            mlp.load_model(pth_file_path="tmp_save.pth")
        print(f"After loading tmp_save.pth:")
        self.test(epoch=0)
        self.test(epoch=0)
        # self.test(epoch=0)

        print(f"Load pth bestRecall_all_clus_0276.pth again:")
        for mlp in self.ls_mlp:
            mlp.load_model(pth_file_path="tmp_save.pth")

        print(f"After loading bestRecall_all_clus_0276.pth:")
        self.test(epoch=0)
        self.test(epoch=0)

        print(f"===== END SANITY CHECK MODEL LOADING =====")
        print(f"===== END DEBUG =====")
        exit(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #  random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    ...
