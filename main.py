''' fix the size of the graph'''

import torch
from tqdm import tqdm
import torch_geometric.data
from runner.mlp_runner import ModelnData, BestRecord, cal_metrics, BaseRunner, DualOutput

from dataset.data_process import DataProcess
from torch.utils.data import DataLoader, Dataset
#import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data as gnnData
#from torch_geometric.data import Dataset as GDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv
import numpy as np
import torch.nn.init as init
import os
import shutil
import yaml
import sys
from torch.utils.tensorboard import SummaryWriter
import random
import wandb
import argparse as ap


class lupiGNN(torch.nn.Module):
    def __init__(self, batch_size=32):
        super(lupiGNN, self).__init__()
        self.pi_encoder = lupiEncoder(batch_size=batch_size)
        self.obs_encoder = obsEncoder(batch_size=batch_size)
        self.decoder = decoder(batch_size=batch_size)
        self.device = None

    def set_device(self, device):
        self.device = device
        self.pi_encoder.device = device
        self.obs_encoder.device = device
        self.decoder.device = device
    
    def set_graphs(self, num_graphs):
        self.decoder.num_graphs = num_graphs

    def latent_forward(self, inputs, pi_inputs, pi_train=False):
        if pi_train == True:
            r_p = self.pi_encoder(pi_inputs)
            return None, r_p
        
        if pi_inputs != None:
            r_p = self.pi_encoder(pi_inputs)
            r_obs = self.obs_encoder(inputs)
            return r_obs, r_p
        else:
            r_obs = self.obs_encoder(inputs)
            return r_obs, None

    def forward(self, inputs, edge_index):
        return self.decoder(inputs, edge_index)


class lupiEncoder(torch.nn.Module):
    def __init__(self, out_dim=39, batch_size=32):
        super(lupiEncoder, self).__init__()
        self.conv1 = SAGEConv(-1, 16)
        self.conv2 = SAGEConv(-1, 16)
        self.out = torch.nn.LazyLinear(out_dim)
        self.relu = torch.nn.ReLU()
        self.batch_size = batch_size
        self.device = None

    def forward(self, inputs: torch_geometric.data.Batch):
        x = inputs.x.to(self.device)
        edge_index = inputs.edge_index.to(self.device)
        partial_edge_index, edge_mask = torch_geometric.utils.dropout_edge(edge_index)

        x = self.conv1(x, partial_edge_index)
        x = self.relu(x)

        size = int(len(x) / inputs.num_graphs)

        batch_out = []
        for batch in range(inputs.num_graphs):
            graph = x[batch * size : (batch + 1) * size]
            graph = torch.flatten(graph)
            out = self.out(graph)
            out = out.unsqueeze(1)

            batch_out.append(out)
        batch_out = torch.stack(batch_out, dim=0)

        return batch_out
        #return batch_out.unsqueeze(1)
    
class obsEncoder(torch.nn.Module):
    def __init__(self, out_dim=39, batch_size=32):
        super(obsEncoder, self).__init__()
        self.conv1 = SAGEConv(-1, 16)
        self.conv2 = SAGEConv(-1, 16)
        self.out = torch.nn.LazyLinear(out_dim)
        self.relu = torch.nn.ReLU()
        self.device = None

    def forward(self, inputs: torch_geometric.data.Batch):
        x = inputs.x.to(self.device)

        size = int(len(x) / inputs.num_graphs)
        edge_index = inputs.edge_index.to(self.device)
        partial_edge_index, edge_mask = torch_geometric.utils.dropout_edge(edge_index)

        x = self.conv1(x, partial_edge_index)
        x = self.relu(x)

        batch_out = []
        for batch in range(inputs.num_graphs):
            graph = x[batch * size : (batch + 1) * size]
            graph = torch.flatten(graph)
            out = self.out(graph)
            out = out.unsqueeze(1)

            batch_out.append(out)
        batch_out = torch.stack(batch_out, dim=0)

        return batch_out
        #return batch_out.unsqueeze(1)
    
class decoder(torch.nn.Module):
    def __init__(self, out_dim=2, batch_size=32):
        super(decoder, self).__init__()
        self.conv1 = SAGEConv(-1, 16)
        self.conv2 = SAGEConv(-1, 16)
        self.out = torch.nn.LazyLinear(out_dim)
        self.relu = torch.nn.ReLU()
        self.num_graphs = batch_size
        self.device = None

    def forward(self, x, edge_index):
        x = x.to(self.device)
        x = x.view(x.shape[0] * x.shape[1], 1)
        size = int(len(x) / self.num_graphs)
        edge_index = edge_index.to(self.device)

        partial_edge_index, edge_mask = torch_geometric.utils.dropout_edge(edge_index)
        x = self.conv1(x, partial_edge_index)
        x = self.relu(x)

        partial_edge_index, edge_mask = torch_geometric.utils.dropout_edge(edge_index)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        batch_out = []
        for batch in range(self.num_graphs):
            graph = x[batch * size : (batch + 1) * size]
            graph = torch.flatten(graph)
            out = self.out(graph)

            batch_out.append(out)
        batch_out = torch.stack(batch_out, dim=0)


        return batch_out.squeeze()
        #return batch_out.squeeze(1)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.737, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        #alpha = 0.85 #attempt to increase the recall?
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        #targets = torch.eye(inputs.size(-1)).to(inputs.device)[targets].to(inputs.device)


        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss

class fastGNNDataset(Dataset):
    def __init__(self, data_train, labels):
        """
        data_train: pandas.DataFrame: a pre-processed pandas dataframe containing the non-priviledged information
        extradim = extra features added to data_train for padding
        """
        self.data_train = self.graphProcess(data_train, labels)

    def graphProcess(self, data, labels):
        edge_index = self.constructGraph(data[0])
        graphs = []
        for i in tqdm(range(data.shape[0]), desc="graph construction"):
            x = torch.tensor(data[i], dtype=torch.float, requires_grad=True).unsqueeze(1)
            label = torch.eye(2)[labels[i]].squeeze(0)
            graphs.append(gnnData(x=x, edge_index=edge_index.detach().clone(), y=label))

            # if i == 1000:
            #     break
        return graphs

    def constructGraph(self, data) -> torch.tensor:
        num_features = data.shape[0]
        edge_index = (
            torch.tensor(
                [
                    (i, j)
                    for i in range(num_features)
                    for j in range(num_features)
                    if i != j
                ],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        return edge_index

    def __getitem__(self, index):
        return self.data_train[index]

    def __len__(self):
        return len(self.data_train)

class fastLUPIDataset(fastGNNDataset):
    def __init__(self, data_train, data_LUPI, labels):
        super(fastLUPIDataset, self).__init__(data_train, labels)
        self.data_LUPI = fastGNNDataset(data_LUPI, labels)

    def __getitem__(self, index):
        return self.data_train[index], self.data_LUPI[index]

    def __len__(self):
        return len(self.data_train)


class ModelGNN(ModelnData):
    def __init__(
        self,
        train_data_path=None,
        test_data_path=None,
        classify_input="wo_lab_enc",
        cluster_method="cluster_v1",
        cluster_idx=None,
        model_name=None,
        lr=0.0001,
        ls_hidden_dim=None,
        num_epochs=200,
        batch_size=32,
        is_print=True,
        is_debug=False,
        criterion=None,
        dropout_rate=0.5,
        pi_model_path=None,
        pi_train=False,
    ) -> None:
        """
        cluster_idx: 0, 1, 2, None. None means all clusters combined.
        """
        self.cluster_idx = cluster_idx
        self.model_name = model_name
        self.is_debug = is_debug
        self.is_print = is_print

        # self.LUPI_loss_mode = LUPI_loss_mode

        self.ls_hidden_dim = ls_hidden_dim
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.learning_rate = lr

        #Set the mode to train the PI model or regular model
        self.pi_train = pi_train

        if self.is_print:
            print(f"===== Model: {self.model_name} =====")
            print(
                f"   -- Cluster: Index: {self.cluster_idx}, Method: {cluster_method}, ls_hidden_dim: {self.ls_hidden_dim}, lr: {self.learning_rate}"
            )
        if train_data_path != None:
            self.train_loader = self.gen_dataloader_train_LUPI(
                mode="train",
                csv_filepath=train_data_path,
                classify_input=classify_input,
                cluster_method=cluster_method,
                cluster_idx=cluster_idx,
                batch_size=batch_size,
            )
        else:
            self.train_loader = None

        #use the PI dataset if the 
        if self.pi_train == True:
            self.test_loader = self.gen_dataloader_train_LUPI(
                mode="test",
                csv_filepath=test_data_path,
                classify_input=classify_input,
                cluster_method=cluster_method,
                cluster_idx=cluster_idx,
                batch_size=batch_size,
            )
        else:
            self.test_loader = self.gen_dataloader(
                mode="test",
                csv_filepath=test_data_path,
                classify_input=classify_input,
                cluster_method=cluster_method,
                cluster_idx=cluster_idx,
                batch_size=batch_size,
            )

        # network creation
        self.net = lupiGNN(batch_size=batch_size)
        print("===== Model created: mymodel =====")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("===== Device: " + str(self.device) + " =====")
        self.net.to(self.device)

        #set the device for the networks
        self.net.set_device(self.device)

        #if not PI mode then load the trained PI model
        if self.pi_train == False:
            assert pi_model_path is not None
            self.load_model(pi_model_path)

        #change when this is used
        if self.pi_train:
            self.optimizer = torch.optim.Adam(list(self.net.pi_encoder.parameters()) + list(self.net.decoder.parameters()), lr=self.learning_rate, weight_decay=1e-4)

            self.obs_optimizer = torch.optim.Adam(self.net.obs_encoder.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(
                self.net.obs_encoder.parameters(), lr=self.learning_rate, weight_decay=1e-4
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.99
        )

        # ## Set up the criterion
        if criterion == "focal":
            self.criterion = FocalLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 0.75]).to(self.device))


        # if wandb.config['criterion'] == "focal":
        #     self.criterion = FocalLoss(wandb.config['obs_alpha'], wandb.config['obs_gamma'])
        # else:
        #     self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 0.737]).to(self.device))
        #self.criterion = FocalLoss()
        self.pi_criterion = torch.nn.CrossEntropyLoss()

        self.train_best_record = BestRecord()
        self.test_best_record = BestRecord()


    def gen_dataloader_train_LUPI(
        self,
        mode="train",
        csv_filepath=None,
        classify_input=None,
        cluster_method=None,
        cluster_idx=None,
        batch_size=32,
    ):
        assert csv_filepath is not None
        assert classify_input is not None
        #assert mode == "train"

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
        data = df.drop("diabetes_label", axis=1).values
        if self.is_debug:
            print(labels.value_counts())
            # print(data.head())
        labels = labels.values

        # ## Load LUPI data
        df_LUPI = DataProcess(csv_filepath).data_process_cluster(
            classify_input="full_raw",
            cluster_method=cluster_method,
            cluster_idx=cluster_idx,
        )
        labels_LUPI = df_LUPI["diabetes_label"]
        data_LUPI = df_LUPI.drop("diabetes_label", axis=1).values

        assert (labels == labels_LUPI).all()

        dataset = fastLUPIDataset(data, data_LUPI, labels)

        if mode == "train":
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        elif mode == "test":
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return dataloader

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

        dataset = fastGNNDataset(data, labels)

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

    def train(self, epoch):
        self.net.train()  # Set the model to training mode

        loss_avg_epoch = 0
        pi_loss_avg_epoch = 0

        all_labels = []
        all_predictions = []

        for i, (inputs, lupi) in enumerate(self.train_loader):
            #set number of graphs per epoch

            if epoch == 50 and not self.pi_train:
                self.optimizer = torch.optim.Adam(list(self.net.obs_encoder.parameters()) + list(self.net.decoder.parameters()), lr=self.learning_rate, weight_decay=1e-4)

            
            self.net.set_graphs(inputs.num_graphs)


            inputs_edge = inputs.edge_index.to(self.device)
            
            labels = inputs.y.to(self.device)
            labels = labels.view(int(labels.size(dim=0) / 2), 2)
            labels = labels.squeeze()


            self.optimizer.zero_grad()
            if self.pi_train:
                self.obs_optimizer.zero_grad()

            r_i, r_p = self.net.latent_forward(inputs, lupi)
            
            if not self.pi_train and epoch < 50:
                r_i1 = torch.nn.functional.softmax(r_i, dim=1)
                r_p1 = torch.nn.functional.softmax(r_p, dim=1)
                lupi_loss2 = self.pi_criterion(r_i1, r_p1) * 0.07
                lupi_loss2.backward(retain_graph=True)
                pi_loss_avg_epoch += lupi_loss2.item()


                # lupi_loss = self.pi_criterion(r_i, r_p)
                # lupi_loss.backward(retain_graph=True)

            # r_i1 = torch.nn.functional.softmax(r_i, dim=1)
            # r_p1 = torch.nn.functional.softmax(r_p, dim=1)
            # lupi_loss2 = self.pi_criterion(r_i1, r_p1)
            # lupi_loss2.backward(retain_graph=True)

            # preds = self.net.forward(r_i, inputs_edge)
            # lupi_preds = self.net.forward(r_p, inputs_edge)


            # loss = self.criterion(preds, labels)
            # pi_loss = self.criterion(lupi_preds, labels)

            if self.pi_train:
                #PI Encoder backpropigation
                lupi_preds = self.net.forward(r_p, inputs_edge)
                pi_loss = self.criterion(lupi_preds, labels)

                pi_loss.backward()
                self.optimizer.step()

                preds = self.net.forward(r_i, inputs_edge)
                loss = self.criterion(preds, labels)

                self.obs_optimizer.zero_grad()
                self.optimizer.zero_grad()
                # total_loss = pi_loss + loss
                # total_loss.backward()

                #Observation Encoder backpropigation
                loss.backward()

                self.obs_optimizer.step()

                preds = lupi_preds
                loss = pi_loss
            else:
                preds = self.net.forward(r_i, inputs_edge)
                lupi_preds = self.net.forward(r_p, inputs_edge)

                loss = self.criterion(preds, labels)
                pi_loss = self.criterion(lupi_preds, labels)

                loss.backward()
                self.optimizer.step()


            if labels.dim() == 1:
                labels = labels.view(1, 2)
                preds = preds.view(1, 2)

            predicted = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

            # predicted_LUPI = torch.argmax(outputs_LUPI, dim=1)

            loss_avg_epoch += loss.item()

        loss_avg_epoch /= len(self.train_loader)
        pi_loss_avg_epoch /= len(self.train_loader)

        self.scheduler.step()

        # Calculate precision and recall
        dc_metrics = cal_metrics(all_labels, all_predictions)
        self.train_best_record.update(
            acc=dc_metrics["accuracy"],
            precision=dc_metrics["precision"],
            recall=dc_metrics["recall"],
            f1=dc_metrics["f1"],
            loss=loss_avg_epoch,
            epoch=epoch,
        )

        dc_metrics.update(
            {
                "loss": loss_avg_epoch,
                "best_acc": self.train_best_record.best_acc,
                "best_precision": self.train_best_record.best_precision,
                "best_recall": self.train_best_record.best_recall,
                "all_labels": all_labels,
                "all_predictions": all_predictions,
            }
        )
        if not self.pi_train:
            dc_metrics.update({"pi_loss" : pi_loss_avg_epoch})

        ans = dc_metrics

        if self.is_print:
            if self.model_name is not None:
                print(f"Model: {self.model_name}", end=" ")
            print(f"Epoch: {epoch} TRAIN", end=" ")
            print(f"Loss: {loss_avg_epoch:.4f}", end=" ")
            print(f"Accuracy: {dc_metrics['accuracy'] * 100:.2f}%", end=" ")
            print(f"Precision: {dc_metrics['precision'] * 100:.2f}%", end=" ")
            print(f"Recall: {dc_metrics['recall'] * 100:.2f}%", end=" ")
            print(f"Specificity: {dc_metrics['specificity'] * 100:.2f}%", end=" ")
            print(f"Best accuracy: {ans['best_acc'] * 100:.2f}%", end=" ")
            print(f"Best precision: {ans['best_precision'] * 100:.2f}%", end=" ")
            print(f"Best recall: {ans['best_recall'] * 100:.2f}%")
        return ans
    
    def test(self, epoch):
        test_loader = self.test_loader
        if self.is_debug:
            print(f"===== Test epoch: {epoch} =====")
            print(f"len(test_loader.dataset): {len(test_loader.dataset)}")

        self.net.eval()  # Set the model to evaluation mode
        loss_avg_epoch = 0
        all_labels = []
        all_predictions = []
        i = 0
        with torch.no_grad():
            for data in test_loader:

                if self.pi_train == True:
                    inputs = data[0]
                    lupi_inputs = data[1]
                    data = lupi_inputs
                else:
                    #bandaid fix
                    inputs = data

                inputs_edge = inputs.edge_index.to(self.device)
                labels = inputs.y.to(self.device)

                self.net.set_graphs(data.num_graphs)

                labels = labels.view(int(labels.size(dim=0) / 2), 2)
                labels = labels.squeeze()

                if self.pi_train == True:
                    _, r_p = self.net.latent_forward(None, data, pi_train=True)
                    outputs = self.net.forward(r_p, inputs_edge)

                else:
                    r_i, _ = self.net.latent_forward(data, None)
                    outputs = self.net.forward(r_i, inputs_edge)


                loss = self.criterion(outputs, labels)

                if labels.dim() == 1:
                    labels = labels.view(1, 2)
                    outputs = outputs.view(1, 2)

                # _, predicted = torch.max(outputs.data, 1)

                predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)


                loss_avg_epoch += loss.item()

                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())

        loss_avg_epoch /= len(self.test_loader)

        # Calculate precision and recall
        precision, recall, f1, accuracy, specificity = self.cal_metrics(all_labels, all_predictions)
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
                specificity=specificity,
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
    
class gnnLupiRunner(BaseRunner):
    def __init__(
        self,
        exp_name=None,
        cluster_method=None,
        classify_input=None,
        train_data_path=None,
        test_data_path=None,
        hidden_dim=None,
        criterion="cross_entropy",
        exp_save_dir="runs",
        lr=1e-6,
        dropout_rate=0.5,
        num_epochs=200,
        is_debug=False,
        debug_params=None,
        batch_size=32, #added below
        pi_model_path=None,
        pi_train=False,
        wandbTracking=False,
        config=None,
    ) -> None:
        """
        cluster_method: cluster_v1, cluster_v2, None. None means all clusters combined. For LUPI, cluster_method is None.
        """
        self.is_debug = is_debug
        self.debug_params = debug_params

        self.ls_hidden_dim = hidden_dim
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.learning_rate = lr

        self.wandbTracking = wandbTracking

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

        shutil.copy("main.py", os.path.join(self.save_dir, "main.py"))

        if config is not None:
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = {}

        config['train_file'] = train_data_path
        config['test_file'] = test_data_path
        config['criterion'] = criterion

        if self.wandbTracking == True:
            wandb.init(
                project = "sizeFork",
                name = self.exp_name,
                config = config
            )



        self.setup_model_and_dataloader_LUPI(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            cluster_method=cluster_method,
            classify_input=classify_input,
            ls_hidden_dim=hidden_dim,
            criterion=criterion,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            pi_model_path=pi_model_path,
            pi_train=pi_train,
        )

        # ## Set up the best record
        self.train_best_record = BestRecord()
        self.test_best_record = BestRecord()

        self.debug()

    def setup_model_and_dataloader_LUPI(
        self,
        train_data_path=None,
        test_data_path=None,
        cluster_method=None,
        classify_input=None,
        ls_hidden_dim=None,
        criterion=None,
        lr=None,
        dropout_rate=0.5,
        batch_size=32,
        num_epochs=200,
        pi_model_path=None,
        pi_train=False,
    ):
        """
        cluster_method: cluster_v1, cluster_v2, None. None means all clusters combined. For LUPI, cluster_method is None.
        """
        #assert os.path.exists(train_data_path) and os.path.exists(test_data_path)
        assert os.path.exists(test_data_path)
        assert cluster_method in ["cluster_v1", "cluster_v2", None]
        #assert cluster_method is None

        if cluster_method is None:
            # use data from all clusters combined
            print(f"Setting up model with all clusters combined")
            self.ls_mlp = [
                ModelGNN(train_data_path=train_data_path, test_data_path=test_data_path, batch_size=batch_size, num_epochs=num_epochs,
                         pi_train=pi_train, pi_model_path=pi_model_path, classify_input=classify_input)
            ]
        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")
    


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='HCSS Diabetes Diagnsosis Model')
    parser.add_argument('-i', '--input', type=str, required=False, help='Path to data file')
    args = parser.parse_args()
    input_file = args.input

    train = 'data/ada_train_victor_imbalanced_3_fold.csv'
    test = 'data/ada_test_victor_imbalanced_3_fold.csv'

    test_path_one="./data/ada_test_victor_SMOTE_1:1_encoded_3_fold.csv"
    train_path_one="./data/ada_train_victor_SMOTE_1:1_encoded_3_fold.csv"

    config_file = 'config/gnnconfig.yml'

    pretrain_path = 'config/main_pretrain.pth'

    #command line input override
    if input_file == None:
        test_file_path = test_path_one
    else:
        test_file_path = input_file

    # piRunner = gnnLupiRunner(
    #     config=config_file,
    #     train_data_path=train,
    #     test_data_path=test,
    #     batch_size=32,
    #     exp_save_dir="runs/runs_sizeFork",
    #     classify_input="wo_lab_raw",
    #     pi_train=True,
    #     wandbTracking=False,
    #     criterion='focal'
    # )
    # piRunner.run()

    # obs_focal = gnnLupiRunner(
    #     config=config_file,
    #     train_data_path=train,
    #     test_data_path=test,
    #     batch_size=32,
    #     exp_save_dir="runs/runs_sizeFork",
    #     classify_input="wo_lab_raw",
    #     pi_train=False,
    #     pi_model_path=model_path,
    #     wandbTracking=True,
    #     criterion='focal'
    # )
    # obs_focal.run()

    #one to one runner

    # obsRunner = gnnLupiRunner(
    #     config=config_file,
    #     train_data_path=train_path_one,
    #     test_data_path=test_path_one,
    #     batch_size=32,
    #     exp_save_dir="runs/runs_sizeFork",
    #     classify_input="wo_lab_raw",
    #     pi_model_path=pretrain_path,
    #     pi_train=False,
    #     wandbTracking=False,
    #     criterion='cross_entropy'
    # )
    # obsRunner.run()


    testRunner = gnnLupiRunner(
        config=config_file,
        train_data_path=None,
        test_data_path=test_file_path,
        batch_size=32,
        exp_save_dir="runs/runs_main",
        classify_input="wo_lab_raw",
        pi_model_path=pretrain_path,
        pi_train=False,
        wandbTracking=False,
        criterion='cross_entropy'
    )
    testRunner.test(epoch=1)


    
