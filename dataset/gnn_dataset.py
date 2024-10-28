import pandas as pd
import torch
from torch_geometric.data import Data as gnnData
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class corrgnnDataset(Dataset):
    '''
    Constructs a GNN dataset given a correlation matrix of features to be used for
    specifying edges. Can be used to construct regular and LUPI datasets
    '''
    def __init__(self, data_train, data_lupi, labels, cor_df: pd.DataFrame, df_columns: list, LUPI=True):
        '''
        data_train: numpy array of values for training data
        data_lupi: numpy array of privileged information
        cor_df: pandas dataframe correlation matrix with labels [Low, Medium, High, Perfect]
        df_columns: list of columns names for the data_train matrix
        LUPI: whether or not privileged information is included
        '''
        if LUPI == True:
            self.LUPI = True
        else:
            self.LUPI = False

        #can be set on what to include in edge matrix
        self.acceptable = {'Medium', 'High'}

        #corrmat
        cor_df_columns = list(cor_df.columns)
        self.df_to_corrmat = {}
        for i, column in enumerate(df_columns):
            # print(i, column)
            if column in cor_df_columns:
                self.df_to_corrmat[i] = cor_df_columns.index(column)
            else:
                self.df_to_corrmat[i] = -1

        self.data_train = self.graphProcess(data_train, labels, cor_df)
        if self.LUPI == True:
            self.data_lupi = self.graphProcess(data_lupi, labels, cor_df)

    def graphProcess(self, data, labels, cor_df) -> list[gnnData]:
        edge_index = self.constructGraph(data[0], cor_df)
        graphs = []
        for i in tqdm(range(data.shape[0]), desc="graph construction"):
            x = torch.tensor(data[i], dtype=torch.float, requires_grad=True).unsqueeze(1)
            label = torch.eye(2)[labels[i]].squeeze(0)
            graphs.append(gnnData(x=x, edge_index=edge_index.detach().clone(), y=label))
            # if i == 1000:
            #     break
        return graphs

    def constructGraph(self, data, cor_df: pd.DataFrame) -> torch.Tensor:
        '''
        Creates the Edge Index list.
        '''
        num_features = data.shape[0]
        edge_index = (
            torch.tensor(
                [
                    (i, j)
                    for i in range(num_features)
                    for j in range(num_features)
                    if i != j and cor_df.iloc[self.df_to_corrmat[i], self.df_to_corrmat[j]] in self.acceptable
                ],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        return edge_index

    def __getitem__(self, index):
        if self.LUPI:
            return self.data_train[index], self.data_lupi[index]
        else:
            return self.data_train[index]

    def __len__(self):
        return len(self.data_train)

class fastGNNDataset(Dataset):
    '''
    Creates a dataset a fully connected GNN dataset based on a dataset and labels
    '''
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
    '''
    Constructs an LUPI Dataset. Returns 2 GNN data batches of PI info and
    regular info
    '''
    def __init__(self, data_train, data_LUPI, labels):
        super(fastLUPIDataset, self).__init__(data_train, labels)
        self.data_LUPI = fastGNNDataset(data_LUPI, labels)

    def __getitem__(self, index):
        return self.data_train[index], self.data_LUPI[index]

    def __len__(self):
        return len(self.data_train)

class fastGNNLUPIDataset(fastGNNDataset):
    def __init__(self, data_train, data_LUPI, labels):
        super(fastGNNLUPIDataset, self).__init__(data_train, labels)
        self.data_LUPI = fastGNNDataset(np.concatenate((data_train, data_LUPI), axis=1), labels)

    def __getitem__(self, index):
        return self.data_train[index], self.data_LUPI[index]

    def __len__(self):
        return len(self.data_train)


#Old GNN datasetClasses
class gnnDataset(Dataset):
    def __init__(self, data_train, labels, extradim=0):
        """
        data_train: pandas.DataFrame: a pre-processed pandas dataframe containing the non-priviledged information
        extradim = extra features added to data_train for padding
        """
        self.data_train = self.graphProcess(data_train, labels, extradim=extradim)

    def graphProcess(self, data, labels, extradim=0):
        graphs = []
        for i in tqdm(range(data.shape[0]), desc="graph construction"):
            graphs.append(self.constructGraph(data[i], labels[i], extradim=extradim))
            # if i == 1000:
            #     break
        return graphs

    def constructGraph(self, data, label, extradim=0):
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
        if extradim == 0:
            x = torch.tensor(data, dtype=torch.float, requires_grad=True).unsqueeze(1)
        else:
            x = torch.tensor(np.concatenate((data, np.zeros(extradim))), dtype=torch.float, requires_grad=True).unsqueeze(1)

        label = torch.eye(2)[label].squeeze(0)

        data = gnnData(x=x, edge_index=edge_index, y=label)
        return data

    def __getitem__(self, index):
        return self.data_train[index]

    def __len__(self):
        return len(self.data_train)  

class gnnLUPIDataset(gnnDataset):
    def __init__(self, data_train, data_LUPI, labels):
        super(gnnLUPIDataset, self).__init__(data_train, labels)
        self.data_LUPI = gnnDataset(data_LUPI, labels)

    def __getitem__(self, index):
        return self.data_train[index], self.data_LUPI[index]

    def __len__(self):
        return len(self.data_train)

class gnnLUPIFCNDataset(gnnDataset):
    def __init__(self, data_train, data_LUPI, labels):
        super(gnnLUPIFCNDataset, self).__init__(
            np.concatenate((data_train, data_LUPI), axis=1), labels
        )


    def __getitem__(self, index):
        return self.data_train[index]

    def __len__(self):
        return len(self.data_train)