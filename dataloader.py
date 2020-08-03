import numpy as np 
import pandas as pd 
from biopandas.mol2 import PandasMol2
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from scipy.spatial import distance


class Pocket2VecDataset(Dataset):
    """
    Dataset for pockets for unsupervised graph learning.
    """
    def __init__(self, data_dir, data_list_dir, features_to_use, threshold, transform=None):
        self.data_dir = data_dir
        self.pocket_list = read_list_file(data_list_dir)
        total_features = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
        assert(set(features_to_use).issubset(set(total_features))) # features to use should be subset of ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
        self.features_to_use = features_to_use
        self.threshold = threshold
        
        # hard-coded mapping
        self.hydrophobicity = {'ALA':1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,
                               'CYS':2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,
                               'HIS':-3.2,'ILE':4.5,'LEU':3.8,'LYS':-3.9,
                               'MET':1.9,'PHE':2.8,'PRO':-1.6,'SER':-0.8,
                               'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL':4.2}

        # hard-coded mapping
        self.binding_probability = {'ALA':0.701,'ARG':0.916,'ASN':0.811,'ASP':1.015,
                                    'CYS':1.650,'GLN':0.669,'GLU':0.956,'GLY':0.788,
                                    'HIS':2.286,'ILE':1.006,'LEU':1.045,'LYS':0.468,
                                    'MET':1.894,'PHE':1.952,'PRO':0.212,'SER':0.883,
                                    'THR':0.730,'TRP':3.084,'TYR':1.672,'VAL':0.884}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        pocket_name = self.pocket_list[idx]
        pocket_dir = self.data_dir + pocket_name + '/' + pocket_name + '.mol2' # modify this accordingly
        print(pocket_dir)
        graph_data = self.__read_mol(pocket_dir) # read dataframe as pytorch-geometric graph data

        ''' apply transformation if applicable '''
        #if self.transform:
        #    mol = self.transform(mol)
        
        return graph_data

    def __read_mol(self, mol_path):
        """
        Read the mol2 file as a dataframe. May include pop_path and profile_path in the future.
        """
        atoms = PandasMol2().read_mol2(mol_path)
        atoms = atoms.df[['atom_id','subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
        atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
        atoms['hydrophobicity'] = atoms['residue'].apply(lambda x: self.hydrophobicity[x])
        atoms['binding_probability'] = atoms['residue'].apply(lambda x: self.binding_probability[x])
        center_distances = self.__compute_dist_to_center(atoms[['x','y','z']].to_numpy())
        atoms['distance_to_center'] = center_distances
        siteresidue_list = atoms['subst_name'].tolist()
        #qsasa_data = self.__extract_sasa_data(siteresidue_list, pop_path)
        #atoms['sasa'] = qsasa_data
        #seq_entropy_data = self.__extract_seq_entropy_data(siteresidue_list, profile_path) # sequence entropy data with subst_name as keys
        #atoms['sequence_entropy'] = atoms['subst_name'].apply(lambda x: seq_entropy_data[x])
        
        if atoms.isnull().values.any():
            print('invalid input data (containing nan):')
            print(mol_path)
            #print(atoms)

        bonds = self.bond_parser(mol_path)

        atoms_graph = self.__form_graph(atoms, bonds, self.threshold)
        return atoms_graph

    def bond_parser(self, pocket_path):
        f = open(pocket_path,'r')
        f_text = f.read()
        f.close()
        bond_start = f_text.find('@<TRIPOS>BOND')
        bond_end = -1
        df_bonds = f_text[bond_start:bond_end].replace('@<TRIPOS>BOND\n','')
        df_bonds = df_bonds.replace('am', '1') # amide
        df_bonds = df_bonds.replace('ar', '1') # aromatic
        df_bonds = df_bonds.replace('du', '1') # dummy
        df_bonds = df_bonds.replace('un', '1') # unknown
        df_bonds = df_bonds.replace('nc', '0') # not connected
        df_bonds = df_bonds.replace('\n',' ')
        df_bonds = np.array([int(x) for x in df_bonds.split()]).reshape((-1,4)) # convert the the elements to integer
        df_bonds = pd.DataFrame(df_bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])
        df_bonds.set_index(['bond_id'], inplace=True)
        return df_bonds

    def compute_edge_attr(self, edge_index, bonds):
        """
        Compute the edge attributes according to the chemical bonds. 
        """
        sources = edge_index[0,:]
        targets = edge_index[1,:]
        edge_attr = np.zeros((edge_index.shape[1], 1))
        for index, row in bonds.iterrows():
            # find source == row[1], target == row[0]
            source_locations = set(list(np.where(sources==(row[1]-1))[0])) # minus one because in new setting atom id starts with 0
            target_locations = set(list(np.where(targets==(row[0]-1))[0]))
            edge_location = list(source_locations.intersection(target_locations))[0]
            edge_attr[edge_location] = row[2]

            # find source == row[0], target == row[1]
            source_locations = set(list(np.where(sources==(row[0]-1))[0]))
            target_locations = set(list(np.where(targets==(row[1]-1))[0]))
            edge_location = list(source_locations.intersection(target_locations))[0]
            edge_attr[edge_location] = row[2]
        return edge_attr

    def __form_graph(self, atoms, bonds, threshold):
        """
        Form a graph data structure (Pytorch geometric) according to the input data frame.
        Rule: Each atom represents a node. If the distance between two atoms are less than or 
        equal to 4.5 Angstrom (may become a tunable hyper-parameter in the future), then an 
        undirected edge is formed between these two atoms. 
        
        Input:
        atoms: dataframe containing the 3-d coordinates of atoms.
        bonds: dataframe of chemical bonds represented as edges of the graph.
        threshold: distance threshold to form the edge (chemical bond).
        
        Output:
        A Pytorch-gemometric graph data with following contents:
            - node_attr (Pytorch Tensor): Node feature matrix with shape [num_nodes, num_node_features]. e.g.,
              x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
            - edge_index (Pytorch LongTensor): Graph connectivity in COO format with shape [2, num_edges*2]. e.g.,
              edge_index = torch.tensor([[0, 1, 1, 2],
                                         [1, 0, 2, 1]], dtype=torch.long)
        
        Forming the final output graph:
            data = Data(x=x, edge_index=edge_index)
        """
        A = atoms.loc[:,'x':'z'] # sample matrix
        A_dist = distance.cdist(A, A, 'euclidean') # the distance matrix
        threshold_condition = A_dist > threshold # set the element whose value is larger than threshold to 0
        A_dist[threshold_condition] = 0 # set the element whose value is larger than threshold to 0
        result = np.where(A_dist > 0)
        result = np.vstack((result[0],result[1]))
        edge_attr = self.compute_edge_attr(result, bonds)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_index = torch.tensor(result, dtype=torch.long)
        node_features = torch.tensor(atoms[self.features_to_use].to_numpy(), dtype=torch.float32)
        #label = torch.tensor([label], dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data   

    def __compute_dist_to_center(self, data):
        """
        Given the input data matrix (n by d), return the distances of each points to the
        geometric center.
        """
        center = np.mean(data, axis=0)
        shifted_data = data - center # center the data around origin
        distances = np.sqrt(shifted_data[:,0]**2 + shifted_data[:,1]**2 + shifted_data[:,2]**2) # distances to origin
        return distances        


def read_list_file(file_dir):
    """
    Read the file containing pocket names.
    """
    f = open(file_dir, 'r')
    pocket_list = f.read()
    f.close()
    pocket_list = pocket_list.split('\n')
    
    while('' in pocket_list): 
        pocket_list.remove('')
    
    return pocket_list


if __name__=="__main__":
    """
    main function used for testing only.
    """ 
    pocket_dir = '../../siamese-monet-project/data/googlenet-dataset/'
    pocket_list_dir = '../../siamese-monet-project/data/googlenet.lst'
    # features to use should be subset of ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center', 'sasa', 'sequence_entropy']
    features_to_use = ['charge', 'hydrophobicity', 'binding_probability', 'distance_to_center']
    threshold = 4.5

    dataset = Pocket2VecDataset(data_dir=pocket_dir, data_list_dir=pocket_list_dir, features_to_use=features_to_use, threshold=threshold)
    print(dataset[0])


