import os
import pandas as pd
import numpy as np
from tsl.datasets.prototypes import TabularDataset

class IJCNNHydrologyDataset(TabularDataset):
    """IJCNN Hydrology Dataset, adapted to TSL format"""
    
    def __init__(self, adj_matrix_path, discharge_data_path, root=None, 
                 freq='h', missing_values=np.nan, name='IJCNN_Hydrology'):
        """
        Initialize the hydrology dataset
        
        Args:
            adj_matrix_path: Path to the adjacency matrix file
            discharge_data_path: Path to the discharge data file
            root: Root directory for the data
            freq: Time frequency, default is 'h' (hourly)
            missing_values: Indicator for missing values
            name: Name of the dataset
        """
        self.adj_matrix_path = adj_matrix_path
        self.discharge_data_path = discharge_data_path
        self.freq = freq
        self.missing_values = missing_values
        
        # Load and process data
        df, dist, mask = self.load_data()
        
        # Pass only the basic parameters supported by TabularDataset
        super().__init__(target=df, name=name)
        
        # Manually set other attributes
        self.dist = dist
        self.freq = freq
        self.mask = mask
        self.missing_values = missing_values
    
    def load_data(self):
        """Load and process raw data"""
        print("Loading hydrology data...")
        
        # 1. Load adjacency matrix
        print("Loading adjacency matrix...")
        adj_matrix = pd.read_csv(self.adj_matrix_path, index_col=0)
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        
        # 2. Load discharge data
        print("Loading discharge data...")
        discharge_data = pd.read_csv(self.discharge_data_path, index_col=0)
        print(f"Discharge data shape: {discharge_data.shape}")
        
        # 3. Check and align the number of nodes
        n_adj_nodes = adj_matrix.shape[0]
        n_data_nodes = discharge_data.shape[1]
        
        if n_adj_nodes != n_data_nodes:
            print(f"Warning: Number of nodes in adjacency matrix ({n_adj_nodes}) != number of nodes in discharge data ({n_data_nodes})")
            # Use the smaller number of nodes
            n_nodes = min(n_adj_nodes, n_data_nodes)
            print(f"Using the first {n_nodes} nodes")
            
            # Truncate data
            adj_matrix = adj_matrix.iloc[:n_nodes, :n_nodes]
            discharge_data = discharge_data.iloc[:, :n_nodes]
            
            print(f"Adjusted adjacency matrix shape: {adj_matrix.shape}")
            print(f"Adjusted discharge data shape: {discharge_data.shape}")
        
        # 4. Process discharge data into TSL format
        df = self.prepare_target_data(discharge_data)
        
        # 5. Convert adjacency matrix to distance matrix
        dist = self.prepare_distance_matrix(adj_matrix)
        
        # 6. Create missing value mask
        mask = self.create_mask(df)
        
        return df, dist, mask
    
    def prepare_target_data(self, discharge_data):
        """Convert discharge data to TSL target format"""
        print("Processing discharge data into TSL format...")
        
        # Ensure there is a time index
        if not isinstance(discharge_data.index, pd.DatetimeIndex):
            # If no time index, create one
            print("Creating time index...")
            start_date = '2020-01-01'  # Modify as needed
            time_index = pd.date_range(start=start_date, 
                                       periods=len(discharge_data), 
                                       freq=self.freq)
            discharge_data.index = time_index
        
        # TSL expects the format: MultiIndex with (nodes, channels)
        # Restructure the data
        n_nodes = discharge_data.shape[1]
        n_times = discharge_data.shape[0]
        
        # Create MultiIndex columns (nodes, channels)
        nodes = discharge_data.columns
        channels = [0]  # Single feature (discharge)
        
        multi_columns = pd.MultiIndex.from_product([nodes, channels], 
                                                   names=['nodes', 'channels'])
        
        # Reorganize data
        df = pd.DataFrame(index=discharge_data.index, columns=multi_columns)
        
        for node in nodes:
            df[(node, 0)] = discharge_data[node].values
        
        print(f"Target data shape: {df.shape}")
        print(f"Number of nodes: {n_nodes}")
        print(f"Number of time steps: {n_times}")
        
        return df
    
    def prepare_distance_matrix(self, adj_matrix):
        """Convert adjacency matrix to distance matrix"""
        print("Processing adjacency matrix into distance matrix...")
        
        # Ensure the matrix is a numpy array
        if isinstance(adj_matrix, pd.DataFrame):
            adj_values = adj_matrix.values
        else:
            adj_values = adj_matrix
        
        print(f"Adjacency matrix value range: {np.min(adj_values)} to {np.max(adj_values)}")
        print(f"Unique values: {np.unique(adj_values)}")
        
        # Check if the adjacency matrix has connections
        n_connections = np.sum(adj_values > 0)
        if n_connections == 0:
            print("Warning: Adjacency matrix has no connections! Creating a distance matrix based on geographical positions...")
            # Create a more reasonable distance matrix
            n = adj_values.shape[0]
            dist_matrix = np.full((n, n), np.inf)
            np.fill_diagonal(dist_matrix, 0.0)
            
            # Method 1: Create grid-like connections (each node connects to nearby nodes)
            for i in range(n):
                # Connect to nearby nodes (simulate upstream and downstream relationships)
                for j in range(max(0, i-2), min(n, i+3)):
                    if i != j:
                        dist_matrix[i, j] = abs(i - j)
            
            print("Created a distance matrix based on sequence positions")
        else:
            # Convert adjacency matrix to distance matrix
            # If adjacency matrix is a 0/1 matrix, convert 1 to distance 1, 0 to infinity
            dist_matrix = np.where(adj_values > 0, 1.0, np.inf)
            np.fill_diagonal(dist_matrix, 0.0)
        
        print(f"Distance matrix shape: {dist_matrix.shape}")
        print(f"Number of finite connections: {np.sum(np.isfinite(dist_matrix)) - dist_matrix.shape[0]}")
        
        return dist_matrix
    
    def create_mask(self, df):
        """Create a missing value mask"""
        if self.missing_values is not None:
            mask = df.isna()
            print(f"Number of missing values: {mask.sum().sum()}")
            return mask
        return None

# Example usage and test function
def test_hydrology_dataset():
    """Test hydrology dataset loading"""
    try:
        # Use your data file paths
        adj_path = "./IJCNN/adj_matrix.csv"
        discharge_path = "./IJCNN/merged_discharge_data.csv"
        
        print("Creating IJCNN Hydrology Dataset...")
        dataset = IJCNNHydrologyDataset(
            adj_matrix_path=adj_path,
            discharge_data_path=discharge_path,
            freq='h'  # Use lowercase 'h' to avoid warnings
        )
        
        print("\n✅ Dataset created successfully!")
        print("="*50)
        print(f"Dataset name: {dataset.name}")
        print(f"Data shape: {dataset.target.shape}")
        print(f"Number of nodes: {dataset.n_nodes}")
        print(f"Number of time steps: {dataset.length}")
        print(f"Number of features: {dataset.n_channels}")
        print(f"Distance matrix shape: {dataset.dist.shape}")
        print(f"Time frequency: {dataset.freq}")
        
        print("\nFirst 5 rows of data:")
        print(dataset.target.head())
        
        print("\nDistance matrix summary:")
        finite_distances = dataset.dist[np.isfinite(dataset.dist) & (dataset.dist > 0)]
        if len(finite_distances) > 0:
            print(f"Minimum distance: {np.min(finite_distances)}")
            print(f"Maximum distance: {np.max(finite_distances)}")
            print(f"Average distance: {np.mean(finite_distances):.2f}")
        else:
            print("No finite non-zero distances")
        
        print("\n🎉 Your hydrology dataset is now ready to use in TSL!")
        print("It can be used to train various spatiotemporal graph neural network models, such as:")
        print("- DCRNN")
        print("- GraphWaveNet") 
        print("- STGCN")
        print("- GTS")
        
        return dataset
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_hydrology_dataset()