#imports
import argparse
import os
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from datetime import datetime
import time

# Can run this model training engine from CLI with "python TrainModel.py --<specify_optional_parameters>"

# Class defines custom loss function for orientation
# Method computes orientation of the scalar field in the graph neural network using taylor series approximation, 
# then computes loss based on the difference with the user input orientation data
# Returns: orientation loss and average angular difference
class CustomOrientationLoss(torch.nn.Module):
    def __init__(self):
        super(CustomOrientationLoss, self).__init__()

    def forward(self, out_scalar_field, orientations_data, mask):
        nodes = np.where(mask.cpu())[0]
        edge_indexes = orientations_data.edge_index
        loss = 0.0
        avg_ang_diff = 0.0
        for node in nodes:
            # Initialize a mask for adjacent nodes in the 1-hop neighborhood
            adjacent_nodes_mask = np.zeros(len(orientations_data.x), dtype=bool)
            edges_with_given_node = (edge_indexes == node).any(dim=0)
            # Extract indices of nodes that share an edge with the given node
            adjacent_nodes_pairs = edge_indexes[:, edges_with_given_node]

            for node1, node2 in adjacent_nodes_pairs.T:
                adjacent_node = node2.item() if node1 == node else node1.item()
                # Add adjacent nodes to the mask
                adjacent_nodes_mask[adjacent_node] = True

            #Pv: matrix of x,y,z coords of one-hop neighborhood Nv nodes relative to current node
            Pv = (orientations_data.x[adjacent_nodes_mask] - orientations_data.x[node]).T 
            #Sv: matrix of scalar field values output from neural network for one-hop neighbor Nv nodes relative to current node
            Sv = out_scalar_field[adjacent_nodes_mask] - out_scalar_field[node]
            Zv = Pv @ Sv # matrix multiplication for taylor series approximation of scalar field orientation at current node u
            Av = orientations_data.y[node] # measured orientation at current node u
            # Compute L1 norm
            norm_Zv = torch.norm(Zv).item()
            norm_Av = torch.norm(Av).item()
            # Calculate difference between predicted and measured orientation at current node
            cos_ang_diff = torch.dot(Av.float(),Zv).item() / (norm_Av * norm_Zv)
            ang_diff = math.degrees(math.acos(cos_ang_diff)) # Don't need, temporary for debugging
            avg_ang_diff += ang_diff
            # Add to incremental loss
            loss += 1 - abs(cos_ang_diff)
        
        node_count = len(nodes)

        if node_count > 0:
            avg_ang_diff /= node_count
            loss /= node_count

        return loss, avg_ang_diff
    
#GNN
class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    # part 1 of the network - scalar field regression
    self.sage1 = SAGEConv(dim_in, dim_h)
    self.sage2 = SAGEConv(dim_h, dim_h)
    self.sage3 = SAGEConv(dim_h, 1) #scalar field output

    # part 2 of the network - rock unit classification
    self.sage4 = SAGEConv(1, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)
    self.dim_h = dim_h
    self.dim_out = dim_out

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = self.sage2(h, edge_index)
    h = torch.relu(h)
    h = self.sage3(h, edge_index)

    out_scalar_field = h[:,0]

    h = self.sage4(h, edge_index)

    out_rock_unit = F.log_softmax(h, dim=1)

    return out_rock_unit, out_scalar_field

  def fit(self, data_rock_unit, data_scalar_field, data_orientations, epochs):
    criterion_rock_unit = torch.nn.CrossEntropyLoss() # rock unit
    criterion_scalar_field = torch.nn.MSELoss() # scalar field
    criterion_orientation = CustomOrientationLoss() # orientation
    optimizer = self.optimizer
    train_losses = []
    val_losses = []
    self.train()

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        val_loss = 0
        val_acc = 0

        optimizer.zero_grad()
        out_rock_unit, out_scalar_field = self(data_rock_unit.x, data_rock_unit.edge_index)  # Perform a single forward pass.

        # Compute the loss solely based on the training nodes.
        loss_scalar_field = criterion_scalar_field(out_scalar_field[data_scalar_field.train_mask], data_scalar_field.y[data_scalar_field.train_mask])  
        if math.isnan(loss_scalar_field):
          loss_scalar_field = 0 #handle case of no contact input data
        acc_scalar_field = accuracy_scalar_field(out_scalar_field[data_scalar_field.train_mask],data_scalar_field.y[data_scalar_field.train_mask])

        loss_rock_unit = criterion_rock_unit(out_rock_unit[data_rock_unit.train_mask], data_rock_unit.y[data_rock_unit.train_mask])
        acc_rock_unit = accuracy_rock_unit(out_rock_unit[data_rock_unit.train_mask].argmax(dim=1),data_rock_unit.y[data_rock_unit.train_mask])

        # Pass scalar field to compute orientations, return orientation loss and average angular difference
        loss_orientation, avg_ang_diff = criterion_orientation(out_scalar_field, data_orientations, data_orientations.train_mask)

        total_loss = loss_rock_unit * args.rock_unit_weight_factor + loss_scalar_field * args.geologic_contact_weight_factor + loss_orientation * args.orientation_weight_factor
        total_accuracy = acc_rock_unit + acc_scalar_field

        # (loss_classification + loss_regression.backward() for performance in future, consider)
        total_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients

        #validation
        val_loss_scalar_field = criterion_scalar_field(out_scalar_field[data_scalar_field.val_mask], data_scalar_field.y[data_scalar_field.val_mask])
        if math.isnan(val_loss_scalar_field):
          val_loss_scalar_field = 0 #handle case of no contact input data
        val_acc_scalar_field = accuracy_scalar_field(out_scalar_field[data_scalar_field.val_mask],data_scalar_field.y[data_scalar_field.val_mask])
        val_loss_rock_unit = criterion_rock_unit(out_rock_unit[data_rock_unit.val_mask], data_rock_unit.y[data_rock_unit.val_mask])
        val_acc_rock_unit = accuracy_rock_unit(out_rock_unit[data_rock_unit.val_mask].argmax(dim=1),data_rock_unit.y[data_rock_unit.val_mask])
        val_loss_orientation, val_avg_ang_diff = criterion_orientation(out_scalar_field, data_orientations, data_orientations.val_mask)

        val_loss = val_loss_rock_unit * args.rock_unit_weight_factor + val_loss_scalar_field * args.geologic_contact_weight_factor + val_loss_orientation * args.orientation_weight_factor
        val_acc = val_acc_rock_unit + val_acc_scalar_field

        train_losses.append(total_loss.item())
        val_losses.append(val_loss.item())

        # Print header
        if epoch == 0:
          current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          print(f'{current_time} - Starting training...')
          print('Starting training...')
          print(f'      || TRAIN                                                                           || VALIDATION')
          print(f'Epoch || Loss Rock Unit | Loss Scalar | Loss Ori | Acc Rock Unit | MSE Scalar | Ang Diff || Loss Rock Unit | Loss Scalar | Loss Ori | Acc Rock Unit | MSE Scalar | Ang Diff')

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
          s = '%5.0f || %14.3f | %11.3f | %8.3f | %12.2f%% | %9.2f%% | %5.0fdeg || '%(epoch,loss_rock_unit,loss_scalar_field,loss_orientation,acc_rock_unit,acc_scalar_field,avg_ang_diff)
          s += '%14.3f | %11.3f | %8.3f | %12.2f%% | %9.2f%% | %5.0fdeg'%(val_loss_rock_unit,val_loss_scalar_field,val_loss_orientation,val_acc_rock_unit,val_acc_scalar_field,val_avg_ang_diff)
          print(s)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{current_time} - Finished training...')
    return train_losses,val_losses

def accuracy_rock_unit(pred_y, y):
    """Calculate classification accuracy."""
    if len(y) == 0:
      return 0
    else:
      return ((pred_y == y).sum() / len(y)).item()

def accuracy_scalar_field(pred_y, y):
    """Calculate regression accuracy."""
    if len(y) == 0:
      return 0
    else:
      range = torch.max(y) - torch.min(y)
      MSE = (((pred_y - y)/range) ** 2).mean().item()
      return MSE

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy_rock_unit(out.argmax(dim=1)[data.val_mask], data.y[data.val_mask])
    return acc

def evaluate(model, data):
    """Evaluate the model and return prediction results."""
    model.eval()
    out1, out2 = model(data.x, data.edge_index)
    return out1, out2

# MAIN
def main(args):
    
    start_time = time.time()

    print('GPU Check:')
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load prepared input Data from file
    data_rock_unit = torch.load(os.path.join(args.input_dir, args.data_file_rock_unit))
    data_scalar_field = torch.load(os.path.join(args.input_dir, args.data_file_scalar_field))
    data_orientations = torch.load(os.path.join(args.input_dir, args.data_file_orientations))

    # Create PyTorch Geometric GraphSAGE Model
    num_features = data_rock_unit.num_features
    num_classes = torch.unique(data_rock_unit.y).size(0) # number of rock unit classes
    print(f'Number of features: {num_features}')
    print(f'Number of classes: {num_classes}')
    graph_sage = GraphSAGE(num_features,args.hidden_layer_size,num_classes)
    graph_sage = graph_sage.to(device)

    # Train
    train_losses,val_losses = graph_sage.fit(data_rock_unit.to(device), data_scalar_field.to(device), data_orientations.to(device), args.num_epoch)

    # Save the model and key model outputs
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{current_time} - Saving model and outputs...')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(graph_sage, os.path.join(args.output_dir, "trained_model.pt"))
    torch.save(train_losses,os.path.join(args.output_dir, "train_losses.vec"))
    torch.save(val_losses,os.path.join(args.output_dir, "val_losses.vec"))

    # Calculate and print the elapsed time
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time_seconds))
    print(f'Total ellapsed time: {elapsed_time_formatted}')

def parse_args():
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{current_time} - Parsing input arguments...')
    
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments

    # Input and Output files
    parser.add_argument("--input-dir",default="Intermediate_Data/",type=str,help="Directory containing input data")

    parser.add_argument("--output-dir",default="Output_Data/",type=str,help="Directory to save model output")

    parser.add_argument(
        "--data-file-rock-unit",
        default="data_rock_unit.dat",
        type=str,
        help="Name of the rock unit input data file prepared for the model, in pytorch Data format",
    )

    parser.add_argument(
        "--data-file-scalar-field",
        default="data_scalar_field.dat",
        type=str,
        help="Name of the scalar field (geologic contacts) input data file prepared for model, in pytorch Data format",
    )

    parser.add_argument(
        "--data-file-orientations",
        default="data_orientations.dat",
        type=str,
        help="Name of the orientations input data file prepared for model, in pytorch Data format",
    )

    # Model parameters
    parser.add_argument("--num-epoch",default=600,type=int,help="Number of epochs (iterations)")
    parser.add_argument("--hidden-layer-size",default=128,type=int,help="Number of neurons in each hidden layer")
    # Relative weighting of the three input types in the loss function, does not need to sum to 1:
    parser.add_argument("--rock-unit-weight-factor",default=0.2,type=float,help="Relative weighting to apply to the rock unit loss function")
    parser.add_argument("--geologic-contact-weight-factor",default=0.2,type=float,help="Relative weighting to apply to the scalar field loss function")
    parser.add_argument("--orientation-weight-factor",default=0.6,type=float,help="Relative weighting to apply to the orientations loss function")
    # Proportions of data for training vs validation dataset, should sum to 1:
    parser.add_argument("--orientations-weight-factor",default=0.6,type=float,help="Relative weighting to apply to the orientations loss function")

    # parse args
    args = parser.parse_args()
    return args
    
# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)