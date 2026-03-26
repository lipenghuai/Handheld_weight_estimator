import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from PointCloudProcessor import PointCloudProcessor
from model import PointNetAutoencoder
from train import Train

def plot_point_cloud(points, title="Point Cloud"):
    """
    Plots a 3D point cloud.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    plt.show()
    
def main():
    # Set up basic command-line argument parsing
    parser = argparse.ArgumentParser(description="Process and visualize point clouds.")
    parser.add_argument('-c', '--config', required=True, help='Path to the config JSON file')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as configFile:
        config = json.load(configFile)

    print('Config file loaded!\n')

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize the PointCloudProcessor
    processor = PointCloudProcessor(config["folder_dir"])

    # Optionally display loaded point clouds
    if config.get("show_loaded_pointclouds", "False").lower() == "true":
        for i, point_cloud in enumerate(processor.point_clouds):
            processor.plot_point_cloud(point_cloud, title=f"Point Cloud {i + 1}")
            
    processor.plot_original_sizes_distribution()
    
    min_points = config.get("min_points", "")  
    if not min_points:  # This checks both for an empty string and for the key not existing
        min_points = processor.min_points  # Use the processor's min_points if config's min_points is empty or not set
    else:
        min_points = int(min_points)  
    print(f"Minimum points: {min_points}")
    
    # Initialize and train the model on all point clouds
    print("Training on all point clouds...")
    trainer = Train()

    model_trained_on_all_pointclouds = trainer.train_and_visualize_on_multiple_point_clouds(
        processor.point_clouds, 
        num_points=min_points,
        latent_size=config["train_dim"], 
        epochs=config["epochs"], 
        lr=config["lr"], 
        visualize_every_n_epochs=config["visualize_every_n_epochs"], 
        condition=config.get("visualize_training", "False").lower() == "true"
    )

    # Test the model on a specific point cloud
    print("Testing model on a specific point cloud...")
    test_point_cloud = processor.load_point_cloud_from_ply(config["test_pointcloud_path"])
    test_point_cloud = processor.resample_point_cloud(test_point_cloud, min_points)  # Resample to match min_points

    latent_representation = trainer.encode_point_cloud(model_trained_on_all_pointclouds, test_point_cloud.to(device))
    print("Latent vector shape of the test point cloud:", latent_representation.shape)
    print("Latent vector representation of the test point cloud:", latent_representation)
    
    # After obtaining latent_representation...
    print("Latent vector shape of the test point cloud:", latent_representation.shape)
    
    # Ensure latent_representation is a PyTorch tensor
    if isinstance(latent_representation, np.ndarray):
        latent_representation = torch.tensor(latent_representation).float().to(device)
    
    # Decode latent representation back to point cloud
    reconstructed_point_cloud = trainer.decode_latent_representation(
        model_trained_on_all_pointclouds, latent_representation.unsqueeze(0))  # Now you can unsqueeze

    # Convert to numpy for plotting
    reconstructed_point_cloud_np = reconstructed_point_cloud.squeeze().cpu().numpy()

    # Plot the original test point cloud and the reconstructed one
    test_point_cloud_np = test_point_cloud.cpu().numpy()
    plot_point_cloud(test_point_cloud_np, title="Original Test Point Cloud")
    plot_point_cloud(reconstructed_point_cloud_np, title="Reconstructed from Latent Representation")



    # Example for plotting loss convergence for different latent sizes (if implemented in your Train class)
    latent_sizes = config["test_dims"]
    point_cloud_index = 0  # For demonstration; adjust as needed
    trainer.plot_loss_convergence_for_latent_sizes(
        latent_sizes, 
        min_points,
        processor.point_clouds, 
        point_cloud_index, 
        config["epochs"]
    )
    
    trainer.plot_loss_convergence_for_latent_sizes_using_log(
        latent_sizes, 
        min_points,
        processor.point_clouds, 
        point_cloud_index, 
        config["epochs"]
    )

if __name__ == "__main__":
    main()
