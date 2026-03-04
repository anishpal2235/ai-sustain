import kagglehub

# Download latest version
path = kagglehub.dataset_download("rishabhsnip/earth-observation-delhi-airshed")

print("Path to dataset files:", path)
