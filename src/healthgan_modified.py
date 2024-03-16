import time
import torch
import pandas as pd
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def show_cgm_data(data_tensor, n_to_plot=2):
    '''
    Function for visualizing data. Given a tensor of 1D data, create a line plot for the given n_to_plot.
    Parameters:
        data_tensor: tensor of data
        n_to_plot: number of data samples to plot
        size: size of each data sample
    '''
    # converts to pandas dataframe
    # TODO: see if this is necessary
    data = pd.DataFrame(data_tensor.detach().cpu().numpy())
    # crop to first 48 columns
    data = data.iloc[:, :48]
    plt.figure()
    plt.xlabel("Time, in half hours")
    plt.ylabel("glucemia, mg/dL")
    labels = []
    for i in range(n_to_plot):
        data.iloc[i].plot()
        labels.append("Segment " + str(i))
    plt.legend(labels)
    plt.show()
    plt.pause(0.001)

def show_losses(generator_losses, critic_losses, step_bins):
    plt.figure(figsize=(10, 5))
    num_examples = (len(generator_losses) // step_bins) * step_bins
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Generator Loss"
    )
    plt.plot(
        range(num_examples // step_bins), 
        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Critic Loss"
    )
    plt.title("Generator and critic losses", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xlim(0, num_examples / step_bins)
    plt.legend(loc='upper right')
    plt.show()

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        data_dim: the dimension of the 1D data, fitted for the dataset used, a scalar
          (CGM data is 48 half hours in a day so that is your default)
    '''
    def __init__(self, z_dim=100, data_dim=50):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, data_dim * 2),
            self.get_generator_block(data_dim * 2, int(data_dim * 1.5)),
            nn.Linear(int(data_dim * 1.5), data_dim),
            nn.Sigmoid()
        )
    
    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim), # this is not in original HealthGAN
            nn.ReLU(inplace=True)
        )
    
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)


class Critic(nn.Module):
    '''
    Values:
        data_dim: the dimension of the data, fitted for the dataset used, a scalar
            (CGM data is 48 half hours in a day so that is the default)
        hidden_dim: the inner NN dimension, a scalar
    '''
    def __init__(self, data_dim=50, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.get_critic_block(data_dim, hidden_dim),
            self.get_critic_block(hidden_dim, hidden_dim * 2),
            self.get_critic_block(hidden_dim * 2, hidden_dim * 4),
            self.get_critic_block(hidden_dim * 4, 1, final_layer=True)
        )
        
    def get_critic_block(self, input_dim, output_dim, final_layer=False):
        '''
        Returns:
            a discriminator neural network layer, with a linear transformation 
            followed by an nn.LeakyReLU activation with negative slope of 0.2 
            (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given a data tensor, 
        returns a 1-dimension tensor representing fake/real.
        '''
        return self.crit(image)


def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with how the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = (torch.mean(gradient_norm) - 1)**2
    #### END CODE HERE ####
    return penalty


def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    return -torch.mean(crit_fake_pred)


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    return torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp


train_path = 'data\training\compiled_full_weekday_sdv.csv' # path to the training data
data = pd.read_csv(train_path, skiprows=1, header=None, dtype='float32').to_numpy()

n_epochs = 500
display_step = 500000000000000

# Original HealthGAN hyperparameters
# z_dim = 100
# crit_repeats = 5
# lr_crit = 0.0001

# lr_gen = 0.0001
# beta_1 = 0.5
# beta_2 = 0.9
# c_lambda = 10

# Custom hyperparameters
z_dim = 100
crit_repeats = 10
lr_crit = 0.0006
lr_gen = 0.0001
beta_1 = 0.5
beta_2 = 0.9
c_lambda = 10

batch_size = data.shape[0] // crit_repeats + 5 # original HealthGAN is without + 5 samples
# for original HealthGAN batch size computed as training dataset size / crit_repeats
# meaning critic will review whole epoch of train data in one step

flat_data_length = data.shape[1] # 48 half hours in a day + weekday

# GeForce GTX 1050 has compute capability 6.1, so it should be available
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Take start time
start_time = time.time()

# Load dataset as tensors
dataloader = DataLoader(
    data,
    batch_size = batch_size,
    shuffle = True
)

gen = Generator(z_dim,flat_data_length).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(beta_1, beta_2))
crit = Critic(flat_data_length).to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr_crit, betas=(beta_1, beta_2))

# print settings
print("Number of epochs: ", n_epochs)
print("Batch size: ", batch_size)
print("Learning rate for critic: ", lr_crit)
print("Learning rate for generator: ", lr_gen)
print("Lambda: ", c_lambda)
print("Critic repeats: ", crit_repeats)

cur_step = 0
generator_losses = []
critic_losses = []
fake = None
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for batch in tqdm(dataloader):
        cur_batch_size = len(batch)
        real = batch.to(device)
        
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]
        
        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)

        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()
        gen_opt.step() # Update the weights
        
        generator_losses += [gen_loss.item()] # Keep track of average loss
        
        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"\nEpoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_cgm_data(fake, 4)
            show_cgm_data(real, 4)
            show_losses(generator_losses, critic_losses, 10)
        cur_step += 1
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, progress: {epoch / n_epochs * 100}%")
        
    # if last 25% of epochs, save synthetic data every 5% of epochs
    if epoch > n_epochs * 0.75 and epoch % (n_epochs * 0.05) == 0:
        header = pd.read_csv(train_path, nrows=0)
        file_fake = f'data\generated\sd_{n_epochs}_{batch_size}_{c_lambda}_{crit_repeats}_{epoch}.csv'
        pd.DataFrame(fake.detach().cpu().numpy(), columns=header.columns).to_csv(file_fake, index=False)
        print("Synthetic data saved at " + file_fake)

model_path = 'data\model\generator.pth'
torch.save(gen.state_dict(), model_path) # Save the model

# calculate total run time in seconds
total_time = time.time() - start_time
print("Training finished - Model saved at " + model_path)
print(f"Total training time: {total_time / 60} minutes")

# Generate a batch of synthetic data, same sample amount as train data
fake_noise = get_noise(data.shape[0], z_dim, device=device)
fake = gen(fake_noise)

# save the synthetic data, take header from train file
header = pd.read_csv(train_path, nrows=0)
file_fake = f'data\generated\sd_{n_epochs}_{batch_size}_{c_lambda}_{crit_repeats}_{n_epochs}.csv'
pd.DataFrame(fake.detach().cpu().numpy(), columns=header.columns).to_csv(file_fake, index=False)
print("Synthetic data saved at " + file_fake)

show_cgm_data(fake, 4)
show_losses(generator_losses, critic_losses, 10)