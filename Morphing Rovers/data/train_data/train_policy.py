import torch
import numpy as np
from math import floor
import torch.nn as nn
import torch.nn.functional as F

# Parameters for the rover modes
MASK_SIZE = 11  # size of mask (11x11 matrix)
NUMBER_OF_MODES = 4  # amount of rover modes
NUM_MODE_PARAMETERS = NUMBER_OF_MODES * MASK_SIZE ** 2
MASK_CENTRES = []  # indices of mask centres, when stored in 1 single array
for m_id in range(NUMBER_OF_MODES):
    MASK_CENTRES.append(int(m_id * MASK_SIZE ** 2 + 0.5 * MASK_SIZE ** 2))

# Size and field of view of rover
FIELD_OF_VIEW = int(MASK_SIZE / 2 + 1)
VISIBLE_SIZE = int(8 * MASK_SIZE)  # size of the cutout, which the rover sees
MIN_BORDER_DISTANCE = int(0.6 * VISIBLE_SIZE)  # minimal distance to border of map

# Cooldown of morphing
MODE_COOLDOWN = int(VISIBLE_SIZE / MASK_SIZE)

# Minimum distance when sample is counted as collected
SAMPLE_RADIUS = FIELD_OF_VIEW

# Parameters of the neural network controlling the rover
NETWORK_SETUP = {'filters': 8,
                 'kernel_size': MASK_SIZE,
                 'stride': 2,
                 'dilation': 1,
                 'filters1': 16,
                 'kernel_size1': 4,
                 'pooling_size': 2,
                 'state_neurons': 40,
                 'hidden_neurons': [40, 40]}

# Rover dynamics
DELTA_TIME = 1  # Step of simulation
MAX_TIME = 500  # max simulation time
SIM_TIME_STEPS = int(MAX_TIME / DELTA_TIME)
MAX_VELOCITY = MASK_SIZE
MAX_DV = DELTA_TIME * MAX_VELOCITY  # max step size of the rover
MAX_ANGULAR_VELOCITY = np.pi / 4.
MAX_DA = DELTA_TIME * MAX_ANGULAR_VELOCITY

# Number of maps and scenarios per map
TOTAL_NUM_MAPS = 6  # maps in ./data/maps
MAPS_PER_EVALUATION = 6  # Numer of maps used
SCENARIOS_PER_MAP = 5  # 5 samples on each map
TOTAL_NUM_SCENARIOS = MAPS_PER_EVALUATION * SCENARIOS_PER_MAP
# Kernel size for smoothing maps a little bit with a Gaussian kernel
BLUR_SIZE = 7

# Constants used for numerical stability and parameter ranges
EPS_C = (0.03) ** 2
FLOAT_MIN = -100  # min and max values for parameters of neuronal net
FLOAT_MAX = 100
CENTRE_MIN = 1e-16

# Initialising constants for extracting the map terrain the rover is on
VIEW_LEFT = int(VISIBLE_SIZE / 2)
VIEW_RIGHT = VIEW_LEFT + 1
MODE_VIEW_LEFT = int(MASK_SIZE / 2)
MODE_VIEW_RIGHT = MODE_VIEW_LEFT + 1


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    '''This Function gets the input, the kernel with its size, dilation and stride and gives out the size of the output'''
    '''
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)

    From https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    '''
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def get_conv_size(network_setup):
    '''
    Function returning the layer size after two convolutions in a neural network.
    '''
    cwidth, cheight = conv_output_shape([VISIBLE_SIZE + 1, VISIBLE_SIZE + 1],
                                        network_setup['kernel_size'],
                                        network_setup['stride'],
                                        0,
                                        network_setup['dilation'])
    cwidth, cheight = conv_output_shape([cwidth, cheight],
                                        network_setup['pooling_size'],
                                        network_setup['pooling_size'],
                                        0,
                                        1)
    cwidth, cheight = conv_output_shape([cwidth, cheight],
                                        network_setup['kernel_size1'],
                                        network_setup['stride'],
                                        0,
                                        network_setup['dilation'])
    cwidth, cheight = conv_output_shape([cwidth, cheight],
                                        network_setup['pooling_size'],
                                        network_setup['pooling_size'],
                                        0,
                                        1)
    conv_size = cwidth * cheight * network_setup['filters1']
    return conv_size


def get_number_of_parameters(network_setup):
    '''
    Function returning the number of biases, weights and size of the convolutional layer given a neural network setup.
    '''
    number_biases = 2 + network_setup['filters'] + network_setup['filters1'] + network_setup['state_neurons'] + \
                    network_setup['hidden_neurons'][0] + network_setup['hidden_neurons'][1]

    conv_size = get_conv_size(network_setup)

    number_weights = conv_size * network_setup['hidden_neurons'][0] + \
                     network_setup['state_neurons'] * network_setup['hidden_neurons'][0] + \
                     network_setup['hidden_neurons'][0] * network_setup['hidden_neurons'][1] + \
                     network_setup['hidden_neurons'][1] * 2 + (NUMBER_OF_MODES + 5) * network_setup['state_neurons'] + \
                     network_setup['hidden_neurons'][1] ** 2 + \
                     network_setup['filters'] * network_setup['kernel_size'] ** 2 + network_setup['filters'] * \
                     network_setup['filters1'] * network_setup['kernel_size1'] ** 2

    return number_biases, number_weights, conv_size


NUM_BIASES, NUM_WEIGHTS, CONV_SIZE = get_number_of_parameters(NETWORK_SETUP)
NUM_NN_PARAMS = NUM_BIASES + NUM_WEIGHTS

from torch.utils.data import Dataset

state = torch.load('state.t')
view = torch.load('view.t')
angle = torch.Tensor(np.load('angle.npy'))
latent = torch.load('latent_state.t')


class CustomDataset(Dataset):
    def __init__(self, view, state, latent, angle):
        self.view = view
        self.state = state
        self.latent = latent
        self.angle = angle

    def __len__(self):
        return len(self.angle)

    def __getitem__(self, idx):
        return view[idx].unsqueeze(0), state[idx], latent[idx], angle[idx]


dataset = CustomDataset(view, state, latent, angle)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


class Controller(nn.Module):
    def __init__(self, chromosome):
        '''
        Neural network that controls the rover.

        Initialized from a chromosome specifying the biases, weights and the type of
        pooling layers and activation functions per layer.

        By default, gradient calculation is turned off. If required, please remove

        'self._turn_off_gradients()'

        in the init method.
        '''
        super().__init__()

        # Split up chromosome
        bias_chromosome = chromosome[:NUM_BIASES]
        weight_chromosome = chromosome[NUM_BIASES:NUM_NN_PARAMS]
        self.network_chromosome = chromosome[NUM_NN_PARAMS:]

        # Decode network chromosome
        pooling1 = int(self.network_chromosome[0])
        pooling2 = int(self.network_chromosome[1])
        atype1 = int(self.network_chromosome[2])
        atype2 = int(self.network_chromosome[3])
        atype3 = int(self.network_chromosome[4])
        atype4 = int(self.network_chromosome[5])
        atype5 = int(self.network_chromosome[6])

        # Set up chosen pooling operator and activation functions
        self.pool1 = self._init_pooling_layer(pooling1)
        self.pool2 = self._init_pooling_layer(pooling2)

        self.activation1 = self._init_activation_function(atype1)
        self.activation2 = self._init_activation_function(atype2)
        self.activation3 = self._init_activation_function(atype3)
        self.activation4 = self._init_activation_function(atype4)
        self.activation5 = self._init_activation_function(atype5)

        # Input 1: used rover mode (one-hot), angle to target, distance to target
        # Input 2: map of local landscape
        self.inp = nn.Linear(NUMBER_OF_MODES + 5, NETWORK_SETUP['state_neurons'])
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=NETWORK_SETUP['filters'],
                              kernel_size=NETWORK_SETUP['kernel_size'],
                              stride=NETWORK_SETUP['stride'],
                              dilation=NETWORK_SETUP['dilation'])
        self.conv2 = nn.Conv2d(in_channels=NETWORK_SETUP['filters'],
                               out_channels=NETWORK_SETUP['filters1'],
                               kernel_size=NETWORK_SETUP['kernel_size1'],
                               stride=NETWORK_SETUP['stride'],
                               dilation=NETWORK_SETUP['dilation'])

        # Remaining network
        self.lin2 = nn.Linear(CONV_SIZE, NETWORK_SETUP['hidden_neurons'][0], bias=False)
        self.lin3 = nn.Linear(NETWORK_SETUP['state_neurons'], NETWORK_SETUP['hidden_neurons'][0])

        self.lin4 = nn.Linear(NETWORK_SETUP['hidden_neurons'][0], NETWORK_SETUP['hidden_neurons'][1])
        self.recurr = nn.Linear(NETWORK_SETUP['hidden_neurons'][1], NETWORK_SETUP['hidden_neurons'][1], bias=False)
        self.output = nn.Linear(NETWORK_SETUP['hidden_neurons'][1], 2)

        # self._turn_off_gradients()

        # Load weights and biases from chromosomes
        self._set_weights_from_chromosome(weight_chromosome)
        self._set_biases_from_chromosome(bias_chromosome)

    def forward(self, landscape, state, past_inp):
        '''
        Given the surrounding landscape, rover state and previous network state,
        return:
            - mode control (whether to switch mode)
            - angle control (how to change the orientation of the rover).
            - latent activity of the neural network to be passed to the next iteration.
        '''
        # Add batch and channel dimension if necessary
        if len(landscape.size()) == 2:
            landscape = landscape.unsqueeze(0)
        if len(landscape.size()) == 3:
            landscape = landscape.unsqueeze(0)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        if len(past_inp.size()) == 1:
            past_inp = past_inp.unsqueeze(0)
        # Forward propagation
        # Separate pathways for modalities
        x, y = self.conv(landscape), self.inp(state)
        x, y = self.activation1(x), self.activation2(y)
        x = self.pool1(x)
        x = self.activation3(self.conv2(x))
        x = self.pool2(x).flatten(1)
        # Combine information in common hidden layer
        x = self.lin2(x) + self.lin3(y)
        x = self.activation4(x)
        # Apply another hidden layer + recurrence (memory)
        x = self.lin4(x) + self.recurr(past_inp)
        xlat = self.activation5(x)
        # Get output
        x = self.output(xlat)
        # Rover mode switch command
        mode_command = x[:, 0]
        # Rover orientation
        # angle_command = torch.clamp(x[:,-1], min = -1, max = 1)
        angle_command = x[:, -1]

        return mode_command, angle_command, xlat

    @property
    def chromosome(self):
        '''
        Return chromosome that defines the whole network.
        '''
        chromosomes = {'weights': torch.Tensor([]), 'biases': torch.Tensor([])}
        for param in self.parameters():
            shape = list(param.size())
            if len(shape) > 1:
                whichone = 'weights'
            else:
                whichone = 'biases'
            chromosomes[whichone] = torch.concat([chromosomes[whichone], param.flatten().detach()])

        final_chromosome = list(chromosomes['biases'].detach().numpy()) + \
                           list(chromosomes['weights'].detach().numpy()) + \
                           list(self.network_chromosome)

        return final_chromosome

    def _set_weights_from_chromosome(self, chromosome):
        '''
        Set the weights from a flat vector.
        '''
        if not isinstance(chromosome, torch.Tensor):
            chromosome = torch.Tensor(chromosome)
        prev_slice, next_slice = 0, 0
        for param in self.parameters():
            shape = list(param.size())
            if len(shape) > 1:
                next_slice += np.prod(shape)
                slices = chromosome[prev_slice:next_slice]
                param.data = slices.reshape(shape)
                prev_slice = next_slice
        assert (prev_slice == NUM_WEIGHTS)

    def _set_biases_from_chromosome(self, chromosome):
        '''
        Set the biases from a flat vector.
        '''
        if not isinstance(chromosome, torch.Tensor):
            chromosome = torch.Tensor(chromosome)
        prev_slice, next_slice = 0, 0
        for param in self.parameters():
            shape = list(param.size())
            if len(shape) == 1:
                next_slice += shape[0]
                param.data = chromosome[prev_slice:next_slice]
                prev_slice = next_slice
        assert (prev_slice == NUM_BIASES)

    def _init_pooling_layer(self, chromosome):
        '''
        Convenience function for setting the pooling layer.
        '''
        size = NETWORK_SETUP['pooling_size']
        if chromosome == 0:
            return nn.MaxPool2d(size)
        elif chromosome == 1:
            return nn.AvgPool2d(size)
        else:
            raise Exception('Pooling type with ID {} not implemented.'.format(chromosome))

    def _init_activation_function(self, chromosome):
        '''
        Convenience function for setting the activation function.
        '''
        if chromosome == 0:
            return nn.Sigmoid()
        elif chromosome == 1:
            return nn.Hardsigmoid()
        elif chromosome == 2:
            return torch.tanh
        elif chromosome == 3:
            return nn.Hardtanh()
        elif chromosome == 4:
            return nn.Softsign()
        elif chromosome == 5:
            return nn.Softplus()
        elif chromosome == 6:
            return F.relu
        else:
            raise Exception('Activation type with ID {} not implemented.'.format(chromosome))

    def _turn_off_gradients(self):
        '''
        Convenience function that turns off gradient calculation for all network parameters.
        '''
        for param in self.parameters():
            param.requires_grad = False


chromosome = np.load('../example_rover.npy')
network_chromosome = chromosome[NUM_MODE_PARAMETERS:]
controller = Controller(network_chromosome)

latent_state = torch.zeros(NETWORK_SETUP['hidden_neurons'][1])
lr = 1e-2

optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
loss_fn = nn.MSELoss()

epochs = 3

for e_i in range(epochs):
    controller.train()
    for batch, (v, s, l, a) in enumerate(dataloader):
        switch_mode, angle_change, latent_state = controller(v, s, l)
        loss = loss_fn(angle_change, a)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 30 == 0:
            loss = loss.item()
            print(f"epoch: {e_i}, batch: {batch}, loss: {loss:>7f}")

final_chromosome = controller.chromosome
chromosome[NUM_MODE_PARAMETERS:] = final_chromosome
np.save('test.npy', chromosome)