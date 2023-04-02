# Import Libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Specify transforms using torchvision.transforms as transforms
# library
transformations = transforms.Compose([ #transforms.Compose used to compose multiple transforms together 
                                      #so we can use more than one transformation.
    transforms.Resize(255), #transforms.Resize((255)) resizes the images so the shortest side has a length of 255 pixels. 
                            #The other side is scaled to maintain the aspect ratio of the image.
    transforms.CenterCrop(224), #transforms.CenterCrop(224) crops the center of the image so it is a 
                                #224 by 224 pixels square image.
    transforms.ToTensor(), #transforms.ToTensor() converts the image into numbers.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #subtracts the mean from each value and then 
                                                                                #divides by the standard deviation.
])

# Put into a Dataloader using torch library
# Dataloader is able to spit out random samples of the data, so this model won’t have to deal with the entire dataset every time, to make training more efficient.
# batch_size = 32 means we want to get 32 images at one time.
# images also get shuffled as fed randomly into the AI
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)

# pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True) # choosing a model called densenet161 and specifing that we want it to be pre-trained by setting pretrained=True.
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False # don't calculate the gradients of any parameter since that is only done for training.
    
# A neural network is just a method to find complex patterns and connections between numbers of the input and the output. 
# In this case, it takes the features of the image that were highlighted by the convolution section to determine how likely the image is a certain label.
# Next, to determine the number of outputs, this number should match how many types of images we have. 
# The model will give a list of percentages, each corresponding to how certain the picture is to that label.
# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier.in_features 
num_labels = #PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(nn.Linear(classifier_input, 1024), # nn.Sequential can help us group multiple modules together.
                           nn.ReLU(),
                           nn.Linear(1024, 512), # nn.Linear specifies the interaction between two layers. 
                                                 # We give it 2 numbers, specifying the number of nodes in the two layer.
                           nn.ReLU(), # nn.ReLU is an activation function for hidden layers. 
                                      # Activation functions helps the model learn complex relationships between the input and the output. 
                                      # We use ReLU on all layers except for the output.
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1)) # nn.LogSoftmax is the activation function for the output. 
                                                 # The softmax function turns the outputted numbers into percentages for each label, 
                                                 # and the log function is applied to make it computationally faster. 
                                                 # We must specify that the output layer is a column, so we set dimension equal to 1.
# Replace default classifier with new classifier
model.classifier = classifier

# Training and Evaluating the model
# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

# To evaluate the amount of error our model has, we use nn.NLLLoss. 
# This function takes in the output of our model, for which we used the nn.LogSoftmax function.
# The method of calculating how we adjust our weights and applying it to our weights is called Adam. 
# We use the torch.optim library to use this method and give it our parameters.
# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())

'''Now we train. We want our model to go through the entire dataset multiple times, so we use a for loop. 
Every time it has gone over the entire set of images, it is called an epoch. 
In one epoch we want the model to go through both the training set and the validation set.'''
# clear the adjustments of the weights by declaring optimizer.zero_grad().
''' Then we find the adjustments we need to make to decreases this error by calling loss.backward() and 
    use our optimizer to adjust the weights by calling optimizer.step().'''

# we use torch.exp to reverse the log function.
# .topk gives us the top class that was guessed, and what percentage it guessed it at
# printing the errors for both and the accuracy of the validation set.
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader))
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(val_loader))
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    # Print out the information
    print('Accuracy: ', accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

# set the model for evaluation mode.
model.eval()

# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image
    
    # build a function to use our model to predict the label.
    # Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()
    
    # display the image and turn the image back into an array
    # Show Image
def show_image(image):
    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    
    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))
    
# final processing the guess
# Process Image
image = process_image("root/image1234.jpg")
# Give image to model to predict output
top_prob, top_class = predict(image, model)
# Show the image
show_image(image)
# Print the results
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class  )
