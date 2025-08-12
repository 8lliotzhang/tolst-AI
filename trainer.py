import torch
from torch import nn
from tqdm import tqdm 
from torch.nn.utils import clip_grad_norm_

START_STRING = "Pierre"

learning_rate = 0.001
# step size - how much weights get moved around during backpropagation

num_epochs = 1000 
# number of training rounds to put it thru

seq_length = 200
# number of chars per training sequence

batch_size = 32
# number of simultaneous sequences getting processed

clip_value = 1.0
# prevents "exploding gradients" 
# PS gradient is just a vector pointing towards the steepest increase in loss function
# gradient tells model how to adjust the weights to reduce loss the most
# exponential gradient growth --> instability and learning failure caused by exponential gradient growth
# this leads to dramatic weight changes which basically ruin the model
# smaller = safer but slower

input_size = 256
# size of one of three dimensions used by the embedding layer (other two are batch_size, seq_len)

hidden_size = 512
# dimension of the hidden states (LSTM converts the input to this dimension)

dropout = 0.2
# probability any connection gets reset/excluded. 
# This is called regularization; reduces overfitting.
# should only happen if layers > 1!

num_layers = 2
# number of LSTM stacks (it puts the output back into the LSTM)

temperature = 0.8
# how CRAZY the model gets. lower = less crazy

device = torch.device(
                    'cuda' if torch.cuda.is_available() else 
                    'mps' if torch.backends.mps.is_available() else 
                    'cpu'
                )

assert torch.cuda.is_available(), "CUDA GPU not available!"
device = torch.device("cuda")
# device we r training on

# --------------------------------------------------------------------------------- #
# END VARS BLOCK
# --------------------------------------------------------------------------------- # 
# START CODE AND STUFF
# --------------------------------------------------------------------------------- # 



#read the text file.
with open('pg2600.txt', 'r', encoding='utf-8') as f:
    text = f.read()


#create the list of chars used in War and Peace and assign them each a number so you know what to do with it
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# turn each char in text into the index 
text_as_int = [char_to_idx[c] for c in text]

# split up the text into batches of data
def create_batches(data, batch_size, seq_length):
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = torch.tensor(data, dtype=torch.long)
    
    # Reshape into (batch_size, num_batches * seq_length)
    data = data.view(batch_size, -1)
    
    # Create batches
    for i in range(0, data.size(1) - seq_length, seq_length):
        x = data[:, i:i+seq_length]
        y = data[:, i+1:i+seq_length+1]
        yield x, y

# defining the LSTM
class myLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        #initialize LSTM with certain properties. just defining everything
        
        super(myLSTM, self).__init__()
        self.input_size = input_size

        self.hidden_size = hidden_size 
        
        self.num_layers = num_layers
        
        self.dropout = dropout
    
        self.embedding = nn.Embedding(vocab_size, self.input_size)
        # transforms the char_to_idx index for each char into a self.hidden_size dimension vector.
        # embeddings expands each index to a vector representation of the char
        # related vectors get close together in the 256-dimension map.
        # size of dict: vocab_size; dimension: 256
        
        self.lstm = nn.LSTM(
                            input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers=self.num_layers, 
                            dropout=self.dropout, 
                            batch_first=True 
                            # shape is [batch_size, seq_len, features]
                            )
        # takes in a [batch_size, seq_len, 256] input, outputs [batch_size, seq_len, 512]
        
        self.fc = nn.Linear(self.hidden_size, vocab_size)
        # Fully Connected. turns the hidden layer vector back into a vocab_size dimension vector 
        # which can then be transformed into chars :D

    def forward(self, x, hidden):
    #forward pass (putting above things together to predict next char)
        
        x = self.embedding(x)
        # so x AT START is actually just the char_to_idx thing
        # see notes on self.embedding above
        # x gets transformed from [batch_size, seq_length] --> [batch_size, seq_length, hidden_size] - Because each char is now a 512-dimension vector representation!!!
        
        out,hidden = self.lstm(x,hidden)
        # x is a 3D tensor: [batch_size, sequence length, input_size] and represents the hidden state. Used to predict the next chars.
        # hidden is two tuples: 
            # h_n - Short term memory - [num_layers, batch_size, hidden_size]
            # c_n - Long term memory - same shape

        out = out.reshape(-1,self.hidden_size)
        # ok, so originally out is a 3D tensor [batch_size, sequence_length, hidden_size]
        # this operation converts it into a 2D tensor [batch*sequence, hidden]
        # this is because self.fc requires a 2D input [# examples, vocab size]
        # -1 means "infer dimension" - so it just sorts it out for us ig
        
        out = self.fc(out)
        # see notes on self.fc above
        return out, hidden

    def init_hidden(self, batch_size):
    # sets up the initial hidden and cell states for LSTM (same shape)
    # returns two tuples, filled with 0
        device = next(self.parameters()).device
        # select the right device to train on

        return(
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), # h_0
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)  # c_0
            )
        
        # set all the "neurons" as 0 for now

## why is this block in the middle of these functions? oh well!

model = myLSTM(vocab_size, hidden_size, num_layers).to(device)
# ...and we create a myLSTM with the variables at the top.

criterion = nn.CrossEntropyLoss()
# calculate loss; -log(probability of correct prediction)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Adam optimizer updates the learning rate and paramaters 
# uses something called momentum and variance? cool

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min", # watches for decrease in loss
    factor = 0.5, # learning rate is 50% of previous value if no improvement happens
    patience = 10, # waits this many epochs before triggering
    # verbose=True 
    # apparently only on versions 1.4 and later. Too bad!
    )


def generate_text(model, start_string, length, temp):
    model.eval()
    # set to evaluation mode; disable dropout, BatchNorm, self.training = False

    chars = [ch for ch in start_string]
    # split the start_string into its individual chars
    
    input_seq = torch.tensor([[char_to_idx[c] for c in chars]], dtype = torch.long).to(device)
    # turn the array of chars into an array of indices

    hidden = model.init_hidden(1)
    # initiallizes the hidden state

    with torch.no_grad():
        for _ in range (length):
            output, hidden = model(input_seq, hidden)
            output_dist = output[-1, :].div(temp).exp()

            top_i = torch.multinomial(output_dist, 1)[0]
            # what is the most popular index prediction for the next word???
            
            predicted_char = idx_to_char[top_i.item()]
            # convert this index to the char we want

            chars.append(predicted_char)
            # add the char to chars.

            input_seq = torch.tensor([[top_i]], dtype=torch.long).to(device)
            # update the input sequence and repeat loop
    
    return "".join(chars)


best_loss = float('inf')
print(f"Training on: {device}") 
# just check were not on CPU for some reason
# omg. We ARE on cpu for some reason
# not anymore! whoops so basically i forgot to import the package which lets it use cuda

def train():
    for epoch in tqdm(range(num_epochs), desc = "Training", unit="epoch"):
        hidden = model.init_hidden(batch_size)
        total_loss = 0
        batch_count = 0
        # precompute total number of batches
        # sets up brand new state (all zeroes), resets loss tracking  

        for x,y in create_batches(text_as_int, batch_size, seq_length):
            x, y = x.to(device), y.to(device)
            # sends data to training device
            # x contains input sequences
            # y, target sequences (next chars)
            
            #forward pass
            outputs, hidden = model(x, hidden)
            #outputs: predictions for next char
            loss = criterion(outputs, y.reshape(-1))
            # loss: prediction error measurement

            # backward pass
            optimizer.zero_grad()
            # clear old gradients
            
            loss.backward()
            # does something called backpropagation

            clip_grad_norm_(model.parameters(), clip_value)
            # stop exploding gradients
            
            optimizer.step()
            # update weights
            
            hidden = (hidden[0].detach(), hidden[1].detach())
            # hidden state is detached.

            total_loss += loss.item()
            batch_count += 1
        # track batches and total loss

        avg_loss = total_loss / batch_count 
        # compute avg loss over all

        
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}')
        # outputs loss of current epoch

        scheduler.step(avg_loss)
        # adjust learning rate if no improvement in loss

        

        if (epoch + 1) % 10 == 0:
            model.eval() # eval mode for text generation - no dropout, BatchNorm
            sample = generate_text(model, start_string=START_STRING,length=500, temp = temperature)
            model.train() # back to training now
            print(f"\nSample:\n{sample}\n") # get text :o
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}") # just check the learning rate

    # model.eval() line not needed since generate_text already sets it to eval mode
    print(generate_text(model, start_string=START_STRING,length=500))        
    torch.save(model.state_dict(), 'best_model.pth')
    # save the model and really hope loss didnt go up like right before
    print("new model saved :D")

if __name__ == "__main__":
    train()
    # we only wanna run this if the file itself is run, not imported for textgen.
