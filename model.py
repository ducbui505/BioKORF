from torch import nn
import torch
import numpy as np

# Gaussian encoder that outputs mean and log variance for a latent distribution
class GaussianParametrizer(nn.Module):
    def __init__(self,
                 feature_dim,
                 latent_dim, 
                 ):
        super(GaussianParametrizer,self).__init__()

        # Linear layer to compute the mean of the latent distribution
        self.h1 = nn.Linear(feature_dim, latent_dim) # Outputs mean

        # Linear layer to compute the log variance  of the latent distribution
        self.h2 = nn.Linear(feature_dim, latent_dim) # Outputs log variance
    
    def forward(self, x):
        # Compute mean vector
        mu = self.h1(x)
        # Compute log variance vector
        log_var = self.h2(x) 
        return mu, log_var

# Attention mechanism
class Attention(nn.Module): 
    def __init__(self,inputdim,heads):
        super(Attention,self).__init__()

        self.inputdim = inputdim
        self.heads = heads

        # Dimension per head for query, key, and value
        self.dq = self.dk = self.dv = inputdim//heads

        # Linear layers to project input into queries, keys, and values
        self.WQ = torch.nn.Linear(self.inputdim, self.dq * self.heads, bias=False)
        self.WK = torch.nn.Linear(self.inputdim, self.dk * self.heads, bias=False)
        self.WV = torch.nn.Linear(self.inputdim, self.dv * self.heads, bias=False)

        # First layer normalization 
        self.LN1 = torch.nn.LayerNorm(inputdim)
        # Feed-forward linear layer
        self.l1 = torch.nn.Linear(inputdim, inputdim)
        # Second layer normalization
        self.LN2 = torch.nn.LayerNorm(inputdim)

    def forward(self,x):

        # Project input to query, key, and value and reshape for attention module
        Q = self.WQ(x).view(-1, self.heads, self.dq).transpose(0, 1) 
        K = self.WK(x).view(-1, self.heads, self.dk).transpose(0, 1)
        V = self.WV(x).view(-1, self.heads, self.dv).transpose(0, 1)

        # calculate transform
        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        QK = torch.nn.Softmax(dim=-1)(QK)
        att = torch.matmul(QK, V)                                 
        
        att = att.transpose(1, 2).reshape(-1, self.heads * self.dv)  

        x = self.LN1(att + x)  # Add & Norm
        output = self.l1(x)
        x = self.LN2(output + x) # Add & Norm
                                                    
        return x



# Autoencoder for concatenated features
class EncoderConnection(nn.Module):
    def __init__(self,drugs_inputdim,sides_inputdim,latent_dim,feature_dim,heads,args):
        super(EncoderConnection,self).__init__()

        # parameters
        self.drugs_inputdim = drugs_inputdim              # Dimension of drug input features
        self.sides_inputdim = sides_inputdim              # Dimension of side effect input features
        self.latent_dim = latent_dim                      # Intermediate latent dimension
        self.feature_dim = feature_dim                    # Bottleneck feature dimension
        self.heads = heads                                # Number of attention heads

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(args.dropout))  # Activation + dropout

        # encoder
        self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim+self.sides_inputdim,self.latent_dim),  # First linear layer
            nn.BatchNorm1d(self.latent_dim),                                     # Batch normalization
            self.reluDrop                                                        # Activation + dropout

        )

        # attention block
        self.attention = Attention(inputdim=self.latent_dim,heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim,self.feature_dim)                    # second linear layer

        # decoder
        self.l3 = nn.Sequential(
            nn.Linear(self.feature_dim,self.latent_dim),                         # linear layer
            nn.BatchNorm1d(self.latent_dim),                                     # Batch normalization
            self.reluDrop,                                                       # Activation + dropout

            nn.Linear(self.latent_dim,self.drugs_inputdim+self.sides_inputdim)   # linear layer
        )


    def forward(self,drugs,sides):
        x = torch.cat((drugs,sides),dim=1) # Concatenate
        x = self.l1(x)                     # Linear + BN + Activation
        x = self.attention(x)              # Attention
        x = self.l2(x)                     # Linear + BN + Activation

        rec_conn = self.l3(x)              # Reconstruct

        return x,rec_conn                  # Return


# Autoencoder for addition features
class EncoderAddition(nn.Module):
    def __init__(self,drugs_inputdim,sides_inputdim,latent_dim,feature_dim,heads,args):
            super(EncoderAddition,self).__init__()

            # parameters
            self.drugs_inputdim = drugs_inputdim            # Combined drug feature dimension (after addition)
            self.sides_inputdim = sides_inputdim            # Combined side effect feature dimension (after addition)
            self.latent_dim = latent_dim                    # Intermediate latent dimension
            self.feature_dim = feature_dim                  # Bottleneck feature dimension
            self.heads = heads                              # Number of attention heads

            # Common activation + dropout block
            self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(args.dropout))

            # encoder
            self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim + self.sides_inputdim, self.latent_dim),  # Linear layer
            nn.BatchNorm1d(self.latent_dim),                                        # Batch normalization
            self.reluDrop                                                           # Activation + dropout
            )
            self.attention = Attention(inputdim=self.latent_dim,heads=self.heads)
            self.l2 = nn.Linear(self.latent_dim,self.feature_dim)

            # decoder
            self.l3 = nn.Sequential(
                nn.Linear(self.feature_dim,self.latent_dim),                         # Linear layer
                nn.BatchNorm1d(self.latent_dim),                                     # Batch normalization
                self.reluDrop,                                                       # Activation + dropout

                nn.Linear(self.latent_dim,self.drugs_inputdim+self.sides_inputdim)   # Linear layer
            )


    def forward(self,drug_features,side_features):

        # Split drug features
        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10,drug11 = drug_features.chunk(11, 1)
        # Split side effect features
        side1, side2, side3, side4 = side_features.chunk(4, 1)
        
        # Perform element-wise addition of drug 
        drugs = drug1+ drug2+ drug3+ drug4+ drug5+ drug6+ drug7+ drug8+ drug9+ drug10+drug11

        # Perform element-wise addition of side effect
        sides = side1+side2+side3+side4

        # Concatenate the summed drug and side effect representations
        add_features = torch.cat((drugs,sides),dim=1)
        x = self.l1(add_features) # Linear + BN + Activation
        x = self.attention(x)     # attention
        x = self.l2(x)            # Linear + BN + Activation

        rec_add = self.l3(x)      # Reconstruct

        return x,rec_add          # Return



# Preprocesses drug and side effect features for cross-product operations
class Preprocess(nn.Module):
    def __init__(self,drug_inputdim,side_inputdim,embeddim,args):
        super(Preprocess,self).__init__()

        # Store input dimensions and embedding output size
        self.drug_inputdim = drug_inputdim
        self.side_inputdim = side_inputdim
        self.embdeddim = embeddim

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(args.dropout)) # Activation + dropout block

        # Define drug preprocessing layers
        self.drug1_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug2_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug3_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug4_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug5_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug6_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug7_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug8_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug9_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug10_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.drug11_pre = nn.Sequential(
            nn.Linear(self.drug_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )

        # Define side effect preprocessing layers
        self.side1_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side2_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side3_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )
        self.side4_pre = nn.Sequential(
            nn.Linear(self.side_inputdim,self.embdeddim),
            nn.BatchNorm1d(self.embdeddim),
            self.reluDrop
        )



    def forward(self,drug_features,side_features):
        # Split the drug feature
        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10,drug11 = drug_features.chunk(11, 1)
        # Split the side effect
        side1, side2, side3, side4 = side_features.chunk(4, 1)
        
        # drug preprocess
        drug1 = self.drug1_pre(drug1)
        drug2 = self.drug2_pre(drug2)
        drug3 = self.drug3_pre(drug3)
        drug4 = self.drug4_pre(drug4)
        drug5 = self.drug5_pre(drug5)
        drug6 = self.drug6_pre(drug6)
        drug7 = self.drug7_pre(drug7)
        drug8 = self.drug8_pre(drug8)
        drug9 = self.drug9_pre(drug9)
        drug10 = self.drug10_pre(drug10)
        drug11 = self.drug11_pre(drug11)
        
        # side preprocess
        side1 = self.side1_pre(side1)
        side2 = self.side2_pre(side2)
        side3 = self.side3_pre(side3)
        side4 = self.side4_pre(side4)

        drugs = [drug1,drug2,drug3,drug4,drug5,drug6,drug7,drug8,drug9,drug10,drug11] # preprocessed drug features
        sides = [side1,side2,side3,side4]                                             # preprocessed side features

        return drugs,sides # Return

# Interation graph features extractor using cross product and CNN layers
class CrossProduction(nn.Module): 
    def __init__(self,cross_dim,feature_dim,input_channel):
        super(CrossProduction,self).__init__()

        # Dimensionality of input embeddings
        self.cross_dim = cross_dim
        self.feature_dim = feature_dim

        # CNN settings
        self.kernel_size = 4
        self.strides = 4
        self.latent_channel = 32
        self.input_channel = input_channel

        # Convolutional layers
        self.cnn = nn.Sequential(
        
            nn.Conv2d(self.input_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides), 
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
        
        )

    def forward(self,drugs,sides):
        # Generate pairwise outer products between drug and side effect features
        crosspro = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                crosspro.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))
        # add channels
        crosspro2d = crosspro[0].view((-1, 1, self.cross_dim, self.cross_dim))

        # Stack all outer products as 2D image-like tensors
        for i in range(1, len(crosspro)):
            crossproEach = crosspro[i].view((-1, 1, self.cross_dim, self.cross_dim))
            crosspro2d = torch.cat([crosspro2d, crossproEach], dim=1)

        # Pass through CNN and flatten
        x = self.cnn(crosspro2d).view((-1,self.feature_dim))
        return x

# Classification module        
class Classifier(nn.Module):
    def __init__(self,latent_dim,classes,args):
        super(Classifier,self).__init__()

        # parameters
        self.latent_dim = latent_dim
        self.classes = classes

        # activation and dropout block
        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01),nn.Dropout(args.dropout))

        # Define the classification head (MLP)
        self.classifier=nn.Sequential(      
            nn.Linear(self.latent_dim,self.latent_dim//2),
            self.reluDrop,

            nn.Linear(self.latent_dim//2,self.classes), 
        ) 


    def forward(self,x):
        x = self.classifier(x) # classification
        return x


# MSSF
class Mulmodel(nn.Module):
    def __init__(self,args):
        super(Mulmodel,self).__init__()

        self.args=args
        self.feature_nums = 4*11  # Number of pairwise cross features
        self.encoderConnection = EncoderConnection(drugs_inputdim=757*11,sides_inputdim=994*4,latent_dim=256,feature_dim=128,heads=4,args=args) # Encoder with concatenation strategy
        self.encoderAddition = EncoderAddition(drugs_inputdim=757,sides_inputdim=994,latent_dim=256,feature_dim=128,heads = 4,args=args) # Encoder with addition strategy
        self.preprocess = Preprocess(drug_inputdim=757,side_inputdim=994,embeddim=128,args=args)       # Preprocess each features of drug and side effect representations
        self.crossProduction = CrossProduction(cross_dim=128,feature_dim=128,input_channel=self.feature_nums) # Cross-product module

        self.attention = Attention(inputdim=128*3,heads=4) # Attention module

        self.gaussian_parametrizer = GaussianParametrizer(feature_dim=128*3,latent_dim=args.gp)  # BVE module

        self.classifier = Classifier(latent_dim=args.gp,classes=5,args=args) # prediction module

    # Reparameterization trick for Bayesian variational inference
    def reparameterize(self, mu, logvar): 
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std) 
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Forward pass of the full model
    def forward(self,drugs,sides,device):     
        drugs = drugs.to(device) # Move drug input to device
        sides = sides.to(device) # Move side effect input to device
        
        feature1,recCon = self.encoderConnection(drugs,sides) # Encoder with concatenation features

        feature2,recAdd = self.encoderAddition(drugs,sides) # Encoder with addition features

        drugs,sides = self.preprocess(drugs,sides) # Preprocess

        feature3 = self.crossProduction(drugs,sides) # Compute pairwise outer-product features

        features = torch.cat((feature1,feature2,feature3),dim=1) # Concatenate all three feature types

        features = self.attention(features) # Fuse features using self-attention

        mu,logvar = self.gaussian_parametrizer(features) # Output parameters for variational inference

        latent_features = self.reparameterize(mu,logvar) # Sample latent vector from variational inference

        results = self.classifier(latent_features) # prediction module

        

        return results,recCon,recAdd,mu,logvar # return results