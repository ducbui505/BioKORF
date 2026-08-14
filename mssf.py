import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model import Mulmodel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, \
precision_score, recall_score, confusion_matrix,cohen_kappa_score,matthews_corrcoef,average_precision_score
from sklearn.preprocessing import label_binarize

# Global random seed
np.random.seed(42)
random.seed(42)

# Obtain relevant drug and side effect matrix data
def read_raw_data(rawdata_dir, data_test):

    # Load SMDsimilarity
    gii = open(rawdata_dir + '/' + 'Text_similarity_one.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()

    # Load SMDexperimental
    gii = open(rawdata_dir + '/' + 'Text_similarity_two.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()

    # Load SMDdatabase
    gii = open(rawdata_dir + '/' + 'Text_similarity_three.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()

    # Load SMDtext
    gii = open(rawdata_dir + '/' + 'Text_similarity_four.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()

    # Load SMDcombined
    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()

    # Load semantic similarity of side effects
    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()

    # Load molecular embeddings for drugs and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    Drug_word_sim = cosine_similarity(Drug_word2vec)

    # Load GloVe word embeddings for side effects and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_sim = cosine_similarity(glove_word)

    # Load drug-target information and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'drug_target.pkl', 'rb')
    drug_target = pickle.load(gii)
    gii.close()
    drug_target_sim = cosine_similarity(drug_target)

    # Load drug fingerprint similarity
    gii = open(rawdata_dir + '/' + 'fingerprint_similarity.pkl', 'rb')
    drug_f_sim = pickle.load(gii)
    gii.close()

    # Load drug-side effect interaction matrix
    gii = open(rawdata_dir + '/' + 'drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    
    # Load drug-pathway-enzyme similarity matrix
    gii = open(rawdata_dir + '/' + 'drug_pathway_enzyme_similarity.pkl', 'rb')
    drug_p_e_sim = pickle.load(gii)
    gii.close()

    # Remove test set information from the training matrix
    for i in range(data_test.shape[0]):
        drug_side[data_test[i, 0], data_test[i, 1]] = 0

    # Compute cosine similarity over the remaining drug-side effect frequency matrix, SMD_DIPF
    drug_side_sim = cosine_similarity(drug_side)

    # Generate binary label matrix from the frequency matrix
    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1

    # Compute cosine similarity over the binary label matrix, SMD_DIPA
    drug_side_label_sim = cosine_similarity(drug_side_label)

    # Assemble drug features
    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(Drug_word_sim)
    drug_features.append(drug_target_sim)
    drug_features.append(drug_f_sim)
    drug_features.append(drug_side_sim)
    drug_features.append(drug_side_label_sim)
    drug_features.append(drug_p_e_sim)

    # Compute similarity matrices for side effects,SME_DIPA,SME_DIPF
    side_drug_sim = cosine_similarity(drug_side.T)
    side_drug_label_sim = cosine_similarity(drug_side_label.T)

    # Assemble side features
    side_features.append(effect_side_semantic)
    side_features.append(side_glove_sim)
    side_features.append(side_drug_sim)
    side_features.append(side_drug_label_sim)

    # return drug and side features
    return drug_features, side_features

# Get the drug-side effect indices for the training and test sets,
# as well as the frequency information of the corresponding drug-side effect pairs
def fold_files(data_train, data_test,args):

    # Get the path to the raw data directory from arguments
    rawdata_dir = args.rawpath

    # Convert input training and testing data to NumPy arrays
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    # Obtain relevant drug and side effect matrix data
    drug_features, side_features = read_raw_data(rawdata_dir, data_test)

    # Initialize the drug feature matrix using the first feature
    drug_features_matrix = drug_features[0]

    # Concatenate all drug feature matrices horizontally to form a full feature matrix
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    # Initialize the side effect feature matrix using the first feature
    side_features_matrix = side_features[0]

    # Concatenate all side effect feature matrices horizontally to form a full feature matrix
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    # Extract test drug and side effect features based on test indices
    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]

    # Extract test frequencies
    f_test = data_test[:, 2]

    # Extract training drug and side effect features based on training indices
    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]

    # Extract training frequencies
    f_train = data_train[:, 2]

    # Return train/test features and frequencies
    return drug_test, side_test, f_test, drug_train, side_train, f_train

def train_test(data_train, data_test, args):
    # Prepare training and test sets by extracting corresponding drug/side effect features and frequencies
    drug_test, side_test, f_test, drug_train, side_train, f_train = fold_files(data_train, data_test,args)

    # Create training dataset as a PyTorch TensorDataset
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                              torch.FloatTensor(f_train))
    # Create testing dataset
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                             torch.FloatTensor(f_test))
    
    # Wrap datasets in DataLoader for batch training and evaluation
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          pin_memory=False)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        pin_memory=False)

    # Set the runtime device for the program
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Set GPU device index
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu") # Select CUDA if available

    # Instantiate the model and move it to the chosen device
    model = Mulmodel(args).to(device)

    # Set optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize evaluation metric
    acc_tested = 0
    wf1_tested =  0
    maf1_tested = 0
    ka_tested = 0
    mcc_tested = 0
    maprec_tested = 0
    mareca_tested = 0
    maaupr_tested = 0
    
    # Model training and testing
    for epoch in range(1, args.epochs + 1):
        # ----------- Training step -----------
        train(model, _train, optimizer, device)

        # ----------- Evaluation on train and test sets -----------
        acc_tr,weighted_f1_tr,macro_f1_tr,kappa_tr,mcc_tr,rating_tr,pred_tr,macro_prec_tr,macro_recall_tr,macro_aupr_tr = test(model,_train,device)
        acc_te,weighted_f1_te,macro_f1_te,kappa_te,mcc_te,rating_te,pred_te,macro_prec_te,macro_recall_te,macro_aupr_te = test(model,_test,device)

        # If current test accuracy is best so far, update tracked metrics
        if  acc_te>acc_tested:            
            acc_tested = acc_te
            wf1_tested =  weighted_f1_te
            maf1_tested = macro_f1_te
            ka_tested = kappa_te
            mcc_tested = mcc_te
            maprec_tested = macro_prec_te
            mareca_tested = macro_recall_te
            maaupr_tested = macro_aupr_te

        # Print training results of current epoch
        print("Epoch: %d <Train> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr:%.5f" %(
        epoch, acc_tr,weighted_f1_tr,macro_f1_tr,kappa_tr,mcc_tr,macro_prec_tr,macro_recall_tr,macro_aupr_tr))

        # Print test results of current epoch
        print("Epoch: %d <Test> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr:%.5f" %(
        epoch, acc_te,weighted_f1_te,macro_f1_te,kappa_te,mcc_te,macro_prec_te,macro_recall_te,macro_aupr_te))


    # Print best test results achieved
    print(" <Best Test> acc_tr: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr: %.5f" % (
        acc_tested,wf1_tested,maf1_tested,ka_tested,mcc_tested,maprec_tested,mareca_tested,maaupr_tested))

    # Return the performance metrics
    return acc_tested,wf1_tested,maf1_tested,ka_tested,mcc_tested,maprec_tested,mareca_tested,maaupr_tested


# KL divergence calculation function
def kl_func(mu,logvar):
    return - 0.5 * (1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)

# Loss computation
def calculate_loss(multi_pred,recCon,recAdd,mu, logvar,batch_ratings,batch_drug,batch_side,device):
    
    # Compute KL divergence
    kl_div = kl_func(mu, logvar).mean()

    # Define multi-class classification loss
    loss_func = nn.CrossEntropyLoss() 

    # Convert frequencies to integer class indices: -> {0,1,2,3,4}
    multi_labels = (batch_ratings.long()-1).to(device)

    # Concatenate drug and side effect vectors 
    batch_vec = torch.cat((batch_drug, batch_side), dim=1)

    # Split the drug features into 11 different parts
    drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10, drug11 = batch_drug.chunk(11, 1)

    # Split the side effect features into 4 parts
    side1, side2, side3, side4 = batch_side.chunk(4, 1)

    # Sum all drug features into a single vector
    drugs = drug1+ drug2+ drug3+ drug4+ drug5+ drug6+ drug7+ drug8+ drug9+ drug10 +drug11
    # Sum all side features  into a single vector
    sides = side1+side2+side3+side4

    # Concatenate aggregated drug and side effect features
    add_features = torch.cat((drugs,sides),dim=1)

    # Classification loss
    multi_loss = loss_func(multi_pred,multi_labels)

    # Define reconstruction loss using MSE without reduction for flexibility
    reconst_loss = nn.MSELoss(reduction='none')

    # Compute concatenation reconstruction loss
    rec_loss1 = reconst_loss(recCon,batch_vec.to(device)).sum(dim=-1).mean()
    # Compute addition reconstruction loss
    rec_loss2 = reconst_loss(recAdd,add_features.to(device)).sum(dim=-1).mean()

    # Compute the total loss
    Loss = multi_loss+0.001*kl_div+0.0001*rec_loss1+0.0001*rec_loss2
   
    return Loss


# Model training
def train(model, train_loader, optimizer, device):

    # Set the model to training mode
    model.train()
    avg_loss = 0.0

    # Iterate over training data loader
    for i, data in enumerate(train_loader, 0):

        # Unpack the batch: drug features, side effect features, and corresponding frequencies
        batch_drug, batch_side, batch_ratings = data
       
        # Clear gradients from the previous step
        optimizer.zero_grad()

        # model outputs 
        multi_pred,recCon,recAdd,mu,logvar = model(batch_drug,batch_side, device)
        
        # Calculate the loss
        loss = calculate_loss(multi_pred,recCon,recAdd,mu, logvar,batch_ratings,batch_drug,batch_side,device)

        # Backward pass to compute gradients
        loss.backward(retain_graph = True)

        # Update model parameters
        optimizer.step()

        # Accumulate loss
        avg_loss += loss.item()

    return 0

# Model testing
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    pred_all = []           # List to store predicted labels
    multi_label_all = []    # List to store true labels
    prob_all = []           # List to store predicted probabilities

    # Iterate over test data loader
    for test_drug, test_side, test_ratings in test_loader:

        # Obtain predicted data
        multi_pred,recCon,recAdd,mu, logvar = model(test_drug,test_side, device)

        # Classify the predicted data, and in classification, the indices will be in the form of 0, 1, 2, 3, 4
        pred = torch.argmax(multi_pred.cpu(), dim=1).numpy() 
        pred_all.append(list(pred)) # Store predictions

        # True labels, and subtract 1 from the true labels to match the 0, 1, 2, 3, 4 format
        multi_label_all.append(list((test_ratings.long()-1).cpu().numpy())) 

        # Get predicted probabilities and calculate AUPR
        softmax = torch.nn.Softmax(dim=1)
        pred_prob = softmax(multi_pred).cpu().detach().numpy()  # Convert predictions to probabilities
        prob_all.append(pred_prob)


       
    pred_all = np.array(sum(pred_all, []))  # Flatten the list of predictions into a single array
    multi_label_all = np.array(sum(multi_label_all, []))  # Flatten the list of true labels into a single array
    prob_all = np.vstack(prob_all)  # Stack the probability arrays vertically

    # Compute corresponding metrics
    acc = accuracy_score(multi_label_all, pred_all)  # Accuracy
    weighted_f1 = f1_score(multi_label_all, pred_all, average="weighted")  # Weighted F1-score
    macro_f1 = f1_score(multi_label_all, pred_all, average="macro")  # Macro F1-score
    kappa = cohen_kappa_score(multi_label_all, pred_all)  # Cohen's kappa
    mcc = matthews_corrcoef(multi_label_all, pred_all)  # Matthews correlation coefficient

    # Additional metrics: macro precision, recall, and AUPR
    macro_precision = precision_score(multi_label_all, pred_all, average='macro')
    macro_recall = recall_score(multi_label_all, pred_all, average='macro')
    multi_label_all_onehot = label_binarize(multi_label_all, classes=[0, 1, 2, 3, 4])
    macro_aupr = average_precision_score(multi_label_all_onehot, prob_all, average='macro')
    
    # Return all computed metrics and prediction outputs
    return acc,weighted_f1,macro_f1,kappa,mcc,multi_label_all,pred_all,macro_precision,macro_recall,macro_aupr

# Ten-fold cross-validation
def ten_fold(args):
    rawpath = args.rawpath # Directory path for raw data
    gii = open(rawpath+'/drug_side.pkl', 'rb') # Load the drug-side effect frequency matrix
    drug_side = pickle.load(gii)
    gii.close()

    # Benchmark dataset data
    final_positive_sample = Extract_positive_negative_samples(drug_side)
    final_sample = final_positive_sample

    # Split data into feature variables (X) and frequencies variables (y)
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    
    # Create the dataset
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1])) # (drug, side effect) pair
        data_y.append((int(float(X[i, 2])))) # frequencies
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1

    # Data split
    kfold = StratifiedKFold(10,random_state=42,shuffle=True)

    # Lists to store results for different evaluation metrics
    total_acc, total_wf1, total_maf1,total_kappa,total_mcc,total_prec,total_reca,total_aupr = [], [], [], [], [], [], [], []

    # Ten-fold cross-validation experiment
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        # Convert the dataset into numpy array
        data = np.array(data)

        # Train and test the model on the current fold
        acc,weighted_f1,macro_f1,kappa,mcc,macro_precision,macro_recall,macro_aupr  = train_test(data[train].tolist(), data[test].tolist(), args)

        # Store results of the current fold
        total_acc.append(acc)
        total_wf1.append(weighted_f1)
        total_maf1.append(macro_f1)
        total_kappa.append(kappa)
        total_mcc.append(mcc)
        total_prec.append(macro_precision)
        total_reca.append(macro_recall)
        total_aupr.append(macro_aupr)


        # Print the average results of all folds so far
        print("==================================fold {} end".format(fold))
        print('Total_acc:')
        print(np.mean(total_acc))
        print('Total_weighted_f1:')
        print(np.mean(total_wf1))
        print('Total_macro_f1:')
        print(np.mean(total_maf1))
        print('Total_kappa:')
        print(np.mean(total_kappa))
        print('Total_mcc:')
        print(np.mean(total_mcc))

        print('Total_precision:')
        print(np.mean(total_prec))
        print('Total_recall:')
        print(np.mean(total_reca))
        print('Total_aupr:')
        print(np.mean(total_aupr))

        # Save results to a text file after each fold
        with open("./result.txt",'a') as f:
            print("fold:"+str(fold),file=f)

            print('Total_acc:',file=f)
            print(np.mean(total_acc),file=f)

            print('Total_weighted_f1:',file=f)
            print(np.mean(total_wf1),file=f)

            print('Total_macro_f1:',file=f)
            print(np.mean(total_maf1),file=f)

            print('Total_kappa:',file=f)
            print(np.mean(total_kappa),file=f)

            print('Total_mcc:',file=f)
            print(np.mean(total_mcc),file=f)

            print('Total_precision:',file=f)
            print(np.mean(total_prec),file=f)

            print('Total_recall:',file=f)
            print(np.mean(total_reca),file=f)
            
            print('Total_aupr:',file=f)
            print(np.mean(total_aupr),file=f)

            print("\n",file=f) # Add a newline for separation between folds
        fold += 1 # Increment the fold counter

        sys.stdout.flush()


# Benchmark dataset data extraction
def Extract_positive_negative_samples(DAL):
    k = 0 # Initialize counter for indexing the interaction_target array
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int) # Initialize a zero matrix

    # Iterate through the DAL matrix and store the indices and values in interaction_target
    for i in range(DAL.shape[0]):  # Loop through each row (drug)
        for j in range(DAL.shape[1]):  # Loop through each column (side effect)
            interaction_target[k, 0] = i  # Store the drug index
            interaction_target[k, 1] = j  # Store the side effect index
            interaction_target[k, 2] = DAL[i, j]  # Store the frequency value 
            k = k + 1  # Increment the counter

    # sort all datas
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])

    # obtain benchmark dataset
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]

    return final_positive_sample  # Return the benchmark dataset

# Main entry point for model training and evaluation
def main():

    # =======================
    # Argument Configuration
    # =======================

    # Model and training parameters
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--dropout', type = float, default = 0.4,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--gp', type = int, default = 64,
                        metavar = 'gp', help = 'hyper_gauss')
    
    # Batch size settings
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'hh',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='./Datas',
                        metavar='STRING', help='rawpath')
    
    # Parse arguments
    args = parser.parse_args()


    # =======================
    # Display Hyperparameters
    # =======================
    print('-------------------- Hyperparams --------------------')
    print('learning rate: ' + str(args.lr))
    print('dropout rate: ' + str(args.dropout))
    print('batch_size: ' + str(args.batch_size))
    print('dimension of Bayesian: ' + str(args.gp))
    print('weight decay: ' + str(args.weight_decay))

    # =======================
    # Run Ten-Fold Cross-Validation
    # =======================
    ten_fold(args)

# Run main function
if __name__ == "__main__":
    main()
